#include "rocc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SA_OPCODE 1
#define SA_FUNCT_CONFIG 0
#define SA_FUNCT_RUN 1

#define SA_ROWS 2
#define SA_COLS 2
#define SA_MAX_K 64

#define MAX_M 8
#define MAX_N 8
#define MAX_K 80

typedef struct {
  int M;
  int N;
  int K;
} gemm_case_t;

static inline uint64_t read_cycles(void) {
  uint64_t cycles;
  asm volatile("fence; rdcycle %0" : "=r"(cycles));
  return cycles;
}

static inline uint64_t pack2_u16(uint16_t v0, uint16_t v1) {
  return ((uint64_t)v1 << 16) | (uint64_t)v0;
}

static inline uint64_t sa_config(const uint64_t *a_stream, const uint64_t *b_stream) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, a_stream, b_stream, SA_FUNCT_CONFIG);
  return rd;
}

static inline uint64_t sa_run(uint64_t *c_words, uint64_t k) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, c_words, k, SA_FUNCT_RUN);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static void sw_gemm_u16(
    const uint16_t A[MAX_M][MAX_K],
    const uint16_t B[MAX_K][MAX_N],
    uint32_t C[MAX_M][MAX_N],
    int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      uint32_t acc = 0;
      for (int k = 0; k < K; k++) {
        acc += (uint32_t)A[i][k] * (uint32_t)B[k][j];
      }
      C[i][j] = acc;
    }
  }
}

static int hw_gemm_u16_tiled(
    const uint16_t A[MAX_M][MAX_K],
    const uint16_t B[MAX_K][MAX_N],
    uint32_t C[MAX_M][MAX_N],
    int M, int N, int K) {
  static uint64_t a_stream[SA_MAX_K] __attribute__((aligned(64)));
  static uint64_t b_stream[SA_MAX_K] __attribute__((aligned(64)));
  static uint64_t c_words[SA_ROWS * SA_COLS] __attribute__((aligned(64)));

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  for (int m0 = 0; m0 < M; m0 += SA_ROWS) {
    for (int n0 = 0; n0 < N; n0 += SA_COLS) {
      uint32_t tile_acc[SA_ROWS * SA_COLS] = {0, 0, 0, 0};

      for (int k0 = 0; k0 < K; k0 += SA_MAX_K) {
        int k_chunk = K - k0;
        if (k_chunk > SA_MAX_K) {
          k_chunk = SA_MAX_K;
        }

        for (int kk = 0; kk < k_chunk; kk++) {
          uint16_t a0 = (m0 + 0 < M) ? A[m0 + 0][k0 + kk] : 0;
          uint16_t a1 = (m0 + 1 < M) ? A[m0 + 1][k0 + kk] : 0;
          uint16_t b0 = (n0 + 0 < N) ? B[k0 + kk][n0 + 0] : 0;
          uint16_t b1 = (n0 + 1 < N) ? B[k0 + kk][n0 + 1] : 0;
          a_stream[kk] = pack2_u16(a0, a1);
          b_stream[kk] = pack2_u16(b0, b1);
        }

        uint64_t cfg_rc = sa_config(a_stream, b_stream);
        uint64_t run_rc = sa_run(c_words, (uint64_t)k_chunk);
        if (cfg_rc != 0 || run_rc != 0) {
          return -1;
        }

        for (int idx = 0; idx < SA_ROWS * SA_COLS; idx++) {
          tile_acc[idx] += (uint32_t)c_words[idx];
        }
      }

      for (int i = 0; i < SA_ROWS; i++) {
        for (int j = 0; j < SA_COLS; j++) {
          int m = m0 + i;
          int n = n0 + j;
          if (m < M && n < N) {
            C[m][n] = tile_acc[i * SA_COLS + j];
          }
        }
      }
    }
  }
  return 0;
}

int main(void) {
  static uint16_t A[MAX_M][MAX_K] __attribute__((aligned(64)));
  static uint16_t B[MAX_K][MAX_N] __attribute__((aligned(64)));
  static uint32_t C_sw[MAX_M][MAX_N];
  static uint32_t C_hw[MAX_M][MAX_N];

  const gemm_case_t tests[] = {
    {3, 4, 5},
    {5, 3, 7},
    {6, 6, 17},
    {7, 5, 33},
    {8, 8, 70}
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));

  srand(1);
  printf("=== Systolic GEMM Random Test ===\n");
  printf("CSV_HEADER,case,M,N,K,sw_cycles,hw_cycles,mismatches\n");

  int total_mismatches = 0;
  for (int t = 0; t < ntests; t++) {
    const int M = tests[t].M;
    const int N = tests[t].N;
    const int K = tests[t].K;

    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        A[i][k] = (uint16_t)(rand() % 16);
      }
    }
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        B[k][j] = (uint16_t)(rand() % 16);
      }
    }

    uint64_t sw_start = read_cycles();
    sw_gemm_u16(A, B, C_sw, M, N, K);
    uint64_t sw_cycles = read_cycles() - sw_start;

    uint64_t hw_start = read_cycles();
    int hw_rc = hw_gemm_u16_tiled(A, B, C_hw, M, N, K);
    uint64_t hw_cycles = read_cycles() - hw_start;

    int mismatches = 0;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        if (C_sw[i][j] != C_hw[i][j]) {
          if (mismatches < 8) {
            printf("Mismatch case=%d at (%d,%d): sw=%u hw=%u\n",
                   t, i, j, C_sw[i][j], C_hw[i][j]);
          }
          mismatches++;
        }
      }
    }

    if (hw_rc != 0) {
      printf("Hardware return code error on case %d\n", t);
      mismatches += 1;
    }

    total_mismatches += mismatches;
    printf("CSV_DATA,%d,%d,%d,%d,%lu,%lu,%d\n",
           t, M, N, K,
           (unsigned long)sw_cycles,
           (unsigned long)hw_cycles,
           mismatches);
  }

  if (total_mismatches == 0) {
    printf("PASS: all randomized GEMM cases matched.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }
  return 0;
}
