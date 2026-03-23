#include "rocc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SA_OPCODE 1
#define SA_FUNCT_CONFIG 0
#define SA_FUNCT_RUN 1

#define SA_ROWS 2
#define SA_COLS 2

#define MAX_DIM 256
#define CPU_BLOCK 32

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

static void cpu_gemm_blocked_u16(
    const uint16_t A[MAX_DIM][MAX_DIM],
    const uint16_t B[MAX_DIM][MAX_DIM],
    uint32_t C[MAX_DIM][MAX_DIM],
    int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  for (int ii = 0; ii < M; ii += CPU_BLOCK) {
    int i_end = ii + CPU_BLOCK;
    if (i_end > M) i_end = M;
    for (int kk = 0; kk < K; kk += CPU_BLOCK) {
      int k_end = kk + CPU_BLOCK;
      if (k_end > K) k_end = K;
      for (int jj = 0; jj < N; jj += CPU_BLOCK) {
        int j_end = jj + CPU_BLOCK;
        if (j_end > N) j_end = N;
        for (int i = ii; i < i_end; i++) {
          for (int k = kk; k < k_end; k++) {
            uint32_t a = (uint32_t)A[i][k];
            for (int j = jj; j < j_end; j++) {
              C[i][j] += a * (uint32_t)B[k][j];
            }
          }
        }
      }
    }
  }
}

static int hw_gemm_tiled_u16(
    const uint16_t A[MAX_DIM][MAX_DIM],
    const uint16_t B[MAX_DIM][MAX_DIM],
    uint32_t C[MAX_DIM][MAX_DIM],
    int M, int N, int K) {
  static uint64_t a_stream[MAX_DIM] __attribute__((aligned(64)));
  static uint64_t b_stream[MAX_DIM] __attribute__((aligned(64)));
  static uint64_t c_words[SA_ROWS * SA_COLS] __attribute__((aligned(64)));

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  for (int m0 = 0; m0 < M; m0 += SA_ROWS) {
    for (int n0 = 0; n0 < N; n0 += SA_COLS) {
      for (int k = 0; k < K; k++) {
        uint16_t a0 = (m0 + 0 < M) ? A[m0 + 0][k] : 0;
        uint16_t a1 = (m0 + 1 < M) ? A[m0 + 1][k] : 0;
        uint16_t b0 = (n0 + 0 < N) ? B[k][n0 + 0] : 0;
        uint16_t b1 = (n0 + 1 < N) ? B[k][n0 + 1] : 0;
        a_stream[k] = pack2_u16(a0, a1);
        b_stream[k] = pack2_u16(b0, b1);
      }

      if (sa_config(a_stream, b_stream) != 0) {
        return -1;
      }
      if (sa_run(c_words, (uint64_t)K) != 0) {
        return -1;
      }

      for (int i = 0; i < SA_ROWS; i++) {
        for (int j = 0; j < SA_COLS; j++) {
          int m = m0 + i;
          int n = n0 + j;
          if (m < M && n < N) {
            C[m][n] = (uint32_t)c_words[i * SA_COLS + j];
          }
        }
      }
    }
  }
  return 0;
}

static int count_mismatches(
    const uint32_t A[MAX_DIM][MAX_DIM],
    const uint32_t B[MAX_DIM][MAX_DIM],
    int M, int N) {
  int mismatches = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (A[i][j] != B[i][j]) {
        mismatches++;
      }
    }
  }
  return mismatches;
}

int main(void) {
  static uint16_t A[MAX_DIM][MAX_DIM] __attribute__((aligned(64)));
  static uint16_t B[MAX_DIM][MAX_DIM] __attribute__((aligned(64)));
  static uint32_t C_cpu[MAX_DIM][MAX_DIM];
  static uint32_t C_hw[MAX_DIM][MAX_DIM];

  const gemm_case_t cases[] = {
    {128, 128, 128},
    {256, 256, 256},
  };
  const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

  srand(1);
  printf("=== Systolic GEMM Benchmark ===\n");
  printf("CSV_HEADER,case,M,N,K,cpu_cycles,hw_cycles,mismatches\n");

  for (int c = 0; c < ncases; c++) {
    const int M = cases[c].M;
    const int N = cases[c].N;
    const int K = cases[c].K;

    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        A[i][k] = (uint16_t)(rand() & 0xF);
      }
    }
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        B[k][j] = (uint16_t)(rand() & 0xF);
      }
    }

    uint64_t cpu_start = read_cycles();
    cpu_gemm_blocked_u16(A, B, C_cpu, M, N, K);
    uint64_t cpu_cycles = read_cycles() - cpu_start;

    uint64_t hw_start = read_cycles();
    int hw_rc = hw_gemm_tiled_u16(A, B, C_hw, M, N, K);
    uint64_t hw_cycles = read_cycles() - hw_start;

    int mismatches = (hw_rc == 0) ? count_mismatches(C_cpu, C_hw, M, N) : -1;

    printf("CSV_DATA,%d,%d,%d,%d,%lu,%lu,%d\n",
           c, M, N, K,
           (unsigned long)cpu_cycles,
           (unsigned long)hw_cycles,
           mismatches);
  }

  return 0;
}
