#include "rocc.h"
#include <stdint.h>
#include <stdio.h>

#define SA_OPCODE 1
#define SA_FUNCT_CONFIG 0
#define SA_FUNCT_RUN 1

#define SA_ROWS 2
#define SA_COLS 2

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

int main(void) {
  // 2xK times Kx2 -> 2x2 output tile.
  // Choose K > maxK (default 64) to exercise hardware-side chunking.
  enum { K = 70 };

  static uint16_t A[SA_ROWS][K];
  static uint16_t B[K][SA_COLS];

  uint32_t sw[SA_ROWS][SA_COLS] = {{0}};
  uint32_t hw[SA_ROWS][SA_COLS] = {{0}};

  static uint64_t a_stream[K] __attribute__((aligned(64)));
  static uint64_t b_stream[K] __attribute__((aligned(64)));
  static uint64_t c_words[SA_ROWS * SA_COLS] __attribute__((aligned(64)));

  for (int i = 0; i < SA_ROWS; i++) {
    for (int k = 0; k < K; k++) {
      A[i][k] = (uint16_t)(((i + 1) * 3 + k) & 0xF);
    }
  }
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < SA_COLS; j++) {
      B[k][j] = (uint16_t)(((j + 1) * 5 + 2 * k) & 0xF);
    }
  }

  for (int k = 0; k < K; k++) {
    a_stream[k] = pack2_u16(A[0][k], A[1][k]);
    b_stream[k] = pack2_u16(B[k][0], B[k][1]);
  }

  uint64_t sw_start = read_cycles();
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      uint32_t acc = 0;
      for (int k = 0; k < K; k++) {
        acc += (uint32_t)A[i][k] * (uint32_t)B[k][j];
      }
      sw[i][j] = acc;
    }
  }
  uint64_t sw_cycles = read_cycles() - sw_start;

  uint64_t hw_start = read_cycles();
  uint64_t cfg_rc = sa_config(a_stream, b_stream);
  uint64_t run_rc = sa_run(c_words, K);
  uint64_t hw_cycles = read_cycles() - hw_start;

  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      hw[i][j] = (uint32_t)c_words[i * SA_COLS + j];
    }
  }

  printf("=== Systolic GEMM Simple Test ===\n");
  printf("Dims: (%d x %d) * (%d x %d) -> (%d x %d)\n", SA_ROWS, K, K, SA_COLS, SA_ROWS, SA_COLS);
  printf("Config rc: %lu, Run rc: %lu\n", (unsigned long)cfg_rc, (unsigned long)run_rc);
  printf("SW cycles: %lu, HW cycles: %lu\n", (unsigned long)sw_cycles, (unsigned long)hw_cycles);

  printf("SW C:\n");
  for (int i = 0; i < SA_ROWS; i++) {
    printf("[%u %u]\n", sw[i][0], sw[i][1]);
  }
  printf("HW C:\n");
  for (int i = 0; i < SA_ROWS; i++) {
    printf("[%u %u]\n", hw[i][0], hw[i][1]);
  }

  int mismatches = 0;
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      if (sw[i][j] != hw[i][j]) {
        mismatches++;
      }
    }
  }

  if (mismatches == 0 && cfg_rc == 0 && run_rc == 0) {
    printf("PASS: hardware matches software.\n");
  } else {
    printf("FAIL: mismatches=%d\n", mismatches);
  }
  return 0;
}
