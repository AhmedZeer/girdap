#include "bfloat16_utils.h"
#include "ws_gemm.h"

#include <stdint.h>
#include <stdio.h>

#define SA_ROWS 4
#define SA_COLS 4
#define SIMPLE_K 140

static volatile float sw_float_checksum_sink;

static float pattern_a(int i, int k) {
  return ((float)((k % 11) - 5) * 0.125f) + ((float)i - 1.5f) * 0.375f;
}

static float pattern_b(int k, int j) {
  return ((float)(((2 * k + j) % 13) - 6) * 0.15625f) - ((float)j - 1.5f) * 0.28125f;
}

static void sw_gemm_bf16_float(
    const uint16_t A[SA_ROWS][SIMPLE_K],
    const uint16_t B[SIMPLE_K][SA_COLS],
    float C[SA_ROWS][SA_COLS]) {
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      float acc = 0.0f;
      for (int k = 0; k < SIMPLE_K; k++) {
        acc += bf16_to_float(A[i][k]) * bf16_to_float(B[k][j]);
      }
      C[i][j] = acc;
    }
  }
}

static void sw_gemm_bf16_fixed_ref(
    const uint16_t A[SA_ROWS][SIMPLE_K],
    const uint16_t B[SIMPLE_K][SA_COLS],
    int64_t C[SA_ROWS][SA_COLS]) {
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      int64_t acc = 0;
      for (int k = 0; k < SIMPLE_K; k++) {
        const int16_t a_fixed = bf16_to_fixed_s16_sat(A[i][k], WS_GEMM_BF16_FIXED_FRAC_BITS);
        const int16_t b_fixed = bf16_to_fixed_s16_sat(B[k][j], WS_GEMM_BF16_FIXED_FRAC_BITS);
        acc += (int64_t)a_fixed * (int64_t)b_fixed;
      }
      C[i][j] = acc;
    }
  }
}

static void quantize_fixed_ref_to_bf16(
    const int64_t C_fixed[SA_ROWS][SA_COLS],
    uint16_t C_bf16[SA_ROWS][SA_COLS],
    float C_float[SA_ROWS][SA_COLS]) {
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      C_bf16[i][j] = fixed_s64_to_bf16_sat(C_fixed[i][j], WS_GEMM_BF16_ACC_FRAC_BITS);
      C_float[i][j] = bf16_to_float(C_bf16[i][j]);
    }
  }
}

static void quantize_float_output_to_bf16(
    const float C_float[SA_ROWS][SA_COLS],
    uint16_t C_bf16[SA_ROWS][SA_COLS]) {
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      C_bf16[i][j] = float_to_bf16(C_float[i][j]);
    }
  }
}

static void print_float_matrix_4x4(const float C[SA_ROWS][SA_COLS]) {
  for (int i = 0; i < SA_ROWS; i++) {
    printf("[");
    for (int j = 0; j < SA_COLS; j++) {
      print_float_inline(C[i][j]);
    }
    printf("]\n");
  }
}

int main(void) {
  enum { K = SIMPLE_K };

  static uint16_t A[SA_ROWS][K] __attribute__((aligned(64)));
  static uint16_t B[K][SA_COLS] __attribute__((aligned(64)));
  static float sw_float[SA_ROWS][SA_COLS];
  static uint16_t sw_perf_bf16[SA_ROWS][SA_COLS];
  static float sw_ref_float[SA_ROWS][SA_COLS];
  static int64_t sw_ref_fixed[SA_ROWS][SA_COLS];
  static uint16_t sw_ref_bf16[SA_ROWS][SA_COLS];
  static uint16_t hw_bf16[SA_ROWS][SA_COLS];
  static float hw_float[SA_ROWS][SA_COLS];
  static uint64_t a_tiles[WS_GEMM_A_TILE_WORDS(SA_ROWS, K)] __attribute__((aligned(64)));
  static uint64_t b_tiles[WS_GEMM_B_TILE_WORDS(SA_COLS, K)] __attribute__((aligned(64)));
  static uint64_t c_words[WS_GEMM_C_TILE_WORDS] __attribute__((aligned(64)));
  ws_gemm_stats_t stats;

  const ws_gemm_workspace_t workspace =
      ws_gemm_make_workspace(a_tiles, b_tiles, c_words, SA_ROWS, SA_COLS, K);

  for (int i = 0; i < SA_ROWS; i++) {
    for (int k = 0; k < K; k++) {
      A[i][k] = float_to_bf16(pattern_a(i, k));
    }
  }
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < SA_COLS; j++) {
      B[k][j] = float_to_bf16(pattern_b(k, j));
    }
  }

  uint64_t sw_start = ws_read_cycles();
  sw_gemm_bf16_float(A, B, sw_float);
  quantize_float_output_to_bf16(sw_float, sw_perf_bf16);
  uint64_t sw_cycles = ws_read_cycles() - sw_start;
  float sw_checksum = 0.0f;
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      sw_checksum += bf16_to_float(sw_perf_bf16[i][j]);
    }
  }
  sw_float_checksum_sink = sw_checksum;

  sw_gemm_bf16_fixed_ref(A, B, sw_ref_fixed);
  quantize_fixed_ref_to_bf16(sw_ref_fixed, sw_ref_bf16, sw_ref_float);

  const int gemm_rc = ws_gemm_bf16(
      &A[0][0], K,
      &B[0][0], SA_COLS,
      &hw_bf16[0][0], SA_COLS,
      SA_ROWS, SA_COLS, K,
      &workspace,
      &stats);

  int mismatches = 0;
  for (int i = 0; i < SA_ROWS; i++) {
    for (int j = 0; j < SA_COLS; j++) {
      hw_float[i][j] = bf16_to_float(hw_bf16[i][j]);
      if (sw_ref_bf16[i][j] != hw_bf16[i][j]) {
        mismatches++;
      }
    }
  }

  printf("=== Systolic GEMM Simple Test (BF16 Weight-Stationary) ===\n");
  printf("Dims: (%d x %d) * (%d x %d) -> (%d x %d)\n", SA_ROWS, K, K, SA_COLS, SA_ROWS, SA_COLS);
  printf("Fixed-point mirror: input Q8.%d, accum Q*.%d\n",
         WS_GEMM_BF16_FIXED_FRAC_BITS,
         WS_GEMM_BF16_ACC_FRAC_BITS);
  printf("GEMM rc: %d\n", gemm_rc);
  printf("SW float cycles: %lu, HW end-to-end cycles: %lu\n",
         (unsigned long)sw_cycles,
         (unsigned long)stats.hw_e2e_cycles);

  printf("SW C (exact fixed mirror, quantized to BF16):\n");
  print_float_matrix_4x4(sw_ref_float);
  printf("HW C:\n");
  print_float_matrix_4x4(hw_float);
  printf("HW PERF:\n");
  ws_print_perf_counters(stats.perf);

  if (mismatches == 0 && gemm_rc == WS_GEMM_OK) {
    printf("PASS: hardware matches software exact BF16 reference.\n");
  } else {
    printf("FAIL: mismatches=%d rc=%d\n", mismatches, gemm_rc);
  }
  return 0;
}
