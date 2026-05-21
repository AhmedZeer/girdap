// SPDX-License-Identifier: Apache-2.0
//
// Minimal BF16-native GemminiDefault8 test.
//
// This avoids upstream int8-oriented model assumptions. It checks:
// 1. static BF16 mvin/mvout
// 2. runtime-filled BF16 mvin/mvout
// 3. tiny BF16 GEMM through Gemmini's tiled_matmul_auto path

#include "bfloat16_utils.h"
#include "include/gemmini_testutils.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_M 16
#define MAX_N 16
#define MAX_K 16

#if DIM != 8
#error "gemmini_bf16_minimal expects GemminiBF16Default8 software params with DIM=8"
#endif

#ifndef ELEM_T_IS_FLOAT
#error "gemmini_bf16_minimal expects BF16/float Gemmini software params"
#endif

static elem_t static_in[DIM][DIM] row_align(1) = {
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
    {0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x3f00},
};

static elem_t runtime_in[DIM][DIM] row_align(1);
static elem_t mvout_buf[DIM][DIM] row_align(1);

static elem_t mat_a[MAX_M][MAX_K] row_align(1);
static elem_t mat_b[MAX_K][MAX_N] row_align(1);
static elem_t mat_hw[MAX_M][MAX_N] row_align(1);
static elem_t mat_ref[MAX_M][MAX_N] row_align(1);

static void print_bf16_row(const char *label, const elem_t *row, int n) {
  printf("%s", label);
  for (int i = 0; i < n; i++) {
    printf(" 0x%04x/", row[i]);
    print_float_inline(bf16_to_float(row[i]));
  }
  printf("\n");
}

static int compare_matrix(
    const char *stage,
    const elem_t *ref,
    const elem_t *got,
    int rows,
    int cols,
    int stride,
    uint32_t tolerance_x100000,
    uint32_t *max_abs_diff_x100000) {
  int mismatches = 0;
  *max_abs_diff_x100000 = 0;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const elem_t ref_bits = ref[r * stride + c];
      const elem_t got_bits = got[r * stride + c];
      const float ref_f = bf16_to_float(ref_bits);
      const float got_f = bf16_to_float(got_bits);
      const float diff = fabsf(ref_f - got_f);
      const uint32_t diff_x100000 = (uint32_t)(diff * 100000.0f);

      if (diff_x100000 > *max_abs_diff_x100000) {
        *max_abs_diff_x100000 = diff_x100000;
      }

      if (diff_x100000 > tolerance_x100000) {
        if (mismatches < 8) {
          printf("%s_MISMATCH,row=%d,col=%d,ref_bf16=0x%04x,got_bf16=0x%04x,ref=",
                 stage, r, c, ref_bits, got_bits);
          print_float_inline(ref_f);
          printf("got=");
          print_float_inline(got_f);
          printf("diff=");
          print_float_inline(diff);
          printf("\n");
        }
        mismatches++;
      }
    }
  }

  return mismatches;
}

static int run_mvin_mvout(const char *name, elem_t in[DIM][DIM]) {
  memset(mvout_buf, 0, sizeof(mvout_buf));

  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  const uint64_t start = read_cycles();
  gemmini_mvin(in, 0);
  gemmini_mvout(mvout_buf, 0);
  gemmini_fence();
  const uint64_t cycles = read_cycles() - start;

  uint32_t max_abs_diff_x100000 = 0;
  const int mismatches = compare_matrix(
      name, &in[0][0], &mvout_buf[0][0], DIM, DIM, DIM, 0,
      &max_abs_diff_x100000);

  printf("MVIN_DATA,%s,%lu,%u,%d\n", name, cycles, max_abs_diff_x100000,
         mismatches);
  print_bf16_row("  ref_row0:", in[0], DIM);
  print_bf16_row("  got_row0:", mvout_buf[0], DIM);
  return mismatches;
}

typedef enum {
  GEMM_ONES = 0,
  GEMM_MIXED = 1,
} gemm_pattern_t;

static void fill_gemm(int m, int n, int k, gemm_pattern_t pattern) {
  memset(mat_a, 0, sizeof(mat_a));
  memset(mat_b, 0, sizeof(mat_b));
  memset(mat_hw, 0, sizeof(mat_hw));
  memset(mat_ref, 0, sizeof(mat_ref));

  for (int r = 0; r < m; r++) {
    for (int c = 0; c < k; c++) {
      float value = 1.0f;
      if (pattern == GEMM_MIXED) {
        value = (float)(((r + c) % 5) - 2) * 0.25f;
      }
      mat_a[r][c] = float_to_bf16(value);
    }
  }

  for (int r = 0; r < k; r++) {
    for (int c = 0; c < n; c++) {
      float value = 1.0f;
      if (pattern == GEMM_MIXED) {
        value = (float)(((r * 3 + c) % 7) - 3) * 0.125f;
      }
      mat_b[r][c] = float_to_bf16(value);
    }
  }

  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; kk++) {
        acc += bf16_to_float(mat_a[r][kk]) * bf16_to_float(mat_b[kk][c]);
      }
      mat_ref[r][c] = float_to_bf16(acc);
    }
  }
}

static int run_gemm(
    const char *name,
    int m,
    int n,
    int k,
    gemm_pattern_t pattern,
    uint32_t tolerance_x100000) {
  fill_gemm(m, n, k, pattern);

  gemmini_flush(0);
  const uint64_t start = read_cycles();
  tiled_matmul_auto(
      m,
      n,
      k,
      (elem_t *)mat_a,
      (elem_t *)mat_b,
      NULL,
      (elem_t *)mat_hw,
      MAX_K,
      MAX_N,
      0,
      MAX_N,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      NO_ACTIVATION,
      ACC_SCALE_IDENTITY,
      0,
      false,
      false,
      false,
      false,
      false,
      0,
      WS);
  gemmini_fence();
  const uint64_t cycles = read_cycles() - start;

  uint32_t max_abs_diff_x100000 = 0;
  const int mismatches = compare_matrix(
      name, &mat_ref[0][0], &mat_hw[0][0], m, n, MAX_N, tolerance_x100000,
      &max_abs_diff_x100000);

  printf("GEMM_DATA,%s,%d,%d,%d,%lu,%u,%u,%d\n", name, m, n, k, cycles,
         max_abs_diff_x100000, tolerance_x100000, mismatches);
  print_bf16_row("  ref_row0:", mat_ref[0], n < DIM ? n : DIM);
  print_bf16_row("  got_row0:", mat_hw[0], n < DIM ? n : DIM);
  return mismatches;
}

int main(void) {
  printf("=== Gemmini BF16 Minimal Viable Test ===\n");
  printf("gemmini: DIM=%d elem_bits=%d acc_bits=%d custom=%d\n",
         DIM, ELEM_T_EXP_BITS + ELEM_T_SIG_BITS,
         ACC_T_EXP_BITS + ACC_T_SIG_BITS, XCUSTOM_ACC);
  printf("MVIN_HEADER,name,cycles,max_abs_diff_x100000,mismatches\n");
  printf("GEMM_HEADER,name,M,N,K,cycles,max_abs_diff_x100000,tolerance_x100000,mismatches\n");

  for (int r = 0; r < DIM; r++) {
    for (int c = 0; c < DIM; c++) {
      const float value = (float)((r * DIM + c) % 9 - 4) * 0.25f;
      runtime_in[r][c] = float_to_bf16(value);
    }
  }

  int total_mismatches = 0;
  total_mismatches += run_mvin_mvout("MVIN_STATIC", static_in);
  total_mismatches += run_mvin_mvout("MVIN_RUNTIME", runtime_in);
  total_mismatches += run_gemm("GEMM_ONES_8x8x8_WS", 8, 8, 8, GEMM_ONES, 0);
  total_mismatches += run_gemm("GEMM_ONES_8x8x16_WS", 8, 8, 16, GEMM_ONES, 0);
  total_mismatches += run_gemm("GEMM_MIXED_8x8x16_WS", 8, 8, 16, GEMM_MIXED, 2000);

  if (total_mismatches != 0) {
    printf("FAIL: total_mismatches=%d\n", total_mismatches);
    return 1;
  }

  printf("PASS: Gemmini BF16 minimal viable path passed.\n");
  return 0;
}
