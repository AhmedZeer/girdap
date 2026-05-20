// SPDX-License-Identifier: Apache-2.0
//
// Standalone Gemmini BF16 GEMM sanity test.

#include "bfloat16_utils.h"
#include "include/gemmini_testutils.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_M 128
#define MAX_N 128
#define MAX_K 128
#define SAMPLE_COUNT 8

static elem_t mat_a[MAX_M][MAX_K] row_align(1);
static elem_t mat_b[MAX_K][MAX_N] row_align(1);
static elem_t mat_c[MAX_M][MAX_N] row_align(1);
static elem_t mat_ref[MAX_M][MAX_N] row_align(1);

typedef enum {
  DIST_ONES = 0,
  DIST_SMALL_SIGNED = 1,
  DIST_INT_0_2 = 2,
  DIST_SIGNED_INT = 3,
} dist_t;

typedef struct {
  const char *name;
  int m;
  int n;
  int k;
  dist_t dist;
  uint32_t tolerance_x100000;
} gemm_case_t;

static uint32_t rng_state = 1u;

static uint32_t next_rand_u32(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
}

static float rand_unit(void) {
  return (float)(next_rand_u32() & 0xFFFFu) / 65535.0f;
}

static float make_value(dist_t dist) {
  switch (dist) {
    case DIST_ONES:
      return 1.0f;
    case DIST_SMALL_SIGNED:
      return rand_unit() - 0.5f;
    case DIST_INT_0_2:
      return (float)(next_rand_u32() % 3u);
    case DIST_SIGNED_INT:
    default:
      return (float)((int)(next_rand_u32() % 5u) - 2);
  }
}

static void fill_inputs(const gemm_case_t *tc, int case_idx) {
  rng_state = 0xC001D00Du ^ (uint32_t)(case_idx * 0x9E3779B9u);

  for (int r = 0; r < tc->m; r++) {
    for (int c = 0; c < tc->k; c++) {
      mat_a[r][c] = float_to_bf16(make_value(tc->dist));
    }
  }

  for (int r = 0; r < tc->k; r++) {
    for (int c = 0; c < tc->n; c++) {
      mat_b[r][c] = float_to_bf16(make_value(tc->dist));
    }
  }

  memset(mat_c, 0, sizeof(mat_c));
  memset(mat_ref, 0, sizeof(mat_ref));
}

static uint64_t cpu_gemm_ref(const gemm_case_t *tc) {
  const uint64_t start = read_cycles();
  for (int r = 0; r < tc->m; r++) {
    for (int c = 0; c < tc->n; c++) {
      float acc = 0.0f;
      for (int k = 0; k < tc->k; k++) {
        acc += bf16_to_float(mat_a[r][k]) * bf16_to_float(mat_b[k][c]);
      }
      mat_ref[r][c] = float_to_bf16(acc);
    }
  }
  return read_cycles() - start;
}

static uint64_t gemmini_gemm(const gemm_case_t *tc) {
  gemmini_fence();
  const uint64_t start = read_cycles();
  tiled_matmul_auto(
      tc->m,
      tc->n,
      tc->k,
      (elem_t *)mat_a,
      (elem_t *)mat_b,
      NULL,
      (elem_t *)mat_c,
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
  return read_cycles() - start;
}

static void print_sample(
    const char *label,
    const elem_t data[MAX_M][MAX_N],
    int row,
    int start_col,
    int ncols) {
  printf("  %s row=%d cols=%d..%d :", label, row, start_col,
         start_col + ncols - 1);
  for (int i = 0; i < ncols; i++) {
    printf(" ");
    print_float_inline(bf16_to_float(data[row][start_col + i]));
  }
  printf("\n");
}

static int compare_outputs(
    const gemm_case_t *tc,
    int case_idx,
    uint32_t *max_abs_diff_x100000,
    int *bit_mismatches) {
  int mismatches = 0;
  *bit_mismatches = 0;
  *max_abs_diff_x100000 = 0;

  for (int r = 0; r < tc->m; r++) {
    for (int c = 0; c < tc->n; c++) {
      const float ref = bf16_to_float(mat_ref[r][c]);
      const float got = bf16_to_float(mat_c[r][c]);
      const float diff = fabsf(got - ref);
      const uint32_t diff_x100000 = (uint32_t)(diff * 100000.0f);

      if (mat_ref[r][c] != mat_c[r][c]) {
        (*bit_mismatches)++;
      }
      if (diff_x100000 > *max_abs_diff_x100000) {
        *max_abs_diff_x100000 = diff_x100000;
      }
      if (diff_x100000 > tc->tolerance_x100000) {
        if (mismatches < SAMPLE_COUNT) {
          printf("Mismatch case=%d name=%s row=%d col=%d ref_bf16=0x%04x hw_bf16=0x%04x ref=",
                 case_idx, tc->name, r, c, mat_ref[r][c], mat_c[r][c]);
          print_float_inline(ref);
          printf("hw=");
          print_float_inline(got);
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

int main(void) {
  static const gemm_case_t cases[] = {
      {"ones_8x8x16", 8, 8, 16, DIST_ONES, 0},
      {"small_signed_8x8x16", 8, 8, 16, DIST_SMALL_SIGNED, 2000},
      {"int_8x9x16", 8, 9, 16, DIST_INT_0_2, 2000},
      {"small_signed_8x64x64", 8, 64, 64, DIST_SMALL_SIGNED, 3000},
      {"int_16x16x16", 16, 16, 16, DIST_INT_0_2, 3000},
      {"signed_int_32x32x64", 32, 32, 64, DIST_SIGNED_INT, 5000},
      {"small_signed_64x64x64", 64, 64, 64, DIST_SMALL_SIGNED, 5000},
      {"small_signed_128x64x64", 128, 64, 64, DIST_SMALL_SIGNED, 5000},
      {"small_signed_64x128x64", 64, 128, 64, DIST_SMALL_SIGNED, 5000},
      {"small_signed_128x128x128", 128, 128, 128, DIST_SMALL_SIGNED, 7000},
  };

  gemmini_flush(0);

  printf("=== Gemmini BF16 GEMM Sanity Test ===\n");
  printf("gemmini: DIM=%d elem_bits=%d acc_bits=%d custom=%d\n",
         DIM, ELEM_T_EXP_BITS + ELEM_T_SIG_BITS,
         ACC_T_EXP_BITS + ACC_T_SIG_BITS, XCUSTOM_ACC);
  printf("CSV_HEADER,case,name,M,N,K,cpu_cycles,hw_cycles,speedup_x100,max_abs_diff_x100000,tolerance_x100000,bit_mismatches,mismatches\n");

  int total_mismatches = 0;
  for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
    const gemm_case_t *tc = &cases[i];
    fill_inputs(tc, i);

    const uint64_t hw_cycles = gemmini_gemm(tc);
    const uint64_t cpu_cycles = cpu_gemm_ref(tc);
    const uint64_t speedup_x100 =
        hw_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_cycles;

    uint32_t max_abs_diff_x100000 = 0;
    int bit_mismatches = 0;
    const int mismatches = compare_outputs(
        tc, i, &max_abs_diff_x100000, &bit_mismatches);
    total_mismatches += mismatches;

    printf("CSV_DATA,%d,%s,%d,%d,%d,%lu,%lu,%lu,%u,%u,%d,%d\n",
           i, tc->name, tc->m, tc->n, tc->k,
           (unsigned long)cpu_cycles,
           (unsigned long)hw_cycles,
           (unsigned long)speedup_x100,
           max_abs_diff_x100000,
           tc->tolerance_x100000,
           bit_mismatches,
           mismatches);

    const int sample_cols = tc->n < SAMPLE_COUNT ? tc->n : SAMPLE_COUNT;
    print_sample("ref", mat_ref, 0, 0, sample_cols);
    print_sample("hw ", mat_c, 0, 0, sample_cols);
  }

  if (total_mismatches == 0) {
    printf("PASS: Gemmini BF16 GEMM matched CPU tolerance.\n");
  } else {
    printf("FAIL: total Gemmini BF16 GEMM mismatches = %d\n", total_mismatches);
  }

  return total_mismatches == 0 ? 0 : 1;
}
