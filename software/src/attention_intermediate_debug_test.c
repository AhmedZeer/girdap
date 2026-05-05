#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "generated/attention_intermediate_cases.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint16_t out_matrix[ATTN_INTERMEDIATE_MAX_Q_ROWS][ATTN_INTERMEDIATE_MAX_VALUE_COLS]
    __attribute__((aligned(64)));
static int64_t hw_score_fixed[ATTN_INTERMEDIATE_MAX_Q_ROWS][ATTN_INTERMEDIATE_MAX_KV_ROWS]
    __attribute__((aligned(64)));
static uint64_t hw_prob_fixed[ATTN_INTERMEDIATE_MAX_Q_ROWS][ATTN_INTERMEDIATE_MAX_KV_ROWS]
    __attribute__((aligned(64)));

static uint64_t q_tiles[FPGA_SAFE_ATTN_Q_TILE_WORDS(ATTN_INTERMEDIATE_MAX_Q_ROWS, ATTN_INTERMEDIATE_MAX_D_K)]
    __attribute__((aligned(64)));
static uint64_t k_tiles[FPGA_SAFE_ATTN_K_TILE_WORDS(ATTN_INTERMEDIATE_MAX_KV_ROWS, ATTN_INTERMEDIATE_MAX_D_K)]
    __attribute__((aligned(64)));
static uint64_t v_tiles[FPGA_SAFE_ATTN_V_TILE_WORDS(ATTN_INTERMEDIATE_MAX_VALUE_COLS, ATTN_INTERMEDIATE_MAX_KV_ROWS)]
    __attribute__((aligned(64)));
static uint64_t out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

static double score_fixed_to_double(int64_t value) {
  return (double)value / (double)(1ULL << ATTN_INTERMEDIATE_SCORE_FRAC_BITS);
}

static double prob_fixed_to_double(uint64_t value) {
  return (double)value / (double)(1ULL << ATTN_INTERMEDIATE_PROB_FRAC_BITS);
}

static double abs_double(double value) {
  return value < 0.0 ? -value : value;
}

static void print_double_inline(double value) {
  if (value < 0.0) {
    putchar('-');
    value = -value;
  }
  const uint64_t whole = (uint64_t)value;
  const uint64_t frac = (uint64_t)((value - (double)whole) * 1000000.0);
  printf("%lu.%06lu", (unsigned long)whole, (unsigned long)frac);
}

static int compare_scores(
    const attention_intermediate_case_t *tc,
    uint64_t *max_raw_diff) {
  int mismatches = 0;
  *max_raw_diff = 0;
  for (int r = 0; r < tc->q_rows; r++) {
    for (int c = 0; c < tc->kv_rows; c++) {
      const int idx = r * tc->kv_rows + c;
      const int64_t ref = tc->score_fixed[idx];
      const int64_t hw = hw_score_fixed[r][c];
      const uint64_t diff = ref > hw ? (uint64_t)(ref - hw) : (uint64_t)(hw - ref);
      if (diff > *max_raw_diff) {
        *max_raw_diff = diff;
      }
      if (diff != 0) {
        if (mismatches < 8) {
          printf("ScoreMismatch case=%s row=%d col=%d ref_raw=%ld hw_raw=%ld ref=",
                 tc->name,
                 r,
                 c,
                 (long)ref,
                 (long)hw);
          print_double_inline(score_fixed_to_double(ref));
          printf(" hw=");
          print_double_inline(score_fixed_to_double(hw));
          printf("\n");
        }
        mismatches++;
      }
    }
  }
  return mismatches;
}

static int compare_probs(
    const attention_intermediate_case_t *tc,
    double *max_abs_diff) {
  int mismatches = 0;
  *max_abs_diff = 0.0;
  for (int r = 0; r < tc->q_rows; r++) {
    for (int c = 0; c < tc->kv_rows; c++) {
      const int idx = r * tc->kv_rows + c;
      const double ref = prob_fixed_to_double(tc->prob_fixed[idx]);
      const double hw = prob_fixed_to_double(hw_prob_fixed[r][c]);
      const double diff = abs_double(hw - ref);
      if (diff > *max_abs_diff) {
        *max_abs_diff = diff;
      }
      if ((uint32_t)(diff * 100000.0) > tc->prob_tolerance_x100000) {
        if (mismatches < 8) {
          printf("ProbMismatch case=%s row=%d col=%d ref=", tc->name, r, c);
          print_double_inline(ref);
          printf(" hw=");
          print_double_inline(hw);
          printf(" diff=");
          print_double_inline(diff);
          printf("\n");
        }
        mismatches++;
      }
    }
  }
  return mismatches;
}

static int compare_outputs(
    const attention_intermediate_case_t *tc,
    float *max_abs_diff) {
  int mismatches = 0;
  *max_abs_diff = 0.0f;
  for (int r = 0; r < tc->q_rows; r++) {
    for (int c = 0; c < tc->value_cols; c++) {
      const int idx = r * tc->value_cols + c;
      const float ref = bf16_to_float(tc->expected[idx]);
      const float hw = bf16_to_float(out_matrix[r][c]);
      const float diff = fabsf(hw - ref);
      if (diff > *max_abs_diff) {
        *max_abs_diff = diff;
      }
      if ((uint32_t)(diff * 100000.0f) > tc->output_tolerance_x100000) {
        if (mismatches < 8) {
          double replay = 0.0;
          for (int kv = 0; kv < tc->kv_rows; kv++) {
            replay += prob_fixed_to_double(hw_prob_fixed[r][kv]) *
                      (double)bf16_to_float(tc->v[kv * tc->value_cols + c]);
          }
          printf("OutputMismatch case=%s row=%d col=%d ref=", tc->name, r, c);
          print_float_inline(ref);
          printf("hw=");
          print_float_inline(hw);
          printf("replay_hw_prob=");
          print_float_inline((float)replay);
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

static void print_samples(const attention_intermediate_case_t *tc, int case_idx) {
  const int rows[] = {0, tc->q_rows / 2, tc->q_rows - 1};
  const int score_cols[] = {0, tc->kv_rows / 2, tc->kv_rows - 1};
  const int out_cols[] = {0, tc->value_cols / 2, tc->value_cols - 1};

  for (int i = 0; i < 3; i++) {
    const int r = rows[i];
    const int sc = score_cols[i];
    const int oc = out_cols[i];
    const int score_idx = r * tc->kv_rows + sc;
    const int out_idx = r * tc->value_cols + oc;
    printf("SAMPLE,%d,%s,row=%d,score_col=%d,out_col=%d,score_ref=",
           case_idx,
           tc->name,
           r,
           sc,
           oc);
    print_double_inline(score_fixed_to_double(tc->score_fixed[score_idx]));
    printf(",score_hw=");
    print_double_inline(score_fixed_to_double(hw_score_fixed[r][sc]));
    printf(",prob_ref=");
    print_double_inline(prob_fixed_to_double(tc->prob_fixed[score_idx]));
    printf(",prob_hw=");
    print_double_inline(prob_fixed_to_double(hw_prob_fixed[r][sc]));
    printf(",out_ref=");
    print_float_inline(bf16_to_float(tc->expected[out_idx]));
    printf(",out_hw=");
    print_float_inline(bf16_to_float(out_matrix[r][oc]));
    printf("\n");
  }
}

int main(void) {
  const fpga_safe_attention_workspace_t workspace =
      fpga_safe_attention_make_workspace(
          q_tiles,
          k_tiles,
          v_tiles,
          out_words,
          ATTN_INTERMEDIATE_MAX_Q_ROWS,
          ATTN_INTERMEDIATE_MAX_KV_ROWS,
          ATTN_INTERMEDIATE_MAX_D_K,
          ATTN_INTERMEDIATE_MAX_VALUE_COLS);

  printf("=== FPGA-Safe Attention Intermediate Debug Test ===\n");
  printf("CSV_HEADER,case,name,q_rows,kv_rows,d_k,value_cols,hw_rc,raw_hw_rc,hw_total_cycles,score_mismatches,prob_mismatches,output_mismatches,max_score_raw_diff,max_prob_abs_diff_x100000,max_output_abs_diff_x100000\n");
  printf("SAMPLE_HEADER,case,name,row,score_col,out_col,score_ref,score_hw,prob_ref,prob_hw,out_ref,out_hw\n");

  int total_mismatches = 0;
  for (int i = 0; i < attention_intermediate_case_count; i++) {
    const attention_intermediate_case_t *tc = &attention_intermediate_cases[i];
    memset(out_matrix, 0, sizeof(out_matrix));
    memset(hw_score_fixed, 0, sizeof(hw_score_fixed));
    memset(hw_prob_fixed, 0, sizeof(hw_prob_fixed));

    fpga_safe_attention_stats_t stats;
    const int hw_rc = fpga_safe_attention_bf16_hwpack_debug_intermediates(
        tc->q,
        tc->d_k,
        tc->k,
        tc->d_k,
        tc->v,
        tc->value_cols,
        tc->q_rows,
        tc->kv_rows,
        tc->d_k,
        tc->value_cols,
        tc->scale_bf16,
        &out_matrix[0][0],
        ATTN_INTERMEDIATE_MAX_VALUE_COLS,
        &hw_score_fixed[0][0],
        &hw_prob_fixed[0][0],
        ATTN_INTERMEDIATE_MAX_KV_ROWS,
        &workspace,
        &stats);

    uint64_t max_score_raw_diff = 0;
    double max_prob_abs_diff = 0.0;
    float max_output_abs_diff = 0.0f;
    const int score_mismatches =
        hw_rc == FPGA_SAFE_ATTN_OK ? compare_scores(tc, &max_score_raw_diff) : 1;
    const int prob_mismatches =
        hw_rc == FPGA_SAFE_ATTN_OK ? compare_probs(tc, &max_prob_abs_diff) : 1;
    const int output_mismatches =
        hw_rc == FPGA_SAFE_ATTN_OK ? compare_outputs(tc, &max_output_abs_diff) : 1;
    total_mismatches += score_mismatches + prob_mismatches + output_mismatches;

    print_samples(tc, i);
    printf("CSV_DATA,%d,%s,%d,%d,%d,%d,%d,%lu,%lu,%d,%d,%d,%lu,%lu,%lu\n",
           i,
           tc->name,
           tc->q_rows,
           tc->kv_rows,
           tc->d_k,
           tc->value_cols,
           hw_rc,
           (unsigned long)stats.raw_hw_rc,
           (unsigned long)(stats.q_pack_cycles + stats.k_pack_cycles + stats.v_pack_cycles +
                           stats.hw_e2e_cycles + stats.copy_out_cycles),
           score_mismatches,
           prob_mismatches,
           output_mismatches,
           (unsigned long)max_score_raw_diff,
           (unsigned long)(max_prob_abs_diff * 100000.0),
           (unsigned long)(max_output_abs_diff * 100000.0f));
  }

  if (total_mismatches == 0) {
    printf("PASS: intermediate debug cases matched expected tolerances.\n");
  } else {
    printf("FAIL: total intermediate mismatches = %d\n", total_mismatches);
  }

  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/attention_intermediate_cases.c"
