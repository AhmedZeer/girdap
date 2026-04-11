#include "attention.h"
#include "bfloat16_utils.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_Q_ROWS 8
#define MAX_KV_ROWS 128
#define MAX_D_K 64
#define MAX_VALUE_COLS 64
#define SAMPLE_WIDTH 10
#define ATTENTION_TOLERANCE 0.20f

typedef struct {
  int q_rows;
  int kv_rows;
  int d_k;
  int value_cols;
  int causal;
} attention_case_t;

static uint16_t q_matrix[MAX_Q_ROWS][MAX_D_K] __attribute__((aligned(64)));
static uint16_t k_matrix[MAX_KV_ROWS][MAX_D_K] __attribute__((aligned(64)));
static uint16_t k_t_matrix[MAX_D_K][MAX_KV_ROWS] __attribute__((aligned(64)));
static uint16_t v_matrix[MAX_KV_ROWS][MAX_VALUE_COLS] __attribute__((aligned(64)));
static uint16_t out_matrix[MAX_Q_ROWS][MAX_VALUE_COLS] __attribute__((aligned(64)));

static float sw_output[MAX_Q_ROWS][MAX_VALUE_COLS];

static uint16_t scores_bf16[MAX_Q_ROWS][MAX_KV_ROWS] __attribute__((aligned(64)));
static uint16_t probs_bf16[MAX_Q_ROWS][MAX_KV_ROWS] __attribute__((aligned(64)));

static uint64_t qk_a_tiles[WS_GEMM_A_TILE_WORDS(MAX_Q_ROWS, MAX_D_K)] __attribute__((aligned(64)));
static uint64_t qk_b_tiles[WS_GEMM_B_TILE_WORDS(MAX_KV_ROWS, MAX_D_K)] __attribute__((aligned(64)));
static uint64_t qk_c_words[WS_GEMM_C_TILE_WORDS] __attribute__((aligned(64)));

static uint64_t pv_a_tiles[WS_GEMM_A_TILE_WORDS(MAX_Q_ROWS, MAX_KV_ROWS)] __attribute__((aligned(64)));
static uint64_t pv_b_tiles[WS_GEMM_B_TILE_WORDS(MAX_VALUE_COLS, MAX_KV_ROWS)] __attribute__((aligned(64)));
static uint64_t pv_c_words[WS_GEMM_C_TILE_WORDS] __attribute__((aligned(64)));

static float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void transpose_k_to_k_t(int kv_rows, int d_k) {
  for (int r = 0; r < kv_rows; r++) {
    for (int c = 0; c < d_k; c++) {
      k_t_matrix[c][r] = k_matrix[r][c];
    }
  }
}

static void sw_attention_reference(
    const uint16_t *Q,
    int ldq,
    const uint16_t *K,
    int ldk,
    const uint16_t *V,
    int ldv,
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols,
    const attention_params_t *params,
    float *output,
    int ldout) {
  float score_row[MAX_KV_ROWS];
  float prob_row[MAX_KV_ROWS];

  for (int qr = 0; qr < q_rows; qr++) {
    float max_score = -INFINITY;
    for (int kv = 0; kv < kv_rows; kv++) {
      double dot = 0.0;
      for (int d = 0; d < d_k; d++) {
        const float q_val = bf16_to_float(Q[qr * ldq + d]);
        const float k_val = bf16_to_float(K[kv * ldk + d]);
        dot += (double)q_val * (double)k_val;
      }

      float score = (float)dot;
      if (params != 0) {
        score *= params->scale;
        if (params->mask_bf16 != 0) {
          score += bf16_to_float(params->mask_bf16[qr * params->mask_stride + kv]);
        }
        if (params->causal && kv > params->query_position_base + qr) {
          score = -1.0e9f;
        }
      }

      score_row[kv] = score;
      if (score > max_score) {
        max_score = score;
      }
    }

    double denom = 0.0;
    for (int kv = 0; kv < kv_rows; kv++) {
      denom += exp((double)(score_row[kv] - max_score));
    }

    for (int kv = 0; kv < kv_rows; kv++) {
      prob_row[kv] = (float)(exp((double)(score_row[kv] - max_score)) / denom);
    }

    for (int vc = 0; vc < value_cols; vc++) {
      double acc = 0.0;
      for (int kv = 0; kv < kv_rows; kv++) {
        const float v_val = bf16_to_float(V[kv * ldv + vc]);
        acc += (double)prob_row[kv] * (double)v_val;
      }
      output[qr * ldout + vc] = (float)acc;
    }
  }
}

static void print_attention_output_samples(
    int case_id,
    int q_rows,
    int value_cols) {
  printf("\n=== Attention Case %d Output Samples ===\n", case_id);
  print_float_array_samples(
      "Software Output Row 0",
      &sw_output[0][0],
      (size_t)value_cols,
      SAMPLE_WIDTH);
  print_bf16_array_samples(
      "Hardware Output Row 0",
      &out_matrix[0][0],
      (size_t)value_cols,
      SAMPLE_WIDTH);

  if (q_rows > 1) {
    print_float_array_samples(
        "Software Output Last Row",
        &sw_output[q_rows - 1][0],
        (size_t)value_cols,
        SAMPLE_WIDTH);
    print_bf16_array_samples(
        "Hardware Output Last Row",
        &out_matrix[q_rows - 1][0],
        (size_t)value_cols,
        SAMPLE_WIDTH);
  }
}

int main(void) {
  const attention_case_t tests[] = {
      {1, 16, 32, 32, 0},
      {4, 16, 32, 32, 0},
      {4, 32, 32, 32, 1},
      {6, 64, 64, 48, 1},
      {8, 96, 64, 64, 0},
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));

  const attention_workspace_t workspace = attention_make_workspace(
      qk_a_tiles,
      qk_b_tiles,
      qk_c_words,
      pv_a_tiles,
      pv_b_tiles,
      pv_c_words,
      &scores_bf16[0][0],
      &probs_bf16[0][0],
      MAX_Q_ROWS,
      MAX_KV_ROWS,
      MAX_D_K,
      MAX_VALUE_COLS);

  srand(11);
  printf("=== BF16 Attention Test ===\n");
  printf("CSV_HEADER,case,q_rows,kv_rows,d_k,value_cols,causal,sw_cycles,hw_total_cycles,hw_accel_cycles,qk_pack_cycles,qk_hw_e2e_cycles,qk_pack_a_cycles,qk_preload_cycles,qk_run_cycles,qk_copy_out_cycles,score_post_cycles,softmax_hw_e2e_cycles,softmax_pass1_cycles,softmax_pass2_cycles,pv_pack_cycles,pv_hw_e2e_cycles,pv_pack_a_cycles,pv_preload_cycles,pv_run_cycles,pv_copy_out_cycles,mismatches\n");

  int total_mismatches = 0;

  for (int t = 0; t < ntests; t++) {
    const attention_case_t tc = tests[t];
    const int query_position_base = tc.causal ? (tc.kv_rows - tc.q_rows) : 0;
    attention_params_t params = attention_default_params();
    params.scale = 1.0f / sqrtf((float)tc.d_k);
    params.causal = tc.causal;
    params.query_position_base = query_position_base;

    for (int r = 0; r < tc.q_rows; r++) {
      for (int c = 0; c < tc.d_k; c++) {
        q_matrix[r][c] = float_to_bf16(rand_float());
      }
    }
    for (int r = 0; r < tc.kv_rows; r++) {
      for (int c = 0; c < tc.d_k; c++) {
        k_matrix[r][c] = float_to_bf16(rand_float());
      }
      for (int c = 0; c < tc.value_cols; c++) {
        v_matrix[r][c] = float_to_bf16(rand_float());
      }
    }
    transpose_k_to_k_t(tc.kv_rows, tc.d_k);

    for (int r = 0; r < tc.q_rows; r++) {
      for (int c = 0; c < tc.value_cols; c++) {
        out_matrix[r][c] = 0;
      }
    }

    const uint64_t sw_start = attention_read_cycles();
    sw_attention_reference(
        &q_matrix[0][0],
        MAX_D_K,
        &k_matrix[0][0],
        MAX_D_K,
        &v_matrix[0][0],
        MAX_VALUE_COLS,
        tc.q_rows,
        tc.kv_rows,
        tc.d_k,
        tc.value_cols,
        &params,
        &sw_output[0][0],
        MAX_VALUE_COLS);
    const uint64_t sw_cycles = attention_read_cycles() - sw_start;

    attention_stats_t stats;
    const int rc = attention_bf16(
        &q_matrix[0][0],
        MAX_D_K,
        &k_t_matrix[0][0],
        MAX_KV_ROWS,
        &v_matrix[0][0],
        MAX_VALUE_COLS,
        tc.q_rows,
        tc.kv_rows,
        tc.d_k,
        tc.value_cols,
        &out_matrix[0][0],
        MAX_VALUE_COLS,
        &workspace,
        &params,
        &stats);
    if (rc != ATTENTION_OK) {
      printf("FAIL: attention_bf16 rc=%d on case %d\n", rc, t);
      return 1;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    for (int r = 0; r < tc.q_rows; r++) {
      for (int c = 0; c < tc.value_cols; c++) {
        const float hw_value = bf16_to_float(out_matrix[r][c]);
        const float diff = fabsf(hw_value - sw_output[r][c]);
        if (diff > max_abs_diff) {
          max_abs_diff = diff;
        }
        if (diff > ATTENTION_TOLERANCE) {
          if (mismatches < 8) {
            printf("Mismatch case=%d row=%d col=%d sw=", t, r, c);
            print_float_inline(sw_output[r][c]);
            printf("hw=");
            print_float_inline(hw_value);
            printf("diff=");
            print_float_inline(diff);
            printf("\n");
          }
          mismatches++;
        }
      }
    }

    total_mismatches += mismatches;
    printf("CSV_DATA,%d,%d,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%d\n",
           t,
           tc.q_rows,
           tc.kv_rows,
           tc.d_k,
           tc.value_cols,
           tc.causal,
           (unsigned long)sw_cycles,
           (unsigned long)stats.total_cycles,
           (unsigned long)attention_accelerator_cycles(&stats),
           (unsigned long)stats.qk_pack_cycles,
           (unsigned long)stats.qk_matmul.hw_e2e_cycles,
           (unsigned long)stats.qk_matmul.stage.pack_a_cycles,
           (unsigned long)stats.qk_matmul.stage.preload_cycles,
           (unsigned long)stats.qk_matmul.stage.run_cycles,
           (unsigned long)stats.qk_matmul.stage.copy_out_cycles,
           (unsigned long)stats.score_post_cycles,
           (unsigned long)stats.softmax.hw_e2e_cycles,
           (unsigned long)stats.softmax.stage.pass1_cycles,
           (unsigned long)stats.softmax.stage.pass2_cycles,
           (unsigned long)stats.pv_pack_cycles,
           (unsigned long)stats.pv_matmul.hw_e2e_cycles,
           (unsigned long)stats.pv_matmul.stage.pack_a_cycles,
           (unsigned long)stats.pv_matmul.stage.preload_cycles,
           (unsigned long)stats.pv_matmul.stage.run_cycles,
           (unsigned long)stats.pv_matmul.stage.copy_out_cycles,
           mismatches);
    print_attention_output_samples(t, tc.q_rows, tc.value_cols);
    print_float("Max abs diff: ", max_abs_diff);

    if (mismatches != 0) {
      break;
    }
  }

  if (total_mismatches == 0) {
    printf("PASS: BF16 attention matched the software reference.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }

  return total_mismatches == 0 ? 0 : 1;
}
