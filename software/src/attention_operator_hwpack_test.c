#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "generated/attention_operator_cases.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint16_t out_matrix[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_VALUE_COLS]
    __attribute__((aligned(64)));
static float sw_scores[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_KV_ROWS];
static float sw_probs[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_KV_ROWS];
static float sw_output[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_VALUE_COLS];

static uint64_t q_tiles[FPGA_SAFE_ATTN_Q_TILE_WORDS(ATTN_OPERATOR_MAX_Q_ROWS, ATTN_OPERATOR_MAX_D_K)]
    __attribute__((aligned(64)));
static uint64_t k_tiles[FPGA_SAFE_ATTN_K_TILE_WORDS(ATTN_OPERATOR_MAX_KV_ROWS, ATTN_OPERATOR_MAX_D_K)]
    __attribute__((aligned(64)));
static uint64_t v_tiles[FPGA_SAFE_ATTN_V_TILE_WORDS(ATTN_OPERATOR_MAX_VALUE_COLS, ATTN_OPERATOR_MAX_KV_ROWS)]
    __attribute__((aligned(64)));
static uint64_t out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

static uint16_t tensor_at(
    const uint16_t *tensor,
    int row,
    int col,
    int ld) {
  return tensor[row * ld + col];
}

static void sw_attention_reference(const attention_operator_case_t *tc) {
  const float scale = bf16_to_float(tc->scale_bf16);

  for (int qr = 0; qr < tc->q_rows; qr++) {
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      double dot = 0.0;
      for (int d = 0; d < tc->d_k; d++) {
        dot += (double)bf16_to_float(tensor_at(tc->q, qr, d, tc->d_k)) *
               (double)bf16_to_float(tensor_at(tc->k, kv, d, tc->d_k));
      }
      sw_scores[qr][kv] = (float)(dot * (double)scale);
    }
  }

  for (int qr = 0; qr < tc->q_rows; qr++) {
    float max_score = -INFINITY;
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      if (sw_scores[qr][kv] > max_score) {
        max_score = sw_scores[qr][kv];
      }
    }

    double denom = 0.0;
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      denom += exp((double)(sw_scores[qr][kv] - max_score));
    }
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      sw_probs[qr][kv] =
          (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
    }
  }

  for (int qr = 0; qr < tc->q_rows; qr++) {
    for (int vc = 0; vc < tc->value_cols; vc++) {
      double acc = 0.0;
      for (int kv = 0; kv < tc->kv_rows; kv++) {
        acc += (double)sw_probs[qr][kv] *
               (double)bf16_to_float(tensor_at(tc->v, kv, vc, tc->value_cols));
      }
      sw_output[qr][vc] = (float)acc;
    }
  }
}

static int compare_bf16_output(
    const attention_operator_case_t *tc,
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
      if ((uint32_t)(diff * 100000.0f) > tc->tolerance_x100000) {
        if (mismatches < 8) {
          printf("Mismatch case=%s row=%d col=%d ref=", tc->name, r, c);
          print_float_inline(ref);
          printf("hw=");
          print_float_inline(hw);
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

static float max_cpu_vs_exported_diff(const attention_operator_case_t *tc) {
  float max_abs_diff = 0.0f;
  for (int r = 0; r < tc->q_rows; r++) {
    for (int c = 0; c < tc->value_cols; c++) {
      const int idx = r * tc->value_cols + c;
      const float ref = bf16_to_float(tc->expected[idx]);
      const float diff = fabsf(sw_output[r][c] - ref);
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
      }
    }
  }
  return max_abs_diff;
}

int main(void) {
  const fpga_safe_attention_workspace_t workspace =
      fpga_safe_attention_make_workspace(
          q_tiles,
          k_tiles,
          v_tiles,
          out_words,
          ATTN_OPERATOR_MAX_Q_ROWS,
          ATTN_OPERATOR_MAX_KV_ROWS,
          ATTN_OPERATOR_MAX_D_K,
          ATTN_OPERATOR_MAX_VALUE_COLS);

  printf("=== Generic BF16 Attention Operator HW-Packer Test ===\n");
  printf("CSV_HEADER,case,name,q_rows,kv_rows,d_k,value_cols,cpu_cycles,hw_total_cycles,hw_accel_cycles,q_pack_cycles,k_pack_cycles,v_pack_cycles,hw_score_cycles,hw_value_cycles,hw_rc,raw_hw_rc,speedup_x100,hw_max_abs_diff_x100000,cpu_ref_max_abs_diff_x100000,mismatches\n");

  int total_mismatches = 0;
  for (int i = 0; i < attention_operator_case_count; i++) {
    const attention_operator_case_t *tc = &attention_operator_cases[i];
    memset(out_matrix, 0, sizeof(out_matrix));

    const uint64_t cpu_start = ws_read_cycles();
    sw_attention_reference(tc);
    const uint64_t cpu_cycles = ws_read_cycles() - cpu_start;
    const float cpu_ref_max_abs_diff = max_cpu_vs_exported_diff(tc);

    fpga_safe_attention_stats_t stats;
    const int hw_rc = fpga_safe_attention_bf16_hwpack(
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
        ATTN_OPERATOR_MAX_VALUE_COLS,
        &workspace,
        &stats);

    float hw_max_abs_diff = 0.0f;
    const int mismatches =
        (hw_rc == FPGA_SAFE_ATTN_OK) ? compare_bf16_output(tc, &hw_max_abs_diff) : 1;
    total_mismatches += mismatches;

    const uint64_t hw_total_cycles =
        stats.q_pack_cycles +
        stats.k_pack_cycles +
        stats.v_pack_cycles +
        stats.hw_e2e_cycles +
        stats.copy_out_cycles;
    const uint64_t speedup_x100 =
        hw_total_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_total_cycles;

    printf("CSV_DATA,%d,%s,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%d,%lu,%lu,%lu,%lu,%d\n",
           i,
           tc->name,
           tc->q_rows,
           tc->kv_rows,
           tc->d_k,
           tc->value_cols,
           (unsigned long)cpu_cycles,
           (unsigned long)hw_total_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           (unsigned long)stats.q_pack_cycles,
           (unsigned long)stats.k_pack_cycles,
           (unsigned long)stats.v_pack_cycles,
           (unsigned long)stats.score_cycles,
           (unsigned long)stats.value_cycles,
           hw_rc,
           (unsigned long)stats.raw_hw_rc,
           (unsigned long)speedup_x100,
           (unsigned long)(hw_max_abs_diff * 100000.0f),
           (unsigned long)(cpu_ref_max_abs_diff * 100000.0f),
           mismatches);
  }

  if (total_mismatches == 0) {
    printf("PASS: generic attention operator cases matched exported PyTorch tolerance.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }

  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/attention_operator_cases.c"
