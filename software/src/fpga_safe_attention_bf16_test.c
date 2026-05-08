#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_Q_ROWS 256
#define MAX_KV_ROWS 256
#define MAX_D_K 256
#define MAX_VALUE_COLS 128
#define ATTENTION_TOLERANCE 0.25f
#ifndef FPGA_SAFE_ATTN_USE_HW_PACKER
#define FPGA_SAFE_ATTN_USE_HW_PACKER 0
#endif

typedef struct {
  int q_rows;
  int kv_rows;
  int d_k;
  int value_cols;
} attention_case_t;

static uint16_t q_matrix[MAX_Q_ROWS][MAX_D_K] __attribute__((aligned(64)));
static uint16_t k_matrix[MAX_KV_ROWS][MAX_D_K] __attribute__((aligned(64)));
static uint16_t k_t_matrix[MAX_D_K][MAX_KV_ROWS] __attribute__((aligned(64)));
static uint16_t v_matrix[MAX_KV_ROWS][MAX_VALUE_COLS] __attribute__((aligned(64)));
static uint16_t out_matrix[MAX_Q_ROWS][MAX_VALUE_COLS] __attribute__((aligned(64)));
static float sw_scores[MAX_Q_ROWS][MAX_KV_ROWS];
static float sw_probs[MAX_Q_ROWS][MAX_KV_ROWS];
static float sw_output[MAX_Q_ROWS][MAX_VALUE_COLS];

static uint64_t q_tiles[FPGA_SAFE_ATTN_Q_TILE_WORDS(MAX_Q_ROWS, MAX_D_K)] __attribute__((aligned(64)));
static uint64_t k_tiles[FPGA_SAFE_ATTN_K_TILE_WORDS(MAX_KV_ROWS, MAX_D_K)] __attribute__((aligned(64)));
static uint64_t v_tiles[FPGA_SAFE_ATTN_V_TILE_WORDS(MAX_VALUE_COLS, MAX_KV_ROWS)] __attribute__((aligned(64)));
static uint64_t out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

static float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static float rand_float_0_to_2(void) {
  return ((float)rand() / (float)RAND_MAX) * 2.0f;
}

static void transpose_k(int kv_rows, int d_k) {
  for (int r = 0; r < kv_rows; r++) {
    for (int c = 0; c < d_k; c++) {
      k_t_matrix[c][r] = k_matrix[r][c];
    }
  }
}

static void sw_attention_reference(
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols,
    float scale) {
  for (int qr = 0; qr < q_rows; qr++) {
    for (int kv = 0; kv < kv_rows; kv++) {
      double dot = 0.0;
      for (int d = 0; d < d_k; d++) {
        dot += (double)bf16_to_float(q_matrix[qr][d]) *
               (double)bf16_to_float(k_matrix[kv][d]);
      }
      sw_scores[qr][kv] = (float)(dot * (double)scale);
    }
  }

  for (int qr = 0; qr < q_rows; qr++) {
    float max_score = -INFINITY;
    for (int kv = 0; kv < kv_rows; kv++) {
      if (sw_scores[qr][kv] > max_score) {
        max_score = sw_scores[qr][kv];
      }
    }
    double denom = 0.0;
    for (int kv = 0; kv < kv_rows; kv++) {
      denom += exp((double)(sw_scores[qr][kv] - max_score));
    }
    for (int kv = 0; kv < kv_rows; kv++) {
      sw_probs[qr][kv] =
          (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
    }
  }

  for (int qr = 0; qr < q_rows; qr++) {
    for (int vc = 0; vc < value_cols; vc++) {
      double acc = 0.0;
      for (int kv = 0; kv < kv_rows; kv++) {
        acc += (double)sw_probs[qr][kv] * (double)bf16_to_float(v_matrix[kv][vc]);
      }
      sw_output[qr][vc] = (float)acc;
    }
  }
}

int main(void) {
  const attention_case_t tests[] = {
      {8, 8, 16, 8},
      {8, 16, 16, 8},
      {8, 64, 64, 8},
      {64, 64, 64, 64},
      {16, 64, 64, 16},
      {32, 128, 64, 32},
      {32, 256, 128, 64},
      {128, 256, 128, 64},
      {128, 256, 128, 128},
      {256, 256, 256, 64},
      {256, 256, 256, 128},
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));
  const fpga_safe_attention_workspace_t workspace =
      fpga_safe_attention_make_workspace(
          q_tiles,
          k_tiles,
          v_tiles,
          out_words,
          MAX_Q_ROWS,
          MAX_KV_ROWS,
          MAX_D_K,
          MAX_VALUE_COLS);

  srand(17);
  printf("=== FPGA-Safe 8x8 Online Attention BF16 %s Test ===\n",
         FPGA_SAFE_ATTN_USE_HW_PACKER ? "HW-Packer" : "SW-Packer");
  printf("CSV_HEADER,case,q_rows,kv_rows,d_k,value_cols,sw_cycles,hw_total_cycles,hw_accel_cycles,hw_score_cycles,hw_value_cycles,hw_rc,raw_hw_rc,speedup_x100,max_abs_diff_x100000,mismatches\n");

  int total_mismatches = 0;
  for (int t = 0; t < ntests; t++) {
    const attention_case_t tc = tests[t];
    const float scale = 1.0f / sqrtf((float)tc.d_k);
    const uint16_t scale_bf16 = float_to_bf16(scale);

    for (int r = 0; r < tc.q_rows; r++) {
      for (int c = 0; c < tc.d_k; c++) {
        q_matrix[r][c] = float_to_bf16(rand_float_0_to_2());
      }
    }
    for (int r = 0; r < tc.kv_rows; r++) {
      for (int c = 0; c < tc.d_k; c++) {
        k_matrix[r][c] = float_to_bf16(rand_float_0_to_2());
      }
      for (int c = 0; c < tc.value_cols; c++) {
        v_matrix[r][c] = float_to_bf16(rand_float_0_to_2());
      }
    }
    if (!FPGA_SAFE_ATTN_USE_HW_PACKER) {
      transpose_k(tc.kv_rows, tc.d_k);
    }

    for (int r = 0; r < tc.q_rows; r++) {
      for (int c = 0; c < tc.value_cols; c++) {
        out_matrix[r][c] = 0;
      }
    }

    const uint64_t sw_start = ws_read_cycles();
    sw_attention_reference(tc.q_rows, tc.kv_rows, tc.d_k, tc.value_cols, scale);
    const uint64_t sw_cycles = ws_read_cycles() - sw_start;

    fpga_safe_attention_stats_t stats;
    const int hw_rc =
#if FPGA_SAFE_ATTN_USE_HW_PACKER
        fpga_safe_attention_bf16_hwpack(
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
            scale_bf16,
            &out_matrix[0][0],
            MAX_VALUE_COLS,
            &workspace,
            &stats);
#else
        fpga_safe_attention_bf16(
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
        scale_bf16,
        &out_matrix[0][0],
        MAX_VALUE_COLS,
        &workspace,
        &stats);
#endif

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    if (hw_rc == FPGA_SAFE_ATTN_OK) {
      for (int r = 0; r < tc.q_rows; r++) {
        for (int c = 0; c < tc.value_cols; c++) {
          const float hw = bf16_to_float(out_matrix[r][c]);
          const float diff = fabsf(hw - sw_output[r][c]);
          if (diff > max_abs_diff) {
            max_abs_diff = diff;
          }
          if (diff > ATTENTION_TOLERANCE) {
            if (mismatches < 8) {
              printf("Mismatch case=%d row=%d col=%d sw=", t, r, c);
              print_float_inline(sw_output[r][c]);
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
    } else {
      mismatches = 1;
    }

    total_mismatches += mismatches;
    const uint64_t hw_total_cycles =
        stats.q_pack_cycles +
        stats.k_pack_cycles +
        stats.v_pack_cycles +
        stats.hw_e2e_cycles +
        stats.copy_out_cycles;
    const uint64_t speedup_x100 =
        hw_total_cycles == 0 ? 0 : (sw_cycles * 100u) / hw_total_cycles;
    printf("CSV_DATA,%d,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%d,%lu,%lu,%lu,%d\n",
           t,
           tc.q_rows,
           tc.kv_rows,
           tc.d_k,
           tc.value_cols,
           (unsigned long)sw_cycles,
           (unsigned long)hw_total_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           (unsigned long)stats.score_cycles,
           (unsigned long)stats.value_cycles,
           hw_rc,
           (unsigned long)stats.raw_hw_rc,
           (unsigned long)speedup_x100,
           (unsigned long)(max_abs_diff * 100000.0f),
           mismatches);
  }

  if (total_mismatches == 0) {
    printf("PASS: FPGA-safe online attention matched software tolerance.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }
  return total_mismatches == 0 ? 0 : 1;
}
