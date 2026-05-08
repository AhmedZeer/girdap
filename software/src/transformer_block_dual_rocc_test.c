#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "generated/transformer_block_cases.h"
#include "ws_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint16_t q_proj[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL]
    __attribute__((aligned(64)));
static uint16_t k_proj[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL]
    __attribute__((aligned(64)));
static uint16_t v_proj[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL]
    __attribute__((aligned(64)));
static uint16_t context[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL]
    __attribute__((aligned(64)));
static uint16_t hw_output[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL]
    __attribute__((aligned(64)));

static float sw_q[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL];
static float sw_k[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL];
static float sw_v[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL];
static float sw_context[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL];
static float sw_output[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL];
static float sw_scores[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_SEQ_LEN];
static float sw_probs[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_SEQ_LEN];

static uint64_t gemm_a_tiles[
    WS_GEMM8_A_TILE_WORDS(TRANSFORMER_BLOCK_MAX_SEQ_LEN, TRANSFORMER_BLOCK_MAX_D_MODEL)]
    __attribute__((aligned(64)));
static uint64_t gemm_b_tiles[
    WS_GEMM8_B_TILE_WORDS(TRANSFORMER_BLOCK_MAX_D_MODEL, TRANSFORMER_BLOCK_MAX_D_MODEL)]
    __attribute__((aligned(64)));
static uint64_t gemm_c_words[WS_GEMM8_C_TILE_WORDS] __attribute__((aligned(64)));

static uint64_t q_tiles[
    FPGA_SAFE_ATTN_Q_TILE_WORDS(TRANSFORMER_BLOCK_MAX_SEQ_LEN, TRANSFORMER_BLOCK_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t k_tiles[
    FPGA_SAFE_ATTN_K_TILE_WORDS(TRANSFORMER_BLOCK_MAX_SEQ_LEN, TRANSFORMER_BLOCK_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t v_tiles[
    FPGA_SAFE_ATTN_V_TILE_WORDS(TRANSFORMER_BLOCK_MAX_HEAD_DIM, TRANSFORMER_BLOCK_MAX_SEQ_LEN)]
    __attribute__((aligned(64)));
static uint64_t attn_out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

static uint16_t tensor_at(
    const uint16_t *tensor,
    int row,
    int col,
    int ld) {
  return tensor[row * ld + col];
}

static void sw_projection(
    const uint16_t *x,
    const uint16_t *w,
    int seq_len,
    int d_model,
    float out[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL]) {
  for (int r = 0; r < seq_len; r++) {
    for (int c = 0; c < d_model; c++) {
      double acc = 0.0;
      for (int k = 0; k < d_model; k++) {
        acc += (double)bf16_to_float(tensor_at(x, r, k, d_model)) *
               (double)bf16_to_float(tensor_at(w, k, c, d_model));
      }
      out[r][c] = bf16_to_float(float_to_bf16((float)acc));
    }
  }
}

static void sw_output_projection(
    const uint16_t *w,
    int seq_len,
    int d_model) {
  for (int r = 0; r < seq_len; r++) {
    for (int c = 0; c < d_model; c++) {
      double acc = 0.0;
      for (int k = 0; k < d_model; k++) {
        acc += (double)sw_context[r][k] *
               (double)bf16_to_float(tensor_at(w, k, c, d_model));
      }
      sw_output[r][c] = bf16_to_float(float_to_bf16((float)acc));
    }
  }
}

static void sw_transformer_block_reference(const transformer_block_case_t *tc) {
  const int seq_len = tc->seq_len;
  const int d_model = tc->d_model;
  const int head_dim = tc->head_dim;
  const float scale = bf16_to_float(tc->scale_bf16);

  sw_projection(tc->x, tc->wq, seq_len, d_model, sw_q);
  sw_projection(tc->x, tc->wk, seq_len, d_model, sw_k);
  sw_projection(tc->x, tc->wv, seq_len, d_model, sw_v);

  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * head_dim;
    for (int qr = 0; qr < seq_len; qr++) {
      for (int kv = 0; kv < seq_len; kv++) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; d++) {
          dot += (double)sw_q[qr][base + d] * (double)sw_k[kv][base + d];
        }
        sw_scores[qr][kv] = (float)(dot * (double)scale);
      }
    }

    for (int qr = 0; qr < seq_len; qr++) {
      float max_score = -INFINITY;
      for (int kv = 0; kv < seq_len; kv++) {
        if (sw_scores[qr][kv] > max_score) {
          max_score = sw_scores[qr][kv];
        }
      }

      double denom = 0.0;
      for (int kv = 0; kv < seq_len; kv++) {
        denom += exp((double)(sw_scores[qr][kv] - max_score));
      }
      for (int kv = 0; kv < seq_len; kv++) {
        sw_probs[qr][kv] =
            (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
      }
    }

    for (int qr = 0; qr < seq_len; qr++) {
      for (int vc = 0; vc < head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv < seq_len; kv++) {
          acc += (double)sw_probs[qr][kv] * (double)sw_v[kv][base + vc];
        }
        sw_context[qr][base + vc] = bf16_to_float(float_to_bf16((float)acc));
      }
    }
  }

  sw_output_projection(tc->wo, seq_len, d_model);
}

static int run_projection(
    const transformer_block_case_t *tc,
    const uint16_t *weights,
    uint16_t out[TRANSFORMER_BLOCK_MAX_SEQ_LEN][TRANSFORMER_BLOCK_MAX_D_MODEL],
    const ws_gemm_workspace_t *workspace,
    ws_gemm_stats_t *stats) {
  return ws_gemm8_bf16(
      tc->x,
      tc->d_model,
      weights,
      tc->d_model,
      &out[0][0],
      TRANSFORMER_BLOCK_MAX_D_MODEL,
      tc->seq_len,
      tc->d_model,
      tc->d_model,
      workspace,
      stats);
}

static int run_output_projection(
    const transformer_block_case_t *tc,
    const ws_gemm_workspace_t *workspace,
    ws_gemm_stats_t *stats) {
  return ws_gemm8_bf16(
      &context[0][0],
      TRANSFORMER_BLOCK_MAX_D_MODEL,
      tc->wo,
      tc->d_model,
      &hw_output[0][0],
      TRANSFORMER_BLOCK_MAX_D_MODEL,
      tc->seq_len,
      tc->d_model,
      tc->d_model,
      workspace,
      stats);
}

static int run_attention_heads(
    const transformer_block_case_t *tc,
    const fpga_safe_attention_workspace_t *workspace,
    uint64_t *attn_e2e_cycles,
    uint64_t *attn_accel_cycles,
    uint64_t *attn_score_cycles,
    uint64_t *attn_value_cycles,
    uint64_t *raw_hw_rc) {
  *attn_e2e_cycles = 0;
  *attn_accel_cycles = 0;
  *attn_score_cycles = 0;
  *attn_value_cycles = 0;
  *raw_hw_rc = 0;

  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    fpga_safe_attention_stats_t stats;
    const int rc = fpga_safe_attention_bf16_hwpack(
        &q_proj[0][base],
        TRANSFORMER_BLOCK_MAX_D_MODEL,
        &k_proj[0][base],
        TRANSFORMER_BLOCK_MAX_D_MODEL,
        &v_proj[0][base],
        TRANSFORMER_BLOCK_MAX_D_MODEL,
        tc->seq_len,
        tc->seq_len,
        tc->head_dim,
        tc->head_dim,
        tc->scale_bf16,
        &context[0][base],
        TRANSFORMER_BLOCK_MAX_D_MODEL,
        workspace,
        &stats);

    *raw_hw_rc = stats.raw_hw_rc;
    *attn_accel_cycles += stats.hw_e2e_cycles;
    *attn_score_cycles += stats.score_cycles;
    *attn_value_cycles += stats.value_cycles;
    *attn_e2e_cycles +=
        stats.q_pack_cycles +
        stats.k_pack_cycles +
        stats.v_pack_cycles +
        stats.hw_e2e_cycles +
        stats.copy_out_cycles;
    if (rc != FPGA_SAFE_ATTN_OK) {
      return rc;
    }
  }

  return FPGA_SAFE_ATTN_OK;
}

static int compare_hw_to_expected(
    const transformer_block_case_t *tc,
    float *max_abs_diff) {
  int mismatches = 0;
  *max_abs_diff = 0.0f;

  for (int r = 0; r < tc->seq_len; r++) {
    for (int c = 0; c < tc->d_model; c++) {
      const float ref = bf16_to_float(tensor_at(tc->expected, r, c, tc->d_model));
      const float hw = bf16_to_float(hw_output[r][c]);
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

static float max_cpu_vs_expected_diff(const transformer_block_case_t *tc) {
  float max_abs_diff = 0.0f;
  for (int r = 0; r < tc->seq_len; r++) {
    for (int c = 0; c < tc->d_model; c++) {
      const float ref = bf16_to_float(tensor_at(tc->expected, r, c, tc->d_model));
      const float diff = fabsf(sw_output[r][c] - ref);
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
      }
    }
  }
  return max_abs_diff;
}

int main(void) {
  const ws_gemm_workspace_t gemm_workspace =
      ws_gemm8_make_workspace(
          gemm_a_tiles,
          gemm_b_tiles,
          gemm_c_words,
          TRANSFORMER_BLOCK_MAX_SEQ_LEN,
          TRANSFORMER_BLOCK_MAX_D_MODEL,
          TRANSFORMER_BLOCK_MAX_D_MODEL);
  const fpga_safe_attention_workspace_t attn_workspace =
      fpga_safe_attention_make_workspace(
          q_tiles,
          k_tiles,
          v_tiles,
          attn_out_words,
          TRANSFORMER_BLOCK_MAX_SEQ_LEN,
          TRANSFORMER_BLOCK_MAX_SEQ_LEN,
          TRANSFORMER_BLOCK_MAX_HEAD_DIM,
          TRANSFORMER_BLOCK_MAX_HEAD_DIM);

  printf("=== Dual-RoCC Transformer Attention Block Test ===\n");
  printf("OPCODE_INFO,matmul=%d,attention=%d\n", SA_MATMUL_OPCODE, SA_ATTN_OPCODE);
  printf("CSV_HEADER,case,name,seq_len,d_model,n_heads,head_dim,cpu_cycles,hw_e2e_cycles,q_proj_cycles,k_proj_cycles,v_proj_cycles,attn_e2e_cycles,attn_accel_cycles,attn_score_cycles,attn_value_cycles,out_proj_cycles,hw_rc,raw_hw_rc,speedup_x100,hw_max_abs_diff_x100000,cpu_ref_max_abs_diff_x100000,mismatches\n");

  int total_mismatches = 0;
  for (int i = 0; i < transformer_block_case_count; i++) {
    const transformer_block_case_t *tc = &transformer_block_cases[i];
    memset(q_proj, 0, sizeof(q_proj));
    memset(k_proj, 0, sizeof(k_proj));
    memset(v_proj, 0, sizeof(v_proj));
    memset(context, 0, sizeof(context));
    memset(hw_output, 0, sizeof(hw_output));

    const uint64_t cpu_start = ws_read_cycles();
    sw_transformer_block_reference(tc);
    const uint64_t cpu_cycles = ws_read_cycles() - cpu_start;
    const float cpu_ref_max_abs_diff = max_cpu_vs_expected_diff(tc);

    ws_gemm_stats_t q_stats;
    ws_gemm_stats_t k_stats;
    ws_gemm_stats_t v_stats;
    ws_gemm_stats_t out_stats;
    memset(&q_stats, 0, sizeof(q_stats));
    memset(&k_stats, 0, sizeof(k_stats));
    memset(&v_stats, 0, sizeof(v_stats));
    memset(&out_stats, 0, sizeof(out_stats));
    int hw_rc = run_projection(tc, tc->wq, q_proj, &gemm_workspace, &q_stats);
    if (hw_rc == WS_GEMM_OK) {
      hw_rc = run_projection(tc, tc->wk, k_proj, &gemm_workspace, &k_stats);
    }
    if (hw_rc == WS_GEMM_OK) {
      hw_rc = run_projection(tc, tc->wv, v_proj, &gemm_workspace, &v_stats);
    }

    uint64_t attn_e2e_cycles = 0;
    uint64_t attn_accel_cycles = 0;
    uint64_t attn_score_cycles = 0;
    uint64_t attn_value_cycles = 0;
    uint64_t raw_hw_rc = 0;
    if (hw_rc == WS_GEMM_OK) {
      hw_rc = run_attention_heads(
          tc,
          &attn_workspace,
          &attn_e2e_cycles,
          &attn_accel_cycles,
          &attn_score_cycles,
          &attn_value_cycles,
          &raw_hw_rc);
    }
    if (hw_rc == FPGA_SAFE_ATTN_OK || hw_rc == WS_GEMM_OK) {
      hw_rc = run_output_projection(tc, &gemm_workspace, &out_stats);
    }

    const uint64_t hw_e2e_cycles =
        q_stats.hw_e2e_cycles +
        k_stats.hw_e2e_cycles +
        v_stats.hw_e2e_cycles +
        attn_e2e_cycles +
        out_stats.hw_e2e_cycles;
    const uint64_t speedup_x100 =
        hw_e2e_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_e2e_cycles;

    float hw_max_abs_diff = 0.0f;
    int mismatches = 1;
    if (hw_rc == WS_GEMM_OK) {
      mismatches = compare_hw_to_expected(tc, &hw_max_abs_diff);
    }
    total_mismatches += mismatches;

    printf("CSV_DATA,%d,%s,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%d,%lu,%lu,%lu,%lu,%d\n",
           i,
           tc->name,
           tc->seq_len,
           tc->d_model,
           tc->n_heads,
           tc->head_dim,
           (unsigned long)cpu_cycles,
           (unsigned long)hw_e2e_cycles,
           (unsigned long)q_stats.hw_e2e_cycles,
           (unsigned long)k_stats.hw_e2e_cycles,
           (unsigned long)v_stats.hw_e2e_cycles,
           (unsigned long)attn_e2e_cycles,
           (unsigned long)attn_accel_cycles,
           (unsigned long)attn_score_cycles,
           (unsigned long)attn_value_cycles,
           (unsigned long)out_stats.hw_e2e_cycles,
           hw_rc,
           (unsigned long)raw_hw_rc,
           (unsigned long)speedup_x100,
           (unsigned long)(hw_max_abs_diff * 100000.0f),
           (unsigned long)(cpu_ref_max_abs_diff * 100000.0f),
           mismatches);
  }

  if (total_mismatches == 0) {
    printf("PASS: dual-RoCC transformer block matched PyTorch tolerance.\n");
  } else {
    printf("FAIL: total transformer block mismatches = %d\n", total_mismatches);
  }

  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/transformer_block_cases.c"
