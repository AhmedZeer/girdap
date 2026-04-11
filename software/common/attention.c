#include "attention.h"

#include "bfloat16_utils.h"

#include <string.h>

static int validate_dims(
    int ldq,
    int ldk_t,
    int ldv,
    int ldout,
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols) {
  if (ldq < d_k || ldk_t < kv_rows || ldv < value_cols || ldout < value_cols) {
    return 0;
  }
  if (q_rows < 0 || kv_rows < 0 || d_k < 0 || value_cols < 0) {
    return 0;
  }
  if (q_rows > 0 && (kv_rows == 0 || d_k == 0 || value_cols == 0)) {
    return 0;
  }
  return 1;
}

static int validate_workspace(
    const attention_workspace_t *workspace,
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols) {
  if (workspace == 0) {
    return 0;
  }
  if (workspace->scores_bf16 == 0 || workspace->probs_bf16 == 0) {
    return 0;
  }
  if (q_rows > workspace->max_q_rows || kv_rows > workspace->max_kv_rows ||
      d_k > workspace->max_d_k || value_cols > workspace->max_value_cols) {
    return 0;
  }
  return 1;
}

static int validate_params(
    const attention_params_t *params,
    int q_rows,
    int kv_rows) {
  if (params == 0 || params->mask_bf16 == 0) {
    return 1;
  }
  if (q_rows == 0 || kv_rows == 0) {
    return 1;
  }
  return params->mask_stride >= kv_rows;
}

static void apply_score_modifiers(
    uint16_t *scores_bf16,
    int rows,
    int cols,
    int stride,
    const attention_params_t *params) {
  const float scale = (params != 0) ? params->scale : 1.0f;
  const uint16_t *mask_bf16 = (params != 0) ? params->mask_bf16 : 0;
  const int mask_stride = (params != 0) ? params->mask_stride : 0;
  const int causal = (params != 0) ? params->causal : 0;
  const int query_position_base = (params != 0) ? params->query_position_base : 0;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float value = bf16_to_float(scores_bf16[r * stride + c]) * scale;
      if (mask_bf16 != 0) {
        value += bf16_to_float(mask_bf16[r * mask_stride + c]);
      }
      if (causal && c > query_position_base + r) {
        value = -1.0e9f;
      }
      scores_bf16[r * stride + c] = float_to_bf16(value);
    }
  }
}

void attention_clear_stats(attention_stats_t *stats) {
  if (stats != 0) {
    memset(stats, 0, sizeof(*stats));
  }
}

int attention_bf16(
    const uint16_t *Q,
    int ldq,
    const uint16_t *K_t,
    int ldk_t,
    const uint16_t *V,
    int ldv,
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols,
    uint16_t *output,
    int ldout,
    const attention_workspace_t *workspace,
    const attention_params_t *params,
    attention_stats_t *stats) {
  ws_gemm_packed_b_t packed_k_t;
  ws_gemm_packed_b_t packed_v;

  if (!validate_dims(ldq, ldk_t, ldv, ldout, q_rows, kv_rows, d_k, value_cols)) {
    return ATTENTION_ERR_BAD_DIMS;
  }
  if ((q_rows > 0 && d_k > 0 && Q == 0) || (kv_rows > 0 && d_k > 0 && K_t == 0) ||
      (kv_rows > 0 && value_cols > 0 && V == 0) || (q_rows > 0 && value_cols > 0 && output == 0)) {
    return ATTENTION_ERR_NULL;
  }
  if (!validate_workspace(workspace, q_rows, kv_rows, d_k, value_cols)) {
    return ATTENTION_ERR_WORKSPACE;
  }
  if (!validate_params(params, q_rows, kv_rows)) {
    return ATTENTION_ERR_BAD_DIMS;
  }

  attention_clear_stats(stats);
  if (q_rows == 0 || value_cols == 0) {
    return ATTENTION_OK;
  }

  const int measure = (stats != 0);
  const uint64_t total_start = measure ? attention_read_cycles() : 0;

  const int qk_pack_rc = ws_gemm_prepare_packed_b_bf16(
      K_t,
      ldk_t,
      kv_rows,
      d_k,
      workspace->qk_workspace.b_tiles,
      workspace->qk_workspace.max_n,
      workspace->qk_workspace.max_k,
      &packed_k_t,
      measure ? &stats->qk_pack_cycles : 0);
  if (qk_pack_rc != WS_GEMM_OK) {
    if (measure) {
      stats->total_cycles = attention_read_cycles() - total_start;
    }
    return ATTENTION_ERR_QK;
  }

  const int qk_rc = ws_gemm_bf16_prepacked_b(
      Q,
      ldq,
      &packed_k_t,
      q_rows,
      workspace->scores_bf16,
      kv_rows,
      &workspace->qk_workspace,
      measure ? &stats->qk_matmul : 0);
  if (qk_rc != WS_GEMM_OK) {
    if (measure) {
      stats->total_cycles = attention_read_cycles() - total_start;
    }
    return ATTENTION_ERR_QK;
  }

  if (measure) {
    const uint64_t score_post_start = attention_read_cycles();
    apply_score_modifiers(workspace->scores_bf16, q_rows, kv_rows, kv_rows, params);
    stats->score_post_cycles = attention_read_cycles() - score_post_start;
  } else {
    apply_score_modifiers(workspace->scores_bf16, q_rows, kv_rows, kv_rows, params);
  }

  const int softmax_rc = online_softmax_bf16_rows(
      workspace->scores_bf16,
      q_rows,
      kv_rows,
      kv_rows,
      workspace->probs_bf16,
      kv_rows,
      measure ? &stats->softmax : 0);
  if (softmax_rc != ONLINE_SOFTMAX_OK) {
    if (measure) {
      stats->total_cycles = attention_read_cycles() - total_start;
    }
    return ATTENTION_ERR_SOFTMAX;
  }

  const int pv_pack_rc = ws_gemm_prepare_packed_b_bf16(
      V,
      ldv,
      value_cols,
      kv_rows,
      workspace->pv_workspace.b_tiles,
      workspace->pv_workspace.max_n,
      workspace->pv_workspace.max_k,
      &packed_v,
      measure ? &stats->pv_pack_cycles : 0);
  if (pv_pack_rc != WS_GEMM_OK) {
    if (measure) {
      stats->total_cycles = attention_read_cycles() - total_start;
    }
    return ATTENTION_ERR_PV;
  }

  const int pv_rc = ws_gemm_bf16_prepacked_b(
      workspace->probs_bf16,
      kv_rows,
      &packed_v,
      q_rows,
      output,
      ldout,
      &workspace->pv_workspace,
      measure ? &stats->pv_matmul : 0);
  if (pv_rc != WS_GEMM_OK) {
    if (measure) {
      stats->total_cycles = attention_read_cycles() - total_start;
    }
    return ATTENTION_ERR_PV;
  }

  if (measure) {
    stats->total_cycles = attention_read_cycles() - total_start;
  }

  return ATTENTION_OK;
}
