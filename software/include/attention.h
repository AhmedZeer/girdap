#ifndef TOYROCC_ATTENTION_H
#define TOYROCC_ATTENTION_H

#include "online_softmax.h"
#include "ws_gemm.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ATTENTION_OK = 0,
  ATTENTION_ERR_BAD_DIMS = 1,
  ATTENTION_ERR_NULL = 2,
  ATTENTION_ERR_WORKSPACE = 3,
  ATTENTION_ERR_QK = 4,
  ATTENTION_ERR_SOFTMAX = 5,
  ATTENTION_ERR_PV = 6,
} attention_status_t;

typedef struct {
  float scale;
  const uint16_t *mask_bf16;
  int mask_stride;
  int causal;
  int query_position_base;
} attention_params_t;

typedef struct {
  ws_gemm_workspace_t qk_workspace;
  ws_gemm_workspace_t pv_workspace;
  uint16_t *scores_bf16;
  uint16_t *probs_bf16;
  int max_q_rows;
  int max_kv_rows;
  int max_d_k;
  int max_value_cols;
} attention_workspace_t;

typedef struct {
  uint64_t total_cycles;
  uint64_t qk_pack_cycles;
  uint64_t score_post_cycles;
  uint64_t pv_pack_cycles;
  ws_gemm_stats_t qk_matmul;
  online_softmax_stats_t softmax;
  ws_gemm_stats_t pv_matmul;
} attention_stats_t;

static inline uint64_t attention_read_cycles(void) {
  uint64_t cycles;
  asm volatile("fence; rdcycle %0" : "=r"(cycles));
  return cycles;
}

static inline attention_params_t attention_default_params(void) {
  attention_params_t params = {
      .scale = 1.0f,
      .mask_bf16 = 0,
      .mask_stride = 0,
      .causal = 0,
      .query_position_base = 0,
  };
  return params;
}

static inline attention_workspace_t attention_make_workspace(
    uint64_t *qk_a_tiles,
    uint64_t *qk_b_tiles,
    uint64_t *qk_c_words,
    uint64_t *pv_a_tiles,
    uint64_t *pv_b_tiles,
    uint64_t *pv_c_words,
    uint16_t *scores_bf16,
    uint16_t *probs_bf16,
    int max_q_rows,
    int max_kv_rows,
    int max_d_k,
    int max_value_cols) {
  attention_workspace_t workspace = {
      .qk_workspace =
          ws_gemm_make_workspace(qk_a_tiles, qk_b_tiles, qk_c_words, max_q_rows, max_kv_rows, max_d_k),
      .pv_workspace =
          ws_gemm_make_workspace(pv_a_tiles, pv_b_tiles, pv_c_words, max_q_rows, max_value_cols, max_kv_rows),
      .scores_bf16 = scores_bf16,
      .probs_bf16 = probs_bf16,
      .max_q_rows = max_q_rows,
      .max_kv_rows = max_kv_rows,
      .max_d_k = max_d_k,
      .max_value_cols = max_value_cols,
  };
  return workspace;
}

static inline uint64_t attention_accelerator_cycles(const attention_stats_t *stats) {
  if (stats == 0) {
    return 0;
  }
  return stats->qk_matmul.hw_e2e_cycles + stats->softmax.hw_e2e_cycles +
         stats->pv_matmul.hw_e2e_cycles;
}

void attention_clear_stats(attention_stats_t *stats);

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
    attention_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif
