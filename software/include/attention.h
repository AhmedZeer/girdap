#ifndef GIRDAP_ATTENTION_H
#define GIRDAP_ATTENTION_H

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

#define ATTENTION_FUSED_OPCODE 0
#define ATTENTION_FUSED_FUNCT_SET_QK_ADDRS 0
#define ATTENTION_FUSED_FUNCT_SET_VOUT_ADDRS 1
#define ATTENTION_FUSED_FUNCT_SET_DIMS 2
#define ATTENTION_FUSED_FUNCT_SET_PARAMS 3
#define ATTENTION_FUSED_FUNCT_RUN 4
#define ATTENTION_FUSED_FUNCT_STATUS 5
#define ATTENTION_FUSED_FUNCT_READ_COUNTER 6
#define ATTENTION_FUSED_FUNCT_CLEAR_COUNTERS 7

#define ATTENTION_FUSED_TILE_ROWS 4
#define ATTENTION_FUSED_TILE_COLS 4
#define ATTENTION_FUSED_MAX_K 256
#define ATTENTION_FUSED_BF16S_PER_WORD ((int)(sizeof(uint64_t) / sizeof(uint16_t)))
#define ATTENTION_FUSED_Q_TILE_WORDS_FOR(tile_rows, max_q_rows, max_d_k) \
  WS_GEMM_A_TILE_WORDS_FOR(tile_rows, max_q_rows, max_d_k)
#define ATTENTION_FUSED_K_TILE_WORDS_FOR(tile_cols, max_kv_rows, fused_max_k) \
  (((((max_kv_rows) + (tile_cols) - 1) / (tile_cols))) * (fused_max_k))
#define ATTENTION_FUSED_V_TILE_WORDS_FOR(max_kv_rows) (max_kv_rows)
#define ATTENTION_FUSED_OUT_TILE_WORDS_FOR(tile_rows, tile_cols) \
  (((tile_rows) * (tile_cols) + ATTENTION_FUSED_BF16S_PER_WORD - 1) / ATTENTION_FUSED_BF16S_PER_WORD)

typedef enum {
  ATTENTION_FUSED_OK = 0,
  ATTENTION_FUSED_ERR_BAD_DIMS = 1,
  ATTENTION_FUSED_ERR_NULL = 2,
  ATTENTION_FUSED_ERR_WORKSPACE = 3,
  ATTENTION_FUSED_ERR_UNSUPPORTED = 4,
  ATTENTION_FUSED_ERR_Q_PACK = 5,
  ATTENTION_FUSED_ERR_K_PACK = 6,
  ATTENTION_FUSED_ERR_V_PACK = 7,
  ATTENTION_FUSED_ERR_RUN = 8,
} attention_fused_status_t;

enum {
  ATTN_FUSED_PERF_BUSY_CYCLES = 0,
  ATTN_FUSED_PERF_RUN_CMDS,
  ATTN_FUSED_PERF_TL_READ_REQS,
  ATTN_FUSED_PERF_TL_WRITE_REQS,
  ATTN_FUSED_PERF_WAIT_Q_FILL_CYCLES,
  ATTN_FUSED_PERF_WAIT_K_FILL_CYCLES,
  ATTN_FUSED_PERF_QK_COMPUTE_CYCLES,
  ATTN_FUSED_PERF_SCORE_POST_CYCLES,
  ATTN_FUSED_PERF_SOFTMAX_PASS1_CYCLES,
  ATTN_FUSED_PERF_SOFTMAX_PASS2_CYCLES,
  ATTN_FUSED_PERF_WAIT_V_FILL_CYCLES,
  ATTN_FUSED_PERF_PV_COMPUTE_CYCLES,
  ATTN_FUSED_PERF_WAIT_PUT_CYCLES,
  ATTN_FUSED_PERF_COUNT
};

static const char *const attention_fused_perf_counter_names[ATTN_FUSED_PERF_COUNT] = {
    "busy_cycles",
    "run_cmds",
    "tl_read_reqs",
    "tl_write_reqs",
    "wait_q_fill_cycles",
    "wait_k_fill_cycles",
    "qk_compute_cycles",
    "score_post_cycles",
    "softmax_pass1_cycles",
    "softmax_pass2_cycles",
    "wait_v_fill_cycles",
    "pv_compute_cycles",
    "wait_put_cycles",
};

typedef struct {
  uint64_t *q_tiles;
  uint64_t *k_tiles;
  uint64_t *v_tiles;
  uint64_t *out_words;
  int max_q_rows;
  int max_kv_rows;
  int max_d_k;
  int max_value_cols;
  int fused_max_k;
  int tile_rows;
  int tile_cols;
} attention_fused_workspace_t;

typedef struct {
  uint64_t q_pack_cycles;
  uint64_t k_pack_cycles;
  uint64_t v_pack_cycles;
  uint64_t copy_out_cycles;
} attention_fused_stage_stats_t;

typedef struct {
  uint64_t total_cycles;
  uint64_t hw_e2e_cycles;
  attention_fused_stage_stats_t stage;
  uint64_t perf[ATTN_FUSED_PERF_COUNT];
} attention_fused_stats_t;

static inline attention_fused_workspace_t attention_make_fused_workspace_tiled(
    uint64_t *q_tiles,
    uint64_t *k_tiles,
    uint64_t *v_tiles,
    uint64_t *out_words,
    int max_q_rows,
    int max_kv_rows,
    int max_d_k,
    int max_value_cols,
    int fused_max_k,
    int tile_rows,
    int tile_cols) {
  attention_fused_workspace_t workspace = {
      .q_tiles = q_tiles,
      .k_tiles = k_tiles,
      .v_tiles = v_tiles,
      .out_words = out_words,
      .max_q_rows = max_q_rows,
      .max_kv_rows = max_kv_rows,
      .max_d_k = max_d_k,
      .max_value_cols = max_value_cols,
      .fused_max_k = fused_max_k,
      .tile_rows = tile_rows,
      .tile_cols = tile_cols,
  };
  return workspace;
}

static inline attention_fused_workspace_t attention_make_fused_workspace(
    uint64_t *q_tiles,
    uint64_t *k_tiles,
    uint64_t *v_tiles,
    uint64_t *out_words,
    int max_q_rows,
    int max_kv_rows,
    int max_d_k,
    int max_value_cols) {
  return attention_make_fused_workspace_tiled(
      q_tiles,
      k_tiles,
      v_tiles,
      out_words,
      max_q_rows,
      max_kv_rows,
      max_d_k,
      max_value_cols,
      ATTENTION_FUSED_MAX_K,
      ATTENTION_FUSED_TILE_ROWS,
      ATTENTION_FUSED_TILE_COLS);
}

static inline uint64_t attention_fused_set_qk_addrs(
    const uint64_t *q_base,
    const uint64_t *k_tiles_base) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      q_base,
      k_tiles_base,
      ATTENTION_FUSED_FUNCT_SET_QK_ADDRS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t attention_fused_set_vout_addrs(
    const uint64_t *v_base,
    uint64_t *out_base) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      v_base,
      out_base,
      ATTENTION_FUSED_FUNCT_SET_VOUT_ADDRS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t attention_fused_set_dims(
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols) {
  const uint64_t packed_dims =
      (((uint64_t)(uint16_t)value_cols) << 48) |
      (((uint64_t)(uint16_t)d_k) << 32) |
      (((uint64_t)(uint16_t)kv_rows) << 16) |
      (uint64_t)(uint16_t)q_rows;
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      packed_dims,
      0,
      ATTENTION_FUSED_FUNCT_SET_DIMS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t attention_fused_set_params_bf16(
    uint16_t scale_bf16,
    int causal,
    int query_position_base) {
  const uint64_t packed_params =
      ((uint64_t)(causal ? 1u : 0u) << 16) | (uint64_t)scale_bf16;
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      packed_params,
      (uint64_t)query_position_base,
      ATTENTION_FUSED_FUNCT_SET_PARAMS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t attention_fused_run(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      0,
      0,
      ATTENTION_FUSED_FUNCT_RUN);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t attention_fused_status(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      0,
      0,
      ATTENTION_FUSED_FUNCT_STATUS);
  return rd;
}

static inline uint64_t attention_fused_read_counter(uint64_t counter_id) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      counter_id,
      0,
      ATTENTION_FUSED_FUNCT_READ_COUNTER);
  return rd;
}

static inline void attention_fused_clear_counters(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      ATTENTION_FUSED_OPCODE,
      rd,
      0,
      0,
      ATTENTION_FUSED_FUNCT_CLEAR_COUNTERS);
  asm volatile("fence rw, rw" ::: "memory");
}

static inline void attention_fused_read_perf_counters(
    uint64_t counters[ATTN_FUSED_PERF_COUNT]) {
  for (int i = 0; i < ATTN_FUSED_PERF_COUNT; i++) {
    counters[i] = attention_fused_read_counter((uint64_t)i);
  }
}

static inline uint64_t attention_fused_accelerator_cycles(
    const attention_fused_stats_t *stats) {
  return (stats != 0) ? stats->hw_e2e_cycles : 0;
}

void attention_fused_clear_stats(attention_fused_stats_t *stats);

int attention_fused_bf16(
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
    const attention_fused_workspace_t *workspace,
    const attention_params_t *params,
    attention_fused_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif
