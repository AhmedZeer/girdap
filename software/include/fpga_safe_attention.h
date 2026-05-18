#ifndef GIRDAP_FPGA_SAFE_ATTENTION_H
#define GIRDAP_FPGA_SAFE_ATTENTION_H

#include "ws_gemm.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FPGA_SAFE_ATTN_TILE_ROWS WS_GEMM8_ROWS
#define FPGA_SAFE_ATTN_TILE_COLS WS_GEMM8_COLS
#define FPGA_SAFE_ATTN_OUT_WORDS WS_GEMM8_C_TILE_WORDS
#define FPGA_SAFE_ATTN_Q_TILE_WORDS(max_q_rows, max_d_k) \
  WS_GEMM8_A_TILE_WORDS(max_q_rows, max_d_k)
#define FPGA_SAFE_ATTN_K_TILE_WORDS(max_kv_rows, max_d_k) \
  WS_GEMM8_B_TILE_WORDS(max_kv_rows, max_d_k)
#define FPGA_SAFE_ATTN_V_TILE_WORDS(max_value_cols, max_kv_rows) \
  WS_GEMM8_B_TILE_WORDS(max_value_cols, max_kv_rows)

typedef enum {
  FPGA_SAFE_ATTN_OK = 0,
  FPGA_SAFE_ATTN_ERR_BAD_DIMS = 1,
  FPGA_SAFE_ATTN_ERR_WORKSPACE = 2,
  FPGA_SAFE_ATTN_ERR_PACK = 3,
  FPGA_SAFE_ATTN_ERR_RUN = 4,
} fpga_safe_attention_status_t;

typedef struct {
  uint64_t q_pack_cycles;
  uint64_t k_pack_cycles;
  uint64_t v_pack_cycles;
  uint64_t hw_e2e_cycles;
  uint64_t score_cycles;
  uint64_t value_cycles;
  uint64_t debug_dump_cycles;
  uint64_t copy_out_cycles;
  uint64_t raw_hw_rc;
} fpga_safe_attention_stats_t;

typedef struct {
  uint64_t *q_tiles;
  uint64_t *k_tiles;
  uint64_t *v_tiles;
  uint64_t *out_words;
  int max_q_rows;
  int max_kv_rows;
  int max_d_k;
  int max_value_cols;
} fpga_safe_attention_workspace_t;

static inline fpga_safe_attention_workspace_t fpga_safe_attention_make_workspace(
    uint64_t *q_tiles,
    uint64_t *k_tiles,
    uint64_t *v_tiles,
    uint64_t *out_words,
    int max_q_rows,
    int max_kv_rows,
    int max_d_k,
    int max_value_cols) {
  fpga_safe_attention_workspace_t workspace = {
      .q_tiles = q_tiles,
      .k_tiles = k_tiles,
      .v_tiles = v_tiles,
      .out_words = out_words,
      .max_q_rows = max_q_rows,
      .max_kv_rows = max_kv_rows,
      .max_d_k = max_d_k,
      .max_value_cols = max_value_cols,
  };
  return workspace;
}

int fpga_safe_attention_bf16(
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
    uint16_t scale_bf16,
    uint16_t *output,
    int ldout,
    const fpga_safe_attention_workspace_t *workspace,
    fpga_safe_attention_stats_t *stats);

int fpga_safe_attention_bf16_hwpack(
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
    uint16_t scale_bf16,
    uint16_t *output,
    int ldout,
    const fpga_safe_attention_workspace_t *workspace,
    fpga_safe_attention_stats_t *stats);

int fpga_safe_attention_bf16_hwpack_debug_intermediates(
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
    uint16_t scale_bf16,
    uint16_t *output,
    int ldout,
    int64_t *score_fixed,
    uint64_t *prob_fixed,
    int ldintermediate,
    const fpga_safe_attention_workspace_t *workspace,
    fpga_safe_attention_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif
