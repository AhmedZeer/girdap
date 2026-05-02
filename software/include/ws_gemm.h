#ifndef TOYROCC_WS_GEMM_H
#define TOYROCC_WS_GEMM_H

#include "systolic_ws.h"

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define WS_GEMM_RESTRICT __restrict__
#else
#define WS_GEMM_RESTRICT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define WS_GEMM_ROWS 4
#define WS_GEMM_COLS 4
#define WS_GEMM_C_TILE_WORDS_FOR(tile_rows, tile_cols) ((tile_rows) * (tile_cols))
#define WS_GEMM_C_TILE_WORDS WS_GEMM_C_TILE_WORDS_FOR(WS_GEMM_ROWS, WS_GEMM_COLS)
#define WS_GEMM_BF16_FIXED_FRAC_BITS 8
#define WS_GEMM_BF16_ACC_FRAC_BITS (2 * WS_GEMM_BF16_FIXED_FRAC_BITS)
#define WS_GEMM_A_TILE_WORDS_FOR(tile_rows, max_m, max_k) \
  (((((max_m) + (tile_rows) - 1) / (tile_rows))) * (max_k))
#define WS_GEMM_B_TILE_WORDS_FOR(tile_cols, max_n, max_k) \
  (((((max_n) + (tile_cols) - 1) / (tile_cols))) * (max_k))
#define WS_GEMM_A_TILE_WORDS(max_m, max_k) \
  WS_GEMM_A_TILE_WORDS_FOR(WS_GEMM_ROWS, max_m, max_k)
#define WS_GEMM_B_TILE_WORDS(max_n, max_k) \
  WS_GEMM_B_TILE_WORDS_FOR(WS_GEMM_COLS, max_n, max_k)

#define WS_GEMM8_ROWS 8
#define WS_GEMM8_COLS 8
// Default is one 128-bit cache beat. Override to 4 for a 256-bit bus build.
#ifndef WS_GEMM8_BEAT_WORDS
#define WS_GEMM8_BEAT_WORDS 2
#endif
#if WS_GEMM8_BEAT_WORDS < 2
#error "WS_GEMM8_BEAT_WORDS must be at least 2 for eight BF16 lanes"
#endif
#define WS_GEMM8_C_TILE_WORDS 16
#define WS_GEMM8_A_TILE_WORDS(max_m, max_k) \
  (((((max_m) + WS_GEMM8_ROWS - 1) / WS_GEMM8_ROWS)) * (max_k) * WS_GEMM8_BEAT_WORDS)
#define WS_GEMM8_B_TILE_WORDS(max_n, max_k) \
  (((((max_n) + WS_GEMM8_COLS - 1) / WS_GEMM8_COLS)) * (max_k) * WS_GEMM8_BEAT_WORDS)

typedef struct {
  uint64_t word[WS_GEMM8_BEAT_WORDS];
} ws_gemm8_beat_t;

typedef enum {
  WS_GEMM_OK = 0,
  WS_GEMM_ERR_BAD_DIMS = 1,
  WS_GEMM_ERR_WORKSPACE = 2,
  WS_GEMM_ERR_PRELOAD = 3,
  WS_GEMM_ERR_RUN = 4,
} ws_gemm_status_t;

typedef struct {
  uint64_t pack_a_cycles;
  uint64_t pack_b_cycles;
  uint64_t preload_cycles;
  uint64_t run_cycles;
  uint64_t copy_out_cycles;
} ws_gemm_stage_stats_t;

typedef struct {
  uint64_t hw_e2e_cycles;
  ws_gemm_stage_stats_t stage;
  uint64_t perf[WS_PERF_COUNT];
} ws_gemm_stats_t;

typedef struct {
  uint64_t *a_tiles;
  uint64_t *b_tiles;
  uint64_t *c_words;
  int max_m;
  int max_n;
  int max_k;
  int tile_rows;
  int tile_cols;
} ws_gemm_workspace_t;

static inline ws_gemm_workspace_t ws_gemm_make_workspace_tiled(
    uint64_t *a_tiles,
    uint64_t *b_tiles,
    uint64_t *c_words,
    int max_m,
    int max_n,
    int max_k,
    int tile_rows,
    int tile_cols) {
  ws_gemm_workspace_t workspace = {
      .a_tiles = a_tiles,
      .b_tiles = b_tiles,
      .c_words = c_words,
      .max_m = max_m,
      .max_n = max_n,
      .max_k = max_k,
      .tile_rows = tile_rows,
      .tile_cols = tile_cols,
  };
  return workspace;
}

static inline uint64_t *ws_gemm_a_tile_ptr(
    uint64_t *a_tiles,
    int max_k,
    int m0,
    int tile_rows) {
  return a_tiles + (m0 / tile_rows) * max_k;
}

static inline const uint64_t *ws_gemm_a_tile_ptr_const(
    const uint64_t *a_tiles,
    int max_k,
    int m0,
    int tile_rows) {
  return a_tiles + (m0 / tile_rows) * max_k;
}

static inline uint64_t *ws_gemm_workspace_a_tile_ptr(
    const ws_gemm_workspace_t *workspace,
    int m0) {
  return ws_gemm_a_tile_ptr(
      workspace->a_tiles, workspace->max_k, m0, workspace->tile_rows);
}

static inline const uint64_t *ws_gemm_workspace_a_tile_ptr_const(
    const ws_gemm_workspace_t *workspace,
    int m0) {
  return ws_gemm_a_tile_ptr_const(
      workspace->a_tiles, workspace->max_k, m0, workspace->tile_rows);
}

static inline ws_gemm_workspace_t ws_gemm_make_workspace(
    uint64_t *a_tiles,
    uint64_t *b_tiles,
    uint64_t *c_words,
    int max_m,
    int max_n,
    int max_k) {
  return ws_gemm_make_workspace_tiled(
      a_tiles, b_tiles, c_words, max_m, max_n, max_k, WS_GEMM_ROWS, WS_GEMM_COLS);
}

static inline ws_gemm_workspace_t ws_gemm8_make_workspace(
    uint64_t *a_tiles,
    uint64_t *b_tiles,
    uint64_t *c_words,
    int max_m,
    int max_n,
    int max_k) {
  return ws_gemm_make_workspace_tiled(
      a_tiles, b_tiles, c_words, max_m, max_n, max_k, WS_GEMM8_ROWS, WS_GEMM8_COLS);
}

typedef struct {
  const uint64_t *tiles;
  int max_n;
  int max_k;
  int N;
  int K;
  int tile_cols;
} ws_gemm_packed_b_t;

static inline ws_gemm_packed_b_t ws_gemm_make_packed_b_tiled(
    const uint64_t *tiles,
    int max_n,
    int max_k,
    int N,
    int K,
    int tile_cols) {
  ws_gemm_packed_b_t packed_b = {
      .tiles = tiles,
      .max_n = max_n,
      .max_k = max_k,
      .N = N,
      .K = K,
      .tile_cols = tile_cols,
  };
  return packed_b;
}

static inline ws_gemm_packed_b_t ws_gemm_make_packed_b(
    const uint64_t *tiles,
    int max_n,
    int max_k,
    int N,
    int K) {
  return ws_gemm_make_packed_b_tiled(tiles, max_n, max_k, N, K, WS_GEMM_COLS);
}

static inline uint64_t *ws_gemm_b_tile_ptr(
    uint64_t *b_tiles,
    int max_k,
    int n0,
    int tile_cols) {
  return b_tiles + (n0 / tile_cols) * max_k;
}

static inline const uint64_t *ws_gemm_b_tile_ptr_const(
    const uint64_t *b_tiles,
    int max_k,
    int n0,
    int tile_cols) {
  return b_tiles + (n0 / tile_cols) * max_k;
}

static inline const uint64_t *ws_gemm_packed_b_tile_ptr(
    const ws_gemm_packed_b_t *packed_b,
    int n0) {
  return ws_gemm_b_tile_ptr_const(
      packed_b->tiles, packed_b->max_k, n0, packed_b->tile_cols);
}

void ws_gemm_clear_stats(ws_gemm_stats_t *stats);

int ws_gemm_pack_a_u16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    int M,
    int K,
    uint64_t *WS_GEMM_RESTRICT a_tiles,
    int max_m,
    int max_k,
    uint64_t *pack_cycles);

int ws_gemm_pack_a_u16_tiled(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    int M,
    int K,
    uint64_t *WS_GEMM_RESTRICT a_tiles,
    int max_m,
    int max_k,
    int tile_rows,
    uint64_t *pack_cycles);

int ws_gemm_pack_a_bf16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    int M,
    int K,
    uint64_t *WS_GEMM_RESTRICT a_tiles,
    int max_m,
    int max_k,
    uint64_t *pack_cycles);

int ws_gemm_pack_a_bf16_tiled(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    int M,
    int K,
    uint64_t *WS_GEMM_RESTRICT a_tiles,
    int max_m,
    int max_k,
    int tile_rows,
    uint64_t *pack_cycles);

int ws_gemm_pack_b_u16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    uint64_t *pack_cycles);

int ws_gemm_pack_b_u16_tiled(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    int tile_cols,
    uint64_t *pack_cycles);

int ws_gemm_prepare_packed_b_u16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    uint64_t *pack_cycles);

int ws_gemm_prepare_packed_b_u16_tiled(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    int tile_cols,
    ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    uint64_t *pack_cycles);

int ws_gemm_prepare_packed_b_bf16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    uint64_t *pack_cycles);

int ws_gemm_prepare_packed_b_bf16_tiled(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    int tile_cols,
    ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    uint64_t *pack_cycles);

int ws_gemm_u16_prepacked_b(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    int M,
    uint32_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats);

int ws_gemm_bf16_prepacked_b(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    int M,
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats);

int ws_gemm_u16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    uint32_t *WS_GEMM_RESTRICT C,
    int ldc,
    int M,
    int N,
    int K,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats);

int ws_gemm_bf16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    int M,
    int N,
    int K,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats);

int ws_gemm8_pack_a_bf16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    int M,
    int K,
    uint64_t *WS_GEMM_RESTRICT a_tiles,
    int max_m,
    int max_k,
    uint64_t *pack_cycles);

int ws_gemm8_pack_b_bf16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    uint64_t *pack_cycles);

int ws_gemm8_bf16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    int M,
    int N,
    int K,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats);

int ws_gemm8_matmul_softmax_tile_bf16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    int M,
    int N,
    int K,
    uint16_t scale_bf16,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats);

#ifdef __cplusplus
}
#endif

#endif
