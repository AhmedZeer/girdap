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
#define WS_GEMM_C_TILE_WORDS (WS_GEMM_ROWS * WS_GEMM_COLS)
#define WS_GEMM_BF16_FIXED_FRAC_BITS 8
#define WS_GEMM_BF16_ACC_FRAC_BITS (2 * WS_GEMM_BF16_FIXED_FRAC_BITS)
#define WS_GEMM_A_TILE_WORDS(max_m, max_k) \
  (((((max_m) + WS_GEMM_ROWS - 1) / WS_GEMM_ROWS)) * (max_k))
#define WS_GEMM_B_TILE_WORDS(max_n, max_k) \
  (((((max_n) + WS_GEMM_COLS - 1) / WS_GEMM_COLS)) * (max_k))

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
} ws_gemm_workspace_t;

static inline ws_gemm_workspace_t ws_gemm_make_workspace(
    uint64_t *a_tiles,
    uint64_t *b_tiles,
    uint64_t *c_words,
    int max_m,
    int max_n,
    int max_k) {
  ws_gemm_workspace_t workspace = {
      .a_tiles = a_tiles,
      .b_tiles = b_tiles,
      .c_words = c_words,
      .max_m = max_m,
      .max_n = max_n,
      .max_k = max_k,
  };
  return workspace;
}

typedef struct {
  const uint64_t *tiles;
  int max_n;
  int max_k;
  int N;
  int K;
} ws_gemm_packed_b_t;

static inline ws_gemm_packed_b_t ws_gemm_make_packed_b(
    const uint64_t *tiles,
    int max_n,
    int max_k,
    int N,
    int K) {
  ws_gemm_packed_b_t packed_b = {
      .tiles = tiles,
      .max_n = max_n,
      .max_k = max_k,
      .N = N,
      .K = K,
  };
  return packed_b;
}

void ws_gemm_clear_stats(ws_gemm_stats_t *stats);

int ws_gemm_pack_b_u16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
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

#ifdef __cplusplus
}
#endif

#endif
