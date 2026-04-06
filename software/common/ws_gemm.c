#include "ws_gemm.h"

#include <stdint.h>
#include <string.h>

static inline __attribute__((always_inline)) uint64_t load_u64_unaligned(
    const void *WS_GEMM_RESTRICT src) {
  uint64_t value;
  memcpy(&value, src, sizeof(value));
  return value;
}

static inline __attribute__((always_inline)) uint64_t load_u64_aligned(
    const void *WS_GEMM_RESTRICT src) {
  uint64_t value;
#if defined(__GNUC__) || defined(__clang__)
  src = __builtin_assume_aligned(src, sizeof(uint64_t));
#endif
  __builtin_memcpy(&value, src, sizeof(value));
  return value;
}

static inline uint64_t *a_tile_ptr(const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace, int m0) {
  return workspace->a_tiles + (m0 / WS_GEMM_ROWS) * workspace->max_k;
}

static inline uint64_t *b_tile_ptr(uint64_t *WS_GEMM_RESTRICT b_tiles, int max_k, int n0) {
  return b_tiles + (n0 / WS_GEMM_COLS) * max_k;
}

static inline const uint64_t *b_tile_ptr_const(const uint64_t *WS_GEMM_RESTRICT b_tiles, int max_k, int n0) {
  return b_tiles + (n0 / WS_GEMM_COLS) * max_k;
}

static inline __attribute__((always_inline)) void pack_a_tile_full(
    const uint16_t *WS_GEMM_RESTRICT row0,
    const uint16_t *WS_GEMM_RESTRICT row1,
    const uint16_t *WS_GEMM_RESTRICT row2,
    const uint16_t *WS_GEMM_RESTRICT row3,
    uint64_t *WS_GEMM_RESTRICT dst,
    int K) {
  int k = 0;
  const uintptr_t align_mask =
      (uintptr_t)row0 | (uintptr_t)row1 | (uintptr_t)row2 | (uintptr_t)row3;

  if ((align_mask & (sizeof(uint64_t) - 1)) == 0) {
    for (; k + 3 < K; k += 4) {
      const uint64_t r0 = load_u64_aligned(&row0[k]);
      const uint64_t r1 = load_u64_aligned(&row1[k]);
      const uint64_t r2 = load_u64_aligned(&row2[k]);
      const uint64_t r3 = load_u64_aligned(&row3[k]);

      dst[k + 0] = (r0 & 0x000000000000FFFFULL) |
                   ((r1 & 0x000000000000FFFFULL) << 16) |
                   ((r2 & 0x000000000000FFFFULL) << 32) |
                   ((r3 & 0x000000000000FFFFULL) << 48);
      dst[k + 1] = ((r0 >> 16) & 0x000000000000FFFFULL) |
                   (r1 & 0x00000000FFFF0000ULL) |
                   ((r2 & 0x00000000FFFF0000ULL) << 16) |
                   ((r3 & 0x00000000FFFF0000ULL) << 32);
      dst[k + 2] = ((r0 >> 32) & 0x000000000000FFFFULL) |
                   ((r1 >> 16) & 0x00000000FFFF0000ULL) |
                   (r2 & 0x0000FFFF00000000ULL) |
                   ((r3 & 0x0000FFFF00000000ULL) << 16);
      dst[k + 3] = ((r0 >> 48) & 0x000000000000FFFFULL) |
                   ((r1 >> 32) & 0x00000000FFFF0000ULL) |
                   ((r2 >> 16) & 0x0000FFFF00000000ULL) |
                   (r3 & 0xFFFF000000000000ULL);
    }
  } else {
    for (; k + 3 < K; k += 4) {
      const uint64_t r0 = load_u64_unaligned(&row0[k]);
      const uint64_t r1 = load_u64_unaligned(&row1[k]);
      const uint64_t r2 = load_u64_unaligned(&row2[k]);
      const uint64_t r3 = load_u64_unaligned(&row3[k]);

      dst[k + 0] = (r0 & 0x000000000000FFFFULL) |
                   ((r1 & 0x000000000000FFFFULL) << 16) |
                   ((r2 & 0x000000000000FFFFULL) << 32) |
                   ((r3 & 0x000000000000FFFFULL) << 48);
      dst[k + 1] = ((r0 >> 16) & 0x000000000000FFFFULL) |
                   (r1 & 0x00000000FFFF0000ULL) |
                   ((r2 & 0x00000000FFFF0000ULL) << 16) |
                   ((r3 & 0x00000000FFFF0000ULL) << 32);
      dst[k + 2] = ((r0 >> 32) & 0x000000000000FFFFULL) |
                   ((r1 >> 16) & 0x00000000FFFF0000ULL) |
                   (r2 & 0x0000FFFF00000000ULL) |
                   ((r3 & 0x0000FFFF00000000ULL) << 16);
      dst[k + 3] = ((r0 >> 48) & 0x000000000000FFFFULL) |
                   ((r1 >> 32) & 0x00000000FFFF0000ULL) |
                   ((r2 >> 16) & 0x0000FFFF00000000ULL) |
                   (r3 & 0xFFFF000000000000ULL);
    }
  }

  for (; k < K; k++) {
    dst[k] = ws_pack4_u16(row0[k], row1[k], row2[k], row3[k]);
  }
}

static inline __attribute__((always_inline)) void pack_a_tile_edge(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    int m0,
    uint64_t *WS_GEMM_RESTRICT dst,
    int M,
    int K) {
  for (int k = 0; k < K; k++) {
    const int a0_idx = (m0 + 0) * lda + k;
    const int a1_idx = (m0 + 1) * lda + k;
    const int a2_idx = (m0 + 2) * lda + k;
    const int a3_idx = (m0 + 3) * lda + k;
    const uint16_t a0 = (m0 + 0 < M) ? A[a0_idx] : 0;
    const uint16_t a1 = (m0 + 1 < M) ? A[a1_idx] : 0;
    const uint16_t a2 = (m0 + 2 < M) ? A[a2_idx] : 0;
    const uint16_t a3 = (m0 + 3 < M) ? A[a3_idx] : 0;
    dst[k] = ws_pack4_u16(a0, a1, a2, a3);
  }
}

static inline __attribute__((always_inline)) void pack_b_tile_full(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int n0,
    uint64_t *WS_GEMM_RESTRICT dst,
    int K) {
  if ((((uintptr_t)(B + n0)) & (sizeof(uint64_t) - 1)) == 0 && ((ldb & 3) == 0)) {
    for (int k = 0; k < K; k++) {
      dst[k] = load_u64_aligned(&B[k * ldb + n0]);
    }
  } else {
    for (int k = 0; k < K; k++) {
      dst[k] = load_u64_unaligned(&B[k * ldb + n0]);
    }
  }
}

static inline __attribute__((always_inline)) void pack_b_tile_edge(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int n0,
    uint64_t *WS_GEMM_RESTRICT dst,
    int N,
    int K) {
  for (int k = 0; k < K; k++) {
    const uint16_t *b = &B[k * ldb];
    const uint16_t b0 = (n0 + 0 < N) ? b[n0 + 0] : 0;
    const uint16_t b1 = (n0 + 1 < N) ? b[n0 + 1] : 0;
    const uint16_t b2 = (n0 + 2 < N) ? b[n0 + 2] : 0;
    const uint16_t b3 = (n0 + 3 < N) ? b[n0 + 3] : 0;
    dst[k] = ws_pack4_u16(b0, b1, b2, b3);
  }
}

static inline __attribute__((always_inline)) void store_c_tile_full(
    uint32_t *WS_GEMM_RESTRICT C,
    int ldc,
    int m0,
    int n0,
    const uint64_t *WS_GEMM_RESTRICT c_words) {
  uint32_t *c0 = &C[(m0 + 0) * ldc + n0];
  uint32_t *c1 = &C[(m0 + 1) * ldc + n0];
  uint32_t *c2 = &C[(m0 + 2) * ldc + n0];
  uint32_t *c3 = &C[(m0 + 3) * ldc + n0];

  c0[0] = (uint32_t)c_words[0];
  c0[1] = (uint32_t)c_words[1];
  c0[2] = (uint32_t)c_words[2];
  c0[3] = (uint32_t)c_words[3];
  c1[0] = (uint32_t)c_words[4];
  c1[1] = (uint32_t)c_words[5];
  c1[2] = (uint32_t)c_words[6];
  c1[3] = (uint32_t)c_words[7];
  c2[0] = (uint32_t)c_words[8];
  c2[1] = (uint32_t)c_words[9];
  c2[2] = (uint32_t)c_words[10];
  c2[3] = (uint32_t)c_words[11];
  c3[0] = (uint32_t)c_words[12];
  c3[1] = (uint32_t)c_words[13];
  c3[2] = (uint32_t)c_words[14];
  c3[3] = (uint32_t)c_words[15];
}

static inline __attribute__((always_inline)) void store_c_tile_edge(
    uint32_t *WS_GEMM_RESTRICT C,
    int ldc,
    int m0,
    int n0,
    const uint64_t *WS_GEMM_RESTRICT c_words,
    int M,
    int N) {
  for (int i = 0; i < WS_GEMM_ROWS; i++) {
    for (int j = 0; j < WS_GEMM_COLS; j++) {
      const int m = m0 + i;
      const int n = n0 + j;
      if (m < M && n < N) {
        C[m * ldc + n] = (uint32_t)c_words[i * WS_GEMM_COLS + j];
      }
    }
  }
}

static inline __attribute__((always_inline)) void store_c_tile_full_s64(
    int64_t *WS_GEMM_RESTRICT C,
    int ldc,
    int m0,
    int n0,
    const uint64_t *WS_GEMM_RESTRICT c_words) {
  int64_t *c0 = &C[(m0 + 0) * ldc + n0];
  int64_t *c1 = &C[(m0 + 1) * ldc + n0];
  int64_t *c2 = &C[(m0 + 2) * ldc + n0];
  int64_t *c3 = &C[(m0 + 3) * ldc + n0];

  c0[0] = (int64_t)c_words[0];
  c0[1] = (int64_t)c_words[1];
  c0[2] = (int64_t)c_words[2];
  c0[3] = (int64_t)c_words[3];
  c1[0] = (int64_t)c_words[4];
  c1[1] = (int64_t)c_words[5];
  c1[2] = (int64_t)c_words[6];
  c1[3] = (int64_t)c_words[7];
  c2[0] = (int64_t)c_words[8];
  c2[1] = (int64_t)c_words[9];
  c2[2] = (int64_t)c_words[10];
  c2[3] = (int64_t)c_words[11];
  c3[0] = (int64_t)c_words[12];
  c3[1] = (int64_t)c_words[13];
  c3[2] = (int64_t)c_words[14];
  c3[3] = (int64_t)c_words[15];
}

static inline __attribute__((always_inline)) void store_c_tile_edge_s64(
    int64_t *WS_GEMM_RESTRICT C,
    int ldc,
    int m0,
    int n0,
    const uint64_t *WS_GEMM_RESTRICT c_words,
    int M,
    int N) {
  for (int i = 0; i < WS_GEMM_ROWS; i++) {
    for (int j = 0; j < WS_GEMM_COLS; j++) {
      const int m = m0 + i;
      const int n = n0 + j;
      if (m < M && n < N) {
        C[m * ldc + n] = (int64_t)c_words[i * WS_GEMM_COLS + j];
      }
    }
  }
}

static inline __attribute__((always_inline)) uint16_t unpack_bf16_lane(
    uint64_t packed_word,
    int lane) {
  return (uint16_t)((packed_word >> (16 * lane)) & 0xFFFFu);
}

static inline __attribute__((always_inline)) void store_c_tile_full_bf16(
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    int m0,
    int n0,
    const uint64_t *WS_GEMM_RESTRICT c_words) {
  uint16_t *c0 = &C[(m0 + 0) * ldc + n0];
  uint16_t *c1 = &C[(m0 + 1) * ldc + n0];
  uint16_t *c2 = &C[(m0 + 2) * ldc + n0];
  uint16_t *c3 = &C[(m0 + 3) * ldc + n0];

  c0[0] = unpack_bf16_lane(c_words[0], 0);
  c0[1] = unpack_bf16_lane(c_words[0], 1);
  c0[2] = unpack_bf16_lane(c_words[0], 2);
  c0[3] = unpack_bf16_lane(c_words[0], 3);
  c1[0] = unpack_bf16_lane(c_words[1], 0);
  c1[1] = unpack_bf16_lane(c_words[1], 1);
  c1[2] = unpack_bf16_lane(c_words[1], 2);
  c1[3] = unpack_bf16_lane(c_words[1], 3);
  c2[0] = unpack_bf16_lane(c_words[2], 0);
  c2[1] = unpack_bf16_lane(c_words[2], 1);
  c2[2] = unpack_bf16_lane(c_words[2], 2);
  c2[3] = unpack_bf16_lane(c_words[2], 3);
  c3[0] = unpack_bf16_lane(c_words[3], 0);
  c3[1] = unpack_bf16_lane(c_words[3], 1);
  c3[2] = unpack_bf16_lane(c_words[3], 2);
  c3[3] = unpack_bf16_lane(c_words[3], 3);
}

static inline __attribute__((always_inline)) void store_c_tile_edge_bf16(
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    int m0,
    int n0,
    const uint64_t *WS_GEMM_RESTRICT c_words,
    int M,
    int N) {
  for (int i = 0; i < WS_GEMM_ROWS; i++) {
    const uint64_t packed_row = c_words[i];
    for (int j = 0; j < WS_GEMM_COLS; j++) {
      const int m = m0 + i;
      const int n = n0 + j;
      if (m < M && n < N) {
        C[m * ldc + n] = unpack_bf16_lane(packed_row, j);
      }
    }
  }
}

static int validate_pack_b_dims(int ldb, int N, int K) {
  if (N < 0 || K < 0) {
    return 0;
  }
  if (ldb < N) {
    return 0;
  }
  return 1;
}

static int validate_run_dims(int lda, int ldc, int M, int N, int K) {
  if (M < 0 || N < 0 || K < 0) {
    return 0;
  }
  if (lda < K || ldc < N) {
    return 0;
  }
  return 1;
}

static int validate_full_dims(int lda, int ldb, int ldc, int M, int N, int K) {
  return validate_run_dims(lda, ldc, M, N, K) && validate_pack_b_dims(ldb, N, K);
}

static int validate_workspace_full(
    const ws_gemm_workspace_t *workspace,
    int M,
    int N,
    int K) {
  if (workspace == 0 || workspace->a_tiles == 0 || workspace->b_tiles == 0 ||
      workspace->c_words == 0) {
    return 0;
  }
  if (M > workspace->max_m || N > workspace->max_n || K > workspace->max_k) {
    return 0;
  }
  return 1;
}

static int validate_workspace_run_only(
    const ws_gemm_workspace_t *workspace,
    int M,
    int K) {
  if (workspace == 0 || workspace->a_tiles == 0 || workspace->c_words == 0) {
    return 0;
  }
  if (M > workspace->max_m || K > workspace->max_k) {
    return 0;
  }
  return 1;
}

static int validate_packed_b(
    const ws_gemm_packed_b_t *packed_b,
    int N,
    int K) {
  if (packed_b == 0 || packed_b->tiles == 0) {
    return 0;
  }
  if (packed_b->N != N || packed_b->K != K) {
    return 0;
  }
  if (packed_b->N > packed_b->max_n || packed_b->K > packed_b->max_k) {
    return 0;
  }
  return 1;
}

void ws_gemm_clear_stats(ws_gemm_stats_t *stats) {
  if (stats != 0) {
    memset(stats, 0, sizeof(*stats));
  }
}

int ws_gemm_pack_b_u16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    uint64_t *pack_cycles) {
  if (!validate_pack_b_dims(ldb, N, K)) {
    return WS_GEMM_ERR_BAD_DIMS;
  }
  if (b_tiles == 0 || (K > 0 && N > 0 && B == 0)) {
    return WS_GEMM_ERR_WORKSPACE;
  }
  if (N > max_n || K > max_k) {
    return WS_GEMM_ERR_WORKSPACE;
  }

  uint64_t start = 0;
  if (pack_cycles != 0) {
    *pack_cycles = 0;
    start = ws_read_cycles();
  }

  for (int n0 = 0; n0 < N; n0 += WS_GEMM_COLS) {
    uint64_t *b_stream = b_tile_ptr(b_tiles, max_k, n0);
    if (n0 + WS_GEMM_COLS <= N) {
      pack_b_tile_full(B, ldb, n0, b_stream, K);
    } else {
      pack_b_tile_edge(B, ldb, n0, b_stream, N, K);
    }
  }

  if (pack_cycles != 0) {
    *pack_cycles = ws_read_cycles() - start;
  }

  return WS_GEMM_OK;
}

int ws_gemm_prepare_packed_b_u16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    uint64_t *pack_cycles) {
  if (packed_b == 0) {
    return WS_GEMM_ERR_WORKSPACE;
  }

  const int rc = ws_gemm_pack_b_u16(B, ldb, N, K, b_tiles, max_n, max_k, pack_cycles);
  if (rc != WS_GEMM_OK) {
    return rc;
  }

  *packed_b = ws_gemm_make_packed_b(b_tiles, max_n, max_k, N, K);
  return WS_GEMM_OK;
}

int ws_gemm_prepare_packed_b_bf16(
    const uint16_t *WS_GEMM_RESTRICT B,
    int ldb,
    int N,
    int K,
    uint64_t *WS_GEMM_RESTRICT b_tiles,
    int max_n,
    int max_k,
    ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    uint64_t *pack_cycles) {
  return ws_gemm_prepare_packed_b_u16(
      B, ldb, N, K, b_tiles, max_n, max_k, packed_b, pack_cycles);
}

static void ws_gemm_pack_a_matrix(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    int M,
    int K,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int measure = (stats != 0);

  uint64_t pack_a_start = measure ? ws_read_cycles() : 0;
  for (int m0 = 0; m0 < M; m0 += WS_GEMM_ROWS) {
    uint64_t *a_stream = a_tile_ptr(workspace, m0);
    if (m0 + WS_GEMM_ROWS <= M) {
      const uint16_t *row0 = &A[(m0 + 0) * lda];
      const uint16_t *row1 = &A[(m0 + 1) * lda];
      const uint16_t *row2 = &A[(m0 + 2) * lda];
      const uint16_t *row3 = &A[(m0 + 3) * lda];
      pack_a_tile_full(row0, row1, row2, row3, a_stream, K);
    } else {
      pack_a_tile_edge(A, lda, m0, a_stream, M, K);
    }
  }
  if (measure) {
    stats->stage.pack_a_cycles += ws_read_cycles() - pack_a_start;
  }
}

static int ws_gemm_run_preloaded_tile(
    int M,
    int N,
    int n0,
    uint32_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int measure = (stats != 0);

  for (int m0 = 0; m0 < M; m0 += WS_GEMM_ROWS) {
    const uint64_t *a_stream = a_tile_ptr(workspace, m0);

    uint64_t run_start = measure ? ws_read_cycles() : 0;
    if (ws_sa_run_preloaded(a_stream, workspace->c_words) != 0) {
      return WS_GEMM_ERR_RUN;
    }
    if (measure) {
      stats->stage.run_cycles += ws_read_cycles() - run_start;
    }

    uint64_t copy_start = measure ? ws_read_cycles() : 0;
    if (m0 + WS_GEMM_ROWS <= M && n0 + WS_GEMM_COLS <= N) {
      store_c_tile_full(C, ldc, m0, n0, workspace->c_words);
    } else {
      store_c_tile_edge(C, ldc, m0, n0, workspace->c_words, M, N);
    }
    if (measure) {
      stats->stage.copy_out_cycles += ws_read_cycles() - copy_start;
    }
  }

  return WS_GEMM_OK;
}

static int ws_gemm_run_preloaded_tile_s64(
    int M,
    int N,
    int n0,
    int64_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int measure = (stats != 0);

  for (int m0 = 0; m0 < M; m0 += WS_GEMM_ROWS) {
    const uint64_t *a_stream = a_tile_ptr(workspace, m0);

    uint64_t run_start = measure ? ws_read_cycles() : 0;
    if (ws_sa_run_preloaded(a_stream, workspace->c_words) != 0) {
      return WS_GEMM_ERR_RUN;
    }
    if (measure) {
      stats->stage.run_cycles += ws_read_cycles() - run_start;
    }

    uint64_t copy_start = measure ? ws_read_cycles() : 0;
    if (m0 + WS_GEMM_ROWS <= M && n0 + WS_GEMM_COLS <= N) {
      store_c_tile_full_s64(C, ldc, m0, n0, workspace->c_words);
    } else {
      store_c_tile_edge_s64(C, ldc, m0, n0, workspace->c_words, M, N);
    }
    if (measure) {
      stats->stage.copy_out_cycles += ws_read_cycles() - copy_start;
    }
  }

  return WS_GEMM_OK;
}

static int ws_gemm_run_preloaded_tile_bf16(
    int M,
    int N,
    int n0,
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int measure = (stats != 0);

  for (int m0 = 0; m0 < M; m0 += WS_GEMM_ROWS) {
    const uint64_t *a_stream = a_tile_ptr(workspace, m0);

    uint64_t run_start = measure ? ws_read_cycles() : 0;
    if (ws_sa_run_preloaded(a_stream, workspace->c_words) != 0) {
      return WS_GEMM_ERR_RUN;
    }
    if (measure) {
      stats->stage.run_cycles += ws_read_cycles() - run_start;
    }

    uint64_t copy_start = measure ? ws_read_cycles() : 0;
    if (m0 + WS_GEMM_ROWS <= M && n0 + WS_GEMM_COLS <= N) {
      store_c_tile_full_bf16(C, ldc, m0, n0, workspace->c_words);
    } else {
      store_c_tile_edge_bf16(C, ldc, m0, n0, workspace->c_words, M, N);
    }
    if (measure) {
      stats->stage.copy_out_cycles += ws_read_cycles() - copy_start;
    }
  }

  return WS_GEMM_OK;
}

static int ws_gemm_run_packed_core(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const uint64_t *WS_GEMM_RESTRICT b_tiles,
    int M,
    int N,
    int K,
    uint32_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int measure = (stats != 0);

  ws_gemm_pack_a_matrix(A, lda, M, K, workspace, stats);

  for (int n0 = 0; n0 < N; n0 += WS_GEMM_COLS) {
    const uint64_t *b_stream = b_tile_ptr_const(b_tiles, workspace->max_k, n0);

    uint64_t preload_start = measure ? ws_read_cycles() : 0;
    if (ws_sa_preload_weights(b_stream, (uint64_t)K) != 0) {
      return WS_GEMM_ERR_PRELOAD;
    }
    if (measure) {
      stats->stage.preload_cycles += ws_read_cycles() - preload_start;
    }

    const int rc = ws_gemm_run_preloaded_tile(M, N, n0, C, ldc, workspace, stats);
    if (rc != WS_GEMM_OK) {
      return rc;
    }
  }

  if (measure) {
    ws_read_perf_counters(stats->perf);
  }

  return WS_GEMM_OK;
}

static int ws_gemm_run_packed_core_s64(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const uint64_t *WS_GEMM_RESTRICT b_tiles,
    int M,
    int N,
    int K,
    int64_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int measure = (stats != 0);

  ws_gemm_pack_a_matrix(A, lda, M, K, workspace, stats);

  for (int n0 = 0; n0 < N; n0 += WS_GEMM_COLS) {
    const uint64_t *b_stream = b_tile_ptr_const(b_tiles, workspace->max_k, n0);

    uint64_t preload_start = measure ? ws_read_cycles() : 0;
    if (ws_sa_preload_weights(b_stream, (uint64_t)K) != 0) {
      return WS_GEMM_ERR_PRELOAD;
    }
    if (measure) {
      stats->stage.preload_cycles += ws_read_cycles() - preload_start;
    }

    const int rc = ws_gemm_run_preloaded_tile_s64(M, N, n0, C, ldc, workspace, stats);
    if (rc != WS_GEMM_OK) {
      return rc;
    }
  }

  if (measure) {
    ws_read_perf_counters(stats->perf);
  }

  return WS_GEMM_OK;
}

static int ws_gemm_run_packed_core_bf16(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const uint64_t *WS_GEMM_RESTRICT b_tiles,
    int M,
    int N,
    int K,
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int measure = (stats != 0);

  ws_gemm_pack_a_matrix(A, lda, M, K, workspace, stats);

  for (int n0 = 0; n0 < N; n0 += WS_GEMM_COLS) {
    const uint64_t *b_stream = b_tile_ptr_const(b_tiles, workspace->max_k, n0);

    uint64_t preload_start = measure ? ws_read_cycles() : 0;
    if (ws_sa_preload_weights(b_stream, (uint64_t)K) != 0) {
      return WS_GEMM_ERR_PRELOAD;
    }
    if (measure) {
      stats->stage.preload_cycles += ws_read_cycles() - preload_start;
    }

    const int rc = ws_gemm_run_preloaded_tile_bf16(M, N, n0, C, ldc, workspace, stats);
    if (rc != WS_GEMM_OK) {
      return rc;
    }
  }

  if (measure) {
    ws_read_perf_counters(stats->perf);
  }

  return WS_GEMM_OK;
}

int ws_gemm_u16_prepacked_b(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    int M,
    uint32_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int N = packed_b != 0 ? packed_b->N : 0;
  const int K = packed_b != 0 ? packed_b->K : 0;

  if (!validate_run_dims(lda, ldc, M, N, K)) {
    return WS_GEMM_ERR_BAD_DIMS;
  }
  if (!validate_workspace_run_only(workspace, M, K) || C == 0 ||
      (M > 0 && K > 0 && A == 0) || !validate_packed_b(packed_b, N, K)) {
    return WS_GEMM_ERR_WORKSPACE;
  }

  ws_gemm_clear_stats(stats);

  const int measure = (stats != 0);
  if (measure) {
    ws_sa_clear_counters();
  }
  const uint64_t total_start = measure ? ws_read_cycles() : 0;

  const int rc = ws_gemm_run_packed_core(
      A, lda, packed_b->tiles, M, N, K, C, ldc, workspace, stats);

  if (measure) {
    stats->hw_e2e_cycles = ws_read_cycles() - total_start;
  }

  return rc;
}

int ws_gemm_bf16_prepacked_b(
    const uint16_t *WS_GEMM_RESTRICT A,
    int lda,
    const ws_gemm_packed_b_t *WS_GEMM_RESTRICT packed_b,
    int M,
    uint16_t *WS_GEMM_RESTRICT C,
    int ldc,
    const ws_gemm_workspace_t *WS_GEMM_RESTRICT workspace,
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  const int N = packed_b != 0 ? packed_b->N : 0;
  const int K = packed_b != 0 ? packed_b->K : 0;

  if (!validate_run_dims(lda, ldc, M, N, K)) {
    return WS_GEMM_ERR_BAD_DIMS;
  }
  if (!validate_workspace_run_only(workspace, M, K) || C == 0 ||
      (M > 0 && K > 0 && A == 0) || !validate_packed_b(packed_b, N, K)) {
    return WS_GEMM_ERR_WORKSPACE;
  }

  ws_gemm_clear_stats(stats);

  const int measure = (stats != 0);
  if (measure) {
    ws_sa_clear_counters();
  }
  const uint64_t total_start = measure ? ws_read_cycles() : 0;

  const int rc = ws_gemm_run_packed_core_bf16(
      A, lda, packed_b->tiles, M, N, K, C, ldc, workspace, stats);

  if (measure) {
    stats->hw_e2e_cycles = ws_read_cycles() - total_start;
  }

  return rc;
}

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
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  if (!validate_full_dims(lda, ldb, ldc, M, N, K)) {
    return WS_GEMM_ERR_BAD_DIMS;
  }
  if (!validate_workspace_full(workspace, M, N, K) || C == 0 ||
      (M > 0 && K > 0 && A == 0) || (K > 0 && N > 0 && B == 0)) {
    return WS_GEMM_ERR_WORKSPACE;
  }

  ws_gemm_clear_stats(stats);

  const int measure = (stats != 0);
  if (measure) {
    ws_sa_clear_counters();
  }
  const uint64_t total_start = measure ? ws_read_cycles() : 0;

  const int pack_rc = ws_gemm_pack_b_u16(
      B, ldb, N, K, workspace->b_tiles, workspace->max_n, workspace->max_k,
      measure ? &stats->stage.pack_b_cycles : 0);
  if (pack_rc != WS_GEMM_OK) {
    if (measure) {
      stats->hw_e2e_cycles = ws_read_cycles() - total_start;
    }
    return pack_rc;
  }

  const ws_gemm_packed_b_t packed_b =
      ws_gemm_make_packed_b(workspace->b_tiles, workspace->max_n, workspace->max_k, N, K);
  const int rc = ws_gemm_run_packed_core(
      A, lda, packed_b.tiles, M, N, K, C, ldc, workspace, stats);

  if (measure) {
    stats->hw_e2e_cycles = ws_read_cycles() - total_start;
  }

  return rc;
}

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
    ws_gemm_stats_t *WS_GEMM_RESTRICT stats) {
  if (!validate_full_dims(lda, ldb, ldc, M, N, K)) {
    return WS_GEMM_ERR_BAD_DIMS;
  }
  if (!validate_workspace_full(workspace, M, N, K) || C == 0 ||
      (M > 0 && K > 0 && A == 0) || (K > 0 && N > 0 && B == 0)) {
    return WS_GEMM_ERR_WORKSPACE;
  }

  ws_gemm_clear_stats(stats);

  const int measure = (stats != 0);
  if (measure) {
    ws_sa_clear_counters();
  }
  const uint64_t total_start = measure ? ws_read_cycles() : 0;

  const int pack_rc = ws_gemm_pack_b_u16(
      B, ldb, N, K, workspace->b_tiles, workspace->max_n, workspace->max_k,
      measure ? &stats->stage.pack_b_cycles : 0);
  if (pack_rc != WS_GEMM_OK) {
    if (measure) {
      stats->hw_e2e_cycles = ws_read_cycles() - total_start;
    }
    return pack_rc;
  }

  const ws_gemm_packed_b_t packed_b =
      ws_gemm_make_packed_b(workspace->b_tiles, workspace->max_n, workspace->max_k, N, K);
  const int rc = ws_gemm_run_packed_core_bf16(
      A, lda, packed_b.tiles, M, N, K, C, ldc, workspace, stats);

  if (measure) {
    stats->hw_e2e_cycles = ws_read_cycles() - total_start;
  }

  return rc;
}
