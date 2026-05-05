#include "fpga_safe_attention.h"

#include "bfloat16_utils.h"
#include "systolic_ws.h"

#include <string.h>

static int min_int(int a, int b) {
  return (a < b) ? a : b;
}

static uint64_t *attn_q_tile_ptr(
    uint64_t *q_tiles,
    int max_d_k,
    int m0) {
  return q_tiles + (m0 / FPGA_SAFE_ATTN_TILE_ROWS) * max_d_k * WS_GEMM8_BEAT_WORDS;
}

static uint64_t *attn_b_tile_ptr(
    uint64_t *tiles,
    int max_k,
    int n0) {
  return tiles + (n0 / FPGA_SAFE_ATTN_TILE_COLS) * max_k * WS_GEMM8_BEAT_WORDS;
}

static uint16_t unpack_bf16_lane(uint64_t word, int lane) {
  return (uint16_t)((word >> (16 * lane)) & 0xFFFFu);
}

static void copy_output_tile(
    const uint64_t *out_words,
    int active_rows,
    int active_cols,
    uint16_t *output,
    int ldout,
    int m0,
    int n0) {
  for (int r = 0; r < FPGA_SAFE_ATTN_TILE_ROWS; r++) {
    for (int c = 0; c < FPGA_SAFE_ATTN_TILE_COLS; c++) {
      if (r < active_rows && c < active_cols) {
        const int phys_idx = r * FPGA_SAFE_ATTN_TILE_COLS + c;
        const int word_idx = phys_idx / 4;
        const int lane_idx = phys_idx % 4;
        output[(m0 + r) * ldout + (n0 + c)] =
            unpack_bf16_lane(out_words[word_idx], lane_idx);
      }
    }
  }
}

static int validate_dims(
    int ldq,
    int ldk_t,
    int ldv,
    int ldout,
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols) {
  if (q_rows < 0 || kv_rows < 0 || d_k < 0 || value_cols < 0) {
    return 0;
  }
  if (ldq < d_k || ldk_t < kv_rows || ldv < value_cols || ldout < value_cols) {
    return 0;
  }
  if (q_rows > 0 && (kv_rows == 0 || d_k == 0 || value_cols == 0)) {
    return 0;
  }
  return 1;
}

static int validate_workspace(
    const fpga_safe_attention_workspace_t *workspace,
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols) {
  if (workspace == 0 || workspace->q_tiles == 0 || workspace->k_tiles == 0 ||
      workspace->v_tiles == 0 || workspace->out_words == 0) {
    return 0;
  }
  if (q_rows > workspace->max_q_rows || kv_rows > workspace->max_kv_rows ||
      d_k > workspace->max_d_k || value_cols > workspace->max_value_cols) {
    return 0;
  }
  return 1;
}

static int run_attention_packed(
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols,
    uint16_t scale_bf16,
    uint16_t *output,
    int ldout,
    const fpga_safe_attention_workspace_t *workspace,
    fpga_safe_attention_stats_t *stats) {
  const int measure = (stats != 0);

  for (int m0 = 0; m0 < q_rows; m0 += FPGA_SAFE_ATTN_TILE_ROWS) {
    const int tile_q_rows = min_int(FPGA_SAFE_ATTN_TILE_ROWS, q_rows - m0);
    const uint64_t *q_tile = attn_q_tile_ptr(workspace->q_tiles, workspace->max_d_k, m0);

    (void)ws_attn_set_qk_addrs(q_tile, workspace->k_tiles);
    (void)ws_attn_set_dims(
        tile_q_rows,
        kv_rows,
        d_k,
        min_int(FPGA_SAFE_ATTN_TILE_COLS, value_cols));
    (void)ws_attn_set_scale_bf16(scale_bf16);

    const uint64_t score_start = measure ? ws_read_cycles() : 0;
    const uint64_t score_rc = ws_attn_precompute_scores();
    if (measure) {
      const uint64_t score_cycles = ws_read_cycles() - score_start;
      stats->score_cycles += score_cycles;
      stats->hw_e2e_cycles += score_cycles;
      stats->raw_hw_rc = score_rc;
    }
    if (score_rc != 0) {
      return FPGA_SAFE_ATTN_ERR_RUN;
    }

    for (int n0 = 0; n0 < value_cols; n0 += FPGA_SAFE_ATTN_TILE_COLS) {
      const int tile_value_cols = min_int(FPGA_SAFE_ATTN_TILE_COLS, value_cols - n0);
      const uint64_t *v_tile = attn_b_tile_ptr(workspace->v_tiles, workspace->max_kv_rows, n0);

      (void)ws_attn_set_vout_addrs(v_tile, workspace->out_words);
      (void)ws_attn_set_dims(tile_q_rows, kv_rows, d_k, tile_value_cols);

      const uint64_t value_start = measure ? ws_read_cycles() : 0;
      const uint64_t value_rc = ws_attn_apply_cached();
      if (measure) {
        const uint64_t value_cycles = ws_read_cycles() - value_start;
        stats->value_cycles += value_cycles;
        stats->hw_e2e_cycles += value_cycles;
        stats->raw_hw_rc = value_rc;
      }
      if (value_rc != 0) {
        return FPGA_SAFE_ATTN_ERR_RUN;
      }

      const uint64_t copy_start = measure ? ws_read_cycles() : 0;
      copy_output_tile(
          workspace->out_words,
          tile_q_rows,
          tile_value_cols,
          output,
          ldout,
          m0,
          n0);
      if (measure) {
        stats->copy_out_cycles += ws_read_cycles() - copy_start;
      }
    }
  }

  return FPGA_SAFE_ATTN_OK;
}

static int hw_pack_tiles(
    const uint16_t *src,
    int rows,
    int cols,
    int ld,
    uint64_t *dst,
    int out_stride,
    int mode,
    uint64_t *pack_cycles) {
  uint64_t start = 0;
  if (pack_cycles != 0) {
    *pack_cycles = 0;
    start = ws_read_cycles();
  }

  uint64_t rc = ws_attn_pack_set_addrs(src, dst);
  if (rc != 0) {
    return FPGA_SAFE_ATTN_ERR_PACK;
  }
  rc = ws_attn_pack_set_dims(rows, cols, ld, out_stride, mode);
  if (rc != 0) {
    return FPGA_SAFE_ATTN_ERR_PACK;
  }
  rc = ws_attn_pack_run();
  if (pack_cycles != 0) {
    *pack_cycles = ws_read_cycles() - start;
  }
  return rc == 0 ? FPGA_SAFE_ATTN_OK : FPGA_SAFE_ATTN_ERR_PACK;
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
    fpga_safe_attention_stats_t *stats) {
  if (stats != 0) {
    memset(stats, 0, sizeof(*stats));
  }
  if (!validate_dims(ldq, ldk_t, ldv, ldout, q_rows, kv_rows, d_k, value_cols)) {
    return FPGA_SAFE_ATTN_ERR_BAD_DIMS;
  }
  if (!validate_workspace(workspace, q_rows, kv_rows, d_k, value_cols) ||
      (q_rows > 0 && d_k > 0 && Q == 0) ||
      (kv_rows > 0 && d_k > 0 && K_t == 0) ||
      (kv_rows > 0 && value_cols > 0 && V == 0) ||
      (q_rows > 0 && value_cols > 0 && output == 0)) {
    return FPGA_SAFE_ATTN_ERR_WORKSPACE;
  }
  if (q_rows == 0 || value_cols == 0) {
    return FPGA_SAFE_ATTN_OK;
  }

  const int measure = (stats != 0);

  int rc = ws_gemm8_pack_a_bf16(
      Q,
      ldq,
      q_rows,
      d_k,
      workspace->q_tiles,
      workspace->max_q_rows,
      workspace->max_d_k,
      measure ? &stats->q_pack_cycles : 0);
  if (rc != WS_GEMM_OK) {
    return FPGA_SAFE_ATTN_ERR_PACK;
  }

  rc = ws_gemm8_pack_b_bf16(
      K_t,
      ldk_t,
      kv_rows,
      d_k,
      workspace->k_tiles,
      workspace->max_kv_rows,
      workspace->max_d_k,
      measure ? &stats->k_pack_cycles : 0);
  if (rc != WS_GEMM_OK) {
    return FPGA_SAFE_ATTN_ERR_PACK;
  }

  rc = ws_gemm8_pack_b_bf16(
      V,
      ldv,
      value_cols,
      kv_rows,
      workspace->v_tiles,
      workspace->max_value_cols,
      workspace->max_kv_rows,
      measure ? &stats->v_pack_cycles : 0);
  if (rc != WS_GEMM_OK) {
    return FPGA_SAFE_ATTN_ERR_PACK;
  }

  return run_attention_packed(
      q_rows,
      kv_rows,
      d_k,
      value_cols,
      scale_bf16,
      output,
      ldout,
      workspace,
      stats);
}

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
    fpga_safe_attention_stats_t *stats) {
  if (stats != 0) {
    memset(stats, 0, sizeof(*stats));
  }
  if (q_rows < 0 || kv_rows < 0 || d_k < 0 || value_cols < 0 ||
      ldq < d_k || ldk < d_k || ldv < value_cols || ldout < value_cols ||
      (q_rows > 0 && (kv_rows == 0 || d_k == 0 || value_cols == 0))) {
    return FPGA_SAFE_ATTN_ERR_BAD_DIMS;
  }
  if (!validate_workspace(workspace, q_rows, kv_rows, d_k, value_cols) ||
      (q_rows > 0 && d_k > 0 && Q == 0) ||
      (kv_rows > 0 && d_k > 0 && K == 0) ||
      (kv_rows > 0 && value_cols > 0 && V == 0) ||
      (q_rows > 0 && value_cols > 0 && output == 0)) {
    return FPGA_SAFE_ATTN_ERR_WORKSPACE;
  }
  if (q_rows == 0 || value_cols == 0) {
    return FPGA_SAFE_ATTN_OK;
  }

  const int measure = (stats != 0);
  int rc = hw_pack_tiles(
      Q,
      q_rows,
      d_k,
      ldq,
      workspace->q_tiles,
      workspace->max_d_k,
      SA_ATTN_PACK_MODE_ROW_MAJOR_TILES,
      measure ? &stats->q_pack_cycles : 0);
  if (rc != FPGA_SAFE_ATTN_OK) {
    return rc;
  }

  rc = hw_pack_tiles(
      K,
      kv_rows,
      d_k,
      ldk,
      workspace->k_tiles,
      workspace->max_d_k,
      SA_ATTN_PACK_MODE_ROW_MAJOR_TILES,
      measure ? &stats->k_pack_cycles : 0);
  if (rc != FPGA_SAFE_ATTN_OK) {
    return rc;
  }

  rc = hw_pack_tiles(
      V,
      value_cols,
      kv_rows,
      ldv,
      workspace->v_tiles,
      workspace->max_kv_rows,
      SA_ATTN_PACK_MODE_COLUMN_TILES,
      measure ? &stats->v_pack_cycles : 0);
  if (rc != FPGA_SAFE_ATTN_OK) {
    return rc;
  }

  return run_attention_packed(
      q_rows,
      kv_rows,
      d_k,
      value_cols,
      scale_bf16,
      output,
      ldout,
      workspace,
      stats);
}

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
    fpga_safe_attention_stats_t *stats) {
  if (stats != 0) {
    memset(stats, 0, sizeof(*stats));
  }
  if (q_rows < 0 || kv_rows < 0 || d_k < 0 || value_cols < 0 ||
      ldq < d_k || ldk < d_k || ldv < value_cols || ldout < value_cols ||
      ldintermediate < kv_rows ||
      (q_rows > 0 && (kv_rows == 0 || d_k == 0 || value_cols == 0))) {
    return FPGA_SAFE_ATTN_ERR_BAD_DIMS;
  }
  if (!validate_workspace(workspace, q_rows, kv_rows, d_k, value_cols) ||
      (q_rows > 0 && d_k > 0 && Q == 0) ||
      (kv_rows > 0 && d_k > 0 && K == 0) ||
      (kv_rows > 0 && value_cols > 0 && V == 0) ||
      (q_rows > 0 && value_cols > 0 && output == 0) ||
      (q_rows > 0 && kv_rows > 0 && (score_fixed == 0 || prob_fixed == 0))) {
    return FPGA_SAFE_ATTN_ERR_WORKSPACE;
  }
  if (q_rows == 0 || value_cols == 0) {
    return FPGA_SAFE_ATTN_OK;
  }

  const int measure = (stats != 0);
  int rc = hw_pack_tiles(
      Q,
      q_rows,
      d_k,
      ldq,
      workspace->q_tiles,
      workspace->max_d_k,
      SA_ATTN_PACK_MODE_ROW_MAJOR_TILES,
      measure ? &stats->q_pack_cycles : 0);
  if (rc != FPGA_SAFE_ATTN_OK) {
    return rc;
  }

  rc = hw_pack_tiles(
      K,
      kv_rows,
      d_k,
      ldk,
      workspace->k_tiles,
      workspace->max_d_k,
      SA_ATTN_PACK_MODE_ROW_MAJOR_TILES,
      measure ? &stats->k_pack_cycles : 0);
  if (rc != FPGA_SAFE_ATTN_OK) {
    return rc;
  }

  rc = hw_pack_tiles(
      V,
      value_cols,
      kv_rows,
      ldv,
      workspace->v_tiles,
      workspace->max_kv_rows,
      SA_ATTN_PACK_MODE_COLUMN_TILES,
      measure ? &stats->v_pack_cycles : 0);
  if (rc != FPGA_SAFE_ATTN_OK) {
    return rc;
  }

  for (int m0 = 0; m0 < q_rows; m0 += FPGA_SAFE_ATTN_TILE_ROWS) {
    const int tile_q_rows = min_int(FPGA_SAFE_ATTN_TILE_ROWS, q_rows - m0);
    const uint64_t *q_tile = attn_q_tile_ptr(workspace->q_tiles, workspace->max_d_k, m0);

    (void)ws_attn_set_qk_addrs(q_tile, workspace->k_tiles);
    (void)ws_attn_set_dims(
        tile_q_rows,
        kv_rows,
        d_k,
        min_int(FPGA_SAFE_ATTN_TILE_COLS, value_cols));
    (void)ws_attn_set_scale_bf16(scale_bf16);

    const uint64_t score_start = measure ? ws_read_cycles() : 0;
    const uint64_t score_rc = ws_attn_precompute_scores();
    if (measure) {
      const uint64_t score_cycles = ws_read_cycles() - score_start;
      stats->score_cycles += score_cycles;
      stats->hw_e2e_cycles += score_cycles;
      stats->raw_hw_rc = score_rc;
    }
    if (score_rc != 0) {
      return FPGA_SAFE_ATTN_ERR_RUN;
    }

    const uint64_t debug_start = measure ? ws_read_cycles() : 0;
    for (int r = 0; r < tile_q_rows; r++) {
      for (int c = 0; c < kv_rows; c++) {
        score_fixed[(m0 + r) * ldintermediate + c] = ws_attn_debug_read_score(r, c);
        prob_fixed[(m0 + r) * ldintermediate + c] = ws_attn_debug_read_prob(r, c);
      }
    }
    if (measure) {
      const uint64_t debug_cycles = ws_read_cycles() - debug_start;
      stats->debug_dump_cycles += debug_cycles;
      stats->hw_e2e_cycles += debug_cycles;
    }

    for (int n0 = 0; n0 < value_cols; n0 += FPGA_SAFE_ATTN_TILE_COLS) {
      const int tile_value_cols = min_int(FPGA_SAFE_ATTN_TILE_COLS, value_cols - n0);
      const uint64_t *v_tile = attn_b_tile_ptr(workspace->v_tiles, workspace->max_kv_rows, n0);

      (void)ws_attn_set_vout_addrs(v_tile, workspace->out_words);
      (void)ws_attn_set_dims(tile_q_rows, kv_rows, d_k, tile_value_cols);

      const uint64_t value_start = measure ? ws_read_cycles() : 0;
      const uint64_t value_rc = ws_attn_apply_cached();
      if (measure) {
        const uint64_t value_cycles = ws_read_cycles() - value_start;
        stats->value_cycles += value_cycles;
        stats->hw_e2e_cycles += value_cycles;
        stats->raw_hw_rc = value_rc;
      }
      if (value_rc != 0) {
        return FPGA_SAFE_ATTN_ERR_RUN;
      }

      const uint64_t copy_start = measure ? ws_read_cycles() : 0;
      copy_output_tile(
          workspace->out_words,
          tile_q_rows,
          tile_value_cols,
          output,
          ldout,
          m0,
          n0);
      if (measure) {
        stats->copy_out_cycles += ws_read_cycles() - copy_start;
      }
    }
  }

  return FPGA_SAFE_ATTN_OK;
}
