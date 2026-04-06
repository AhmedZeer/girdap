#include "ws_gemm.h"

#include <stdio.h>
#include <stdlib.h>

#define MAX_M 16
#define MAX_N 16
#define MAX_K 100
#define MAX_REPEATS 4

typedef struct {
  int M;
  int N;
  int K;
  int repeats;
} reuse_case_t;

static void sw_gemm_u16(
    const uint16_t A[MAX_M][MAX_K],
    const uint16_t B[MAX_K][MAX_N],
    uint32_t C[MAX_M][MAX_N],
    int M,
    int N,
    int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      uint32_t acc = 0;
      for (int k = 0; k < K; k++) {
        acc += (uint32_t)A[i][k] * (uint32_t)B[k][j];
      }
      C[i][j] = acc;
    }
  }
}

static void accumulate_stats(
    ws_gemm_stats_t *dst,
    const ws_gemm_stats_t *src) {
  dst->hw_e2e_cycles += src->hw_e2e_cycles;
  dst->stage.pack_a_cycles += src->stage.pack_a_cycles;
  dst->stage.pack_b_cycles += src->stage.pack_b_cycles;
  dst->stage.preload_cycles += src->stage.preload_cycles;
  dst->stage.run_cycles += src->stage.run_cycles;
  dst->stage.copy_out_cycles += src->stage.copy_out_cycles;
  for (int i = 0; i < WS_PERF_COUNT; i++) {
    dst->perf[i] += src->perf[i];
  }
}

static int count_mismatches(
    const uint32_t C_sw[MAX_REPEATS][MAX_M][MAX_N],
    const uint32_t C_hw[MAX_REPEATS][MAX_M][MAX_N],
    int repeats,
    int M,
    int N) {
  int mismatches = 0;
  for (int r = 0; r < repeats; r++) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        if (C_sw[r][i][j] != C_hw[r][i][j]) {
          mismatches++;
        }
      }
    }
  }
  return mismatches;
}

int main(void) {
  static uint16_t A[MAX_REPEATS][MAX_M][MAX_K] __attribute__((aligned(64)));
  static uint16_t B[MAX_K][MAX_N] __attribute__((aligned(64)));
  static uint32_t C_sw[MAX_REPEATS][MAX_M][MAX_N];
  static uint32_t C_hw_normal[MAX_REPEATS][MAX_M][MAX_N];
  static uint32_t C_hw_prepared[MAX_REPEATS][MAX_M][MAX_N];
  static uint64_t a_tiles[WS_GEMM_A_TILE_WORDS(MAX_M, MAX_K)] __attribute__((aligned(64)));
  static uint64_t b_tiles[WS_GEMM_B_TILE_WORDS(MAX_N, MAX_K)] __attribute__((aligned(64)));
  static uint64_t c_words[WS_GEMM_C_TILE_WORDS] __attribute__((aligned(64)));

  const ws_gemm_workspace_t workspace =
      ws_gemm_make_workspace(a_tiles, b_tiles, c_words, MAX_M, MAX_N, MAX_K);

  const reuse_case_t cases[] = {
      {8, 8, 100, 4},
      {16, 16, 100, 4},
  };
  const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

  srand(1);
  printf("=== Systolic GEMM Weight Reuse Test (Weight-Stationary) ===\n");
  printf("CSV_HEADER,case,repeats,M,N,K,normal_total_cycles,prepared_pack_cycles,prepared_total_cycles,normal_rc,prepared_rc,mismatches\n");
  printf("STAGE_HEADER,case,mode,pack_a_cycles,pack_b_cycles,preload_cycles,run_cycles,copy_out_cycles\n");

  for (int t = 0; t < ncases; t++) {
    const int M = cases[t].M;
    const int N = cases[t].N;
    const int K = cases[t].K;
    const int repeats = cases[t].repeats;
    ws_gemm_stats_t normal_total;
    ws_gemm_stats_t prepared_total;
    ws_gemm_packed_b_t packed_b;
    uint64_t prepared_pack_cycles = 0;
    int normal_rc = WS_GEMM_OK;
    int prepared_rc = WS_GEMM_OK;

    ws_gemm_clear_stats(&normal_total);
    ws_gemm_clear_stats(&prepared_total);

    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        B[k][j] = (uint16_t)(rand() % 16);
      }
    }

    for (int r = 0; r < repeats; r++) {
      for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
          A[r][i][k] = (uint16_t)(rand() % 16);
        }
      }
      sw_gemm_u16(A[r], B, C_sw[r], M, N, K);
    }

    for (int r = 0; r < repeats; r++) {
      ws_gemm_stats_t stats;
      const int rc = ws_gemm_u16(
          &A[r][0][0], MAX_K,
          &B[0][0], MAX_N,
          &C_hw_normal[r][0][0], MAX_N,
          M, N, K,
          &workspace,
          &stats);
      if (rc != WS_GEMM_OK && normal_rc == WS_GEMM_OK) {
        normal_rc = rc;
      }
      accumulate_stats(&normal_total, &stats);
    }

    prepared_rc = ws_gemm_prepare_packed_b_u16(
        &B[0][0], MAX_N,
        N, K,
        b_tiles,
        MAX_N, MAX_K,
        &packed_b,
        &prepared_pack_cycles);

    if (prepared_rc == WS_GEMM_OK) {
      for (int r = 0; r < repeats; r++) {
        ws_gemm_stats_t stats;
        const int rc = ws_gemm_u16_prepacked_b(
            &A[r][0][0], MAX_K,
            &packed_b,
            M,
            &C_hw_prepared[r][0][0], MAX_N,
            &workspace,
            &stats);
        if (rc != WS_GEMM_OK && prepared_rc == WS_GEMM_OK) {
          prepared_rc = rc;
        }
        accumulate_stats(&prepared_total, &stats);
      }
    }

    int mismatches = 0;
    if (normal_rc == WS_GEMM_OK) {
      mismatches += count_mismatches(C_sw, C_hw_normal, repeats, M, N);
    } else {
      mismatches += 1;
    }
    if (prepared_rc == WS_GEMM_OK) {
      mismatches += count_mismatches(C_sw, C_hw_prepared, repeats, M, N);
    } else {
      mismatches += 1;
    }

    printf("CSV_DATA,%d,%d,%d,%d,%d,%lu,%lu,%lu,%d,%d,%d\n",
           t,
           repeats,
           M,
           N,
           K,
           (unsigned long)normal_total.hw_e2e_cycles,
           (unsigned long)prepared_pack_cycles,
           (unsigned long)prepared_total.hw_e2e_cycles,
           normal_rc,
           prepared_rc,
           mismatches);
    printf("STAGE_DATA,%d,normal,%lu,%lu,%lu,%lu,%lu\n",
           t,
           (unsigned long)normal_total.stage.pack_a_cycles,
           (unsigned long)normal_total.stage.pack_b_cycles,
           (unsigned long)normal_total.stage.preload_cycles,
           (unsigned long)normal_total.stage.run_cycles,
           (unsigned long)normal_total.stage.copy_out_cycles);
    printf("STAGE_DATA,%d,prepared,%lu,%lu,%lu,%lu,%lu\n",
           t,
           (unsigned long)prepared_total.stage.pack_a_cycles,
           (unsigned long)prepared_total.stage.pack_b_cycles,
           (unsigned long)prepared_total.stage.preload_cycles,
           (unsigned long)prepared_total.stage.run_cycles,
           (unsigned long)prepared_total.stage.copy_out_cycles);
  }

  return 0;
}
