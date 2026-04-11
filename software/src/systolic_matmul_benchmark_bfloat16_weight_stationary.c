#include "ws_gemm.h"

#include <stdio.h>
#include <stdlib.h>

#define MAX_DIM 256
#define CPU_BLOCK 32

typedef struct {
  int M;
  int N;
  int K;
} gemm_case_t;

static void cpu_gemm_blocked_u16(
    const uint16_t A[MAX_DIM][MAX_DIM],
    const uint16_t B[MAX_DIM][MAX_DIM],
    uint32_t C[MAX_DIM][MAX_DIM],
    int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  for (int ii = 0; ii < M; ii += CPU_BLOCK) {
    int i_end = ii + CPU_BLOCK;
    if (i_end > M) i_end = M;
    for (int kk = 0; kk < K; kk += CPU_BLOCK) {
      int k_end = kk + CPU_BLOCK;
      if (k_end > K) k_end = K;
      for (int jj = 0; jj < N; jj += CPU_BLOCK) {
        int j_end = jj + CPU_BLOCK;
        if (j_end > N) j_end = N;
        for (int i = ii; i < i_end; i++) {
          for (int k = kk; k < k_end; k++) {
            uint32_t a = (uint32_t)A[i][k];
            for (int j = jj; j < j_end; j++) {
              C[i][j] += a * (uint32_t)B[k][j];
            }
          }
        }
      }
    }
  }
}

static int count_mismatches(
    const uint32_t A[MAX_DIM][MAX_DIM],
    const uint32_t B[MAX_DIM][MAX_DIM],
    int M, int N) {
  int mismatches = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (A[i][j] != B[i][j]) {
        mismatches++;
      }
    }
  }
  return mismatches;
}

int main(void) {
  static uint16_t A[MAX_DIM][MAX_DIM] __attribute__((aligned(64)));
  static uint16_t B[MAX_DIM][MAX_DIM] __attribute__((aligned(64)));
  static uint32_t C_cpu[MAX_DIM][MAX_DIM];
  static uint32_t C_hw[MAX_DIM][MAX_DIM];
  static uint64_t a_tiles[WS_GEMM_A_TILE_WORDS(MAX_DIM, MAX_DIM)] __attribute__((aligned(64)));
  static uint64_t b_tiles[WS_GEMM_B_TILE_WORDS(MAX_DIM, MAX_DIM)] __attribute__((aligned(64)));
  static uint64_t c_words[WS_GEMM_C_TILE_WORDS] __attribute__((aligned(64)));

  const ws_gemm_workspace_t workspace =
      ws_gemm_make_workspace(a_tiles, b_tiles, c_words, MAX_DIM, MAX_DIM, MAX_DIM);

  const gemm_case_t cases[] = {
    {128, 128, 128},
    {256, 256, 256},
  };
  const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

  srand(1);
  printf("=== Systolic GEMM Benchmark (Weight-Stationary 4x4) ===\n");
  printf("CSV_HEADER,case,M,N,K,cpu_cycles,hw_total_cycles,hw_rc,mismatches\n");
  printf("PERF_HEADER,case,busy_cycles,run_cmds,preload_cmds,preload_reuse_hits,chunks_started,feed_cycles,capture_rows,tl_b_reads,tl_a_reads,tl_c_writes,wait_fill_b_cycles,wait_fill_a_cycles,load_weight_cycles,wait_chunk_out_cycles,wait_put_cycles\n");
  printf("STAGE_HEADER,case,pack_a_cycles,pack_b_cycles,preload_cycles,run_cycles,copy_out_cycles\n");

  for (int c = 0; c < ncases; c++) {
    const int M = cases[c].M;
    const int N = cases[c].N;
    const int K = cases[c].K;
    ws_gemm_stats_t stats;

    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        A[i][k] = (uint16_t)(rand() & 0xF);
      }
    }
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        B[k][j] = (uint16_t)(rand() & 0xF);
      }
    }

    uint64_t cpu_start = ws_read_cycles();
    cpu_gemm_blocked_u16(A, B, C_cpu, M, N, K);
    uint64_t cpu_cycles = ws_read_cycles() - cpu_start;

    int hw_rc = ws_gemm_u16(
        &A[0][0], MAX_DIM,
        &B[0][0], MAX_DIM,
        &C_hw[0][0], MAX_DIM,
        M, N, K,
        &workspace,
        &stats);

    int mismatches = (hw_rc == WS_GEMM_OK) ? count_mismatches(C_cpu, C_hw, M, N) : -1;

    printf("CSV_DATA,%d,%d,%d,%d,%lu,%lu,%d,%d\n",
           c, M, N, K,
           (unsigned long)cpu_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           hw_rc,
           mismatches);
    printf("PERF_DATA,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n",
           c,
           (unsigned long)stats.perf[WS_PERF_BUSY_CYCLES],
           (unsigned long)stats.perf[WS_PERF_RUN_CMDS],
           (unsigned long)stats.perf[WS_PERF_PRELOAD_CMDS],
           (unsigned long)stats.perf[WS_PERF_PRELOAD_REUSE_HITS],
           (unsigned long)stats.perf[WS_PERF_CHUNKS_STARTED],
           (unsigned long)stats.perf[WS_PERF_FEED_CYCLES],
           (unsigned long)stats.perf[WS_PERF_CAPTURE_ROWS],
           (unsigned long)stats.perf[WS_PERF_TL_B_READS],
           (unsigned long)stats.perf[WS_PERF_TL_A_READS],
           (unsigned long)stats.perf[WS_PERF_TL_C_WRITES],
           (unsigned long)stats.perf[WS_PERF_WAIT_FILL_B_CYCLES],
           (unsigned long)stats.perf[WS_PERF_WAIT_FILL_A_CYCLES],
           (unsigned long)stats.perf[WS_PERF_LOAD_WEIGHT_CYCLES],
           (unsigned long)stats.perf[WS_PERF_WAIT_CHUNK_OUT_CYCLES],
           (unsigned long)stats.perf[WS_PERF_WAIT_PUT_CYCLES]);
    printf("STAGE_DATA,%d,%lu,%lu,%lu,%lu,%lu\n",
           c,
           (unsigned long)stats.stage.pack_a_cycles,
           (unsigned long)stats.stage.pack_b_cycles,
           (unsigned long)stats.stage.preload_cycles,
           (unsigned long)stats.stage.run_cycles,
           (unsigned long)stats.stage.copy_out_cycles);
  }

  return 0;
}
