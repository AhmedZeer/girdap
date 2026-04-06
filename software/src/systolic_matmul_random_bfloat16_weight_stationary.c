#include "bfloat16_utils.h"
#include "ws_gemm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_M 16
#define MAX_N 16
#define MAX_K 100

typedef struct {
  int M;
  int N;
  int K;
} gemm_case_t;

static volatile float sw_float_checksum_sink;

static float rand_float_bf16_range(void) {
  return ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
}

static void sw_gemm_bf16_float(
    const uint16_t A[MAX_M][MAX_K],
    const uint16_t B[MAX_K][MAX_N],
    float C[MAX_M][MAX_N],
    int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += bf16_to_float(A[i][k]) * bf16_to_float(B[k][j]);
      }
      C[i][j] = acc;
    }
  }
}

static void sw_gemm_bf16_fixed_ref(
    const uint16_t A[MAX_M][MAX_K],
    const uint16_t B[MAX_K][MAX_N],
    int64_t C[MAX_M][MAX_N],
    int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int64_t acc = 0;
      for (int k = 0; k < K; k++) {
        const int16_t a_fixed = bf16_to_fixed_s16_sat(A[i][k], WS_GEMM_BF16_FIXED_FRAC_BITS);
        const int16_t b_fixed = bf16_to_fixed_s16_sat(B[k][j], WS_GEMM_BF16_FIXED_FRAC_BITS);
        acc += (int64_t)a_fixed * (int64_t)b_fixed;
      }
      C[i][j] = acc;
    }
  }
}

static void quantize_fixed_ref_to_bf16(
    const int64_t C_fixed[MAX_M][MAX_N],
    uint16_t C_bf16[MAX_M][MAX_N],
    int M,
    int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C_bf16[i][j] = fixed_s64_to_bf16_sat(C_fixed[i][j], WS_GEMM_BF16_ACC_FRAC_BITS);
    }
  }
}

static void quantize_float_output_to_bf16(
    const float C_float[MAX_M][MAX_N],
    uint16_t C_bf16[MAX_M][MAX_N],
    int M,
    int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C_bf16[i][j] = float_to_bf16(C_float[i][j]);
    }
  }
}

int main(void) {
  static uint16_t A[MAX_M][MAX_K] __attribute__((aligned(64)));
  static uint16_t B[MAX_K][MAX_N] __attribute__((aligned(64)));
  static float C_sw[MAX_M][MAX_N];
  static uint16_t C_sw_bf16[MAX_M][MAX_N];
  static int64_t C_ref[MAX_M][MAX_N];
  static uint16_t C_ref_bf16[MAX_M][MAX_N];
  static uint16_t C_hw[MAX_M][MAX_N];
  static uint64_t a_tiles[WS_GEMM_A_TILE_WORDS(MAX_M, MAX_K)] __attribute__((aligned(64)));
  static uint64_t b_tiles[WS_GEMM_B_TILE_WORDS(MAX_N, MAX_K)] __attribute__((aligned(64)));
  static uint64_t c_words[WS_GEMM_C_TILE_WORDS] __attribute__((aligned(64)));

  const ws_gemm_workspace_t workspace =
      ws_gemm_make_workspace(a_tiles, b_tiles, c_words, MAX_M, MAX_N, MAX_K);

  const gemm_case_t tests[] = {
      {4, 4, 100},
      {8, 8, 100},
      {16, 16, 100},
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));

  srand(1);
  printf("=== Systolic GEMM Random Test (BF16 Weight-Stationary) ===\n");
  printf("CSV_HEADER,case,M,N,K,sw_cycles,hw_e2e_cycles,hw_rc,mismatches\n");
  printf("PERF_HEADER,case,busy_cycles,run_cmds,preload_cmds,preload_reuse_hits,chunks_started,feed_cycles,capture_rows,tl_b_reads,tl_a_reads,tl_c_writes,wait_fill_b_cycles,wait_fill_a_cycles,load_weight_cycles,wait_chunk_out_cycles,wait_put_cycles\n");
  printf("STAGE_HEADER,case,pack_a_cycles,pack_b_cycles,preload_cycles,run_cycles,copy_out_cycles\n");

  int total_mismatches = 0;
  for (int t = 0; t < ntests; t++) {
    const int M = tests[t].M;
    const int N = tests[t].N;
    const int K = tests[t].K;
    ws_gemm_stats_t stats;

    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        A[i][k] = float_to_bf16(rand_float_bf16_range());
      }
    }
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        B[k][j] = float_to_bf16(rand_float_bf16_range());
      }
    }

    uint64_t sw_start = ws_read_cycles();
    sw_gemm_bf16_float(A, B, C_sw, M, N, K);
    quantize_float_output_to_bf16(C_sw, C_sw_bf16, M, N);
    uint64_t sw_cycles = ws_read_cycles() - sw_start;
    float sw_checksum = 0.0f;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        sw_checksum += bf16_to_float(C_sw_bf16[i][j]);
      }
    }
    sw_float_checksum_sink = sw_checksum;

    sw_gemm_bf16_fixed_ref(A, B, C_ref, M, N, K);
    quantize_fixed_ref_to_bf16(C_ref, C_ref_bf16, M, N);

    const int hw_rc = ws_gemm_bf16(
        &A[0][0], MAX_K,
        &B[0][0], MAX_N,
        &C_hw[0][0], MAX_N,
        M, N, K,
        &workspace,
        &stats);

    int mismatches = 0;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        if (C_ref_bf16[i][j] != C_hw[i][j]) {
          if (mismatches < 8) {
            const float sw_val = bf16_to_float(C_ref_bf16[i][j]);
            const float hw_val = bf16_to_float(C_hw[i][j]);
            printf("Mismatch case=%d at (%d,%d): sw_bf16=0x%04x hw_bf16=0x%04x sw=",
                   t, i, j,
                   (unsigned)C_ref_bf16[i][j],
                   (unsigned)C_hw[i][j]);
            print_float_inline(sw_val);
            printf("hw=");
            print_float_inline(hw_val);
            printf("\n");
          }
          mismatches++;
        }
      }
    }

    if (hw_rc != WS_GEMM_OK) {
      printf("Hardware return code error on case %d: rc=%d\n", t, hw_rc);
      mismatches += 1;
    }

    total_mismatches += mismatches;
    printf("CSV_DATA,%d,%d,%d,%d,%lu,%lu,%d,%d\n",
           t, M, N, K,
           (unsigned long)sw_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           hw_rc,
           mismatches);
    printf("PERF_DATA,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n",
           t,
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
           t,
           (unsigned long)stats.stage.pack_a_cycles,
           (unsigned long)stats.stage.pack_b_cycles,
           (unsigned long)stats.stage.preload_cycles,
           (unsigned long)stats.stage.run_cycles,
           (unsigned long)stats.stage.copy_out_cycles);
  }

  if (total_mismatches == 0) {
    printf("PASS: all randomized BF16 GEMM cases matched the exact BF16 reference.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }
  return 0;
}
