#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "ws_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GEMM_MAX_M 128
#define GEMM_MAX_N 128
#define GEMM_MAX_K 128

#define ATTN_MAX_Q_ROWS 128
#define ATTN_MAX_KV_ROWS 256
#define ATTN_MAX_D_K 128
#define ATTN_MAX_VALUE_COLS 128
#define ATTN_TOLERANCE 0.25f

typedef struct {
  int M;
  int N;
  int K;
} gemm_case_t;

typedef struct {
  int q_rows;
  int kv_rows;
  int d_k;
  int value_cols;
} attention_case_t;

static uint16_t gemm_a[GEMM_MAX_M][GEMM_MAX_K] __attribute__((aligned(64)));
static uint16_t gemm_b[GEMM_MAX_K][GEMM_MAX_N] __attribute__((aligned(64)));
static int64_t gemm_ref_fixed[GEMM_MAX_M][GEMM_MAX_N];
static uint16_t gemm_ref_bf16[GEMM_MAX_M][GEMM_MAX_N];
static uint16_t gemm_hw[GEMM_MAX_M][GEMM_MAX_N] __attribute__((aligned(64)));
static uint64_t gemm_a_tiles[WS_GEMM8_A_TILE_WORDS(GEMM_MAX_M, GEMM_MAX_K)] __attribute__((aligned(64)));
static uint64_t gemm_b_tiles[WS_GEMM8_B_TILE_WORDS(GEMM_MAX_N, GEMM_MAX_K)] __attribute__((aligned(64)));
static uint64_t gemm_c_words[WS_GEMM8_C_TILE_WORDS] __attribute__((aligned(64)));

static uint16_t q_matrix[ATTN_MAX_Q_ROWS][ATTN_MAX_D_K] __attribute__((aligned(64)));
static uint16_t k_matrix[ATTN_MAX_KV_ROWS][ATTN_MAX_D_K] __attribute__((aligned(64)));
static uint16_t v_matrix[ATTN_MAX_KV_ROWS][ATTN_MAX_VALUE_COLS] __attribute__((aligned(64)));
static uint16_t attn_hw[ATTN_MAX_Q_ROWS][ATTN_MAX_VALUE_COLS] __attribute__((aligned(64)));
static float attn_scores[ATTN_MAX_Q_ROWS][ATTN_MAX_KV_ROWS];
static float attn_probs[ATTN_MAX_Q_ROWS][ATTN_MAX_KV_ROWS];
static float attn_ref[ATTN_MAX_Q_ROWS][ATTN_MAX_VALUE_COLS];
static uint64_t q_tiles[FPGA_SAFE_ATTN_Q_TILE_WORDS(ATTN_MAX_Q_ROWS, ATTN_MAX_D_K)] __attribute__((aligned(64)));
static uint64_t k_tiles[FPGA_SAFE_ATTN_K_TILE_WORDS(ATTN_MAX_KV_ROWS, ATTN_MAX_D_K)] __attribute__((aligned(64)));
static uint64_t v_tiles[FPGA_SAFE_ATTN_V_TILE_WORDS(ATTN_MAX_VALUE_COLS, ATTN_MAX_KV_ROWS)] __attribute__((aligned(64)));
static uint64_t attn_out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

static float rand_float_unit(void) {
  return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void sw_gemm_bf16_fixed_ref(int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int64_t acc = 0;
      for (int k = 0; k < K; k++) {
        const int16_t a_fixed =
            bf16_to_fixed_s16_sat(gemm_a[i][k], WS_GEMM_BF16_FIXED_FRAC_BITS);
        const int16_t b_fixed =
            bf16_to_fixed_s16_sat(gemm_b[k][j], WS_GEMM_BF16_FIXED_FRAC_BITS);
        acc += (int64_t)a_fixed * (int64_t)b_fixed;
      }
      gemm_ref_fixed[i][j] = acc;
      gemm_ref_bf16[i][j] =
          fixed_s64_to_bf16_sat(acc, WS_GEMM_BF16_ACC_FRAC_BITS);
    }
  }
}

static int run_gemm_section(void) {
  const gemm_case_t tests[] = {
      {8, 8, 16},
      {8, 32, 64},
      {32, 8, 64},
      {16, 16, 32},
      {16, 64, 64},
      {64, 16, 64},
      {32, 32, 64},
      {64, 64, 64},
      {128, 64, 64},
      {64, 128, 64},
      {128, 128, 128},
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));
  const ws_gemm_workspace_t workspace =
      ws_gemm8_make_workspace(
          gemm_a_tiles,
          gemm_b_tiles,
          gemm_c_words,
          GEMM_MAX_M,
          GEMM_MAX_N,
          GEMM_MAX_K);

  int total_mismatches = 0;
  printf("=== Dual-RoCC Matmul custom0 Sanity ===\n");
  printf("GEMM_HEADER,case,M,N,K,sw_cycles,hw_e2e_cycles,pack_a_cycles,pack_b_cycles,preload_cycles,run_cycles,copy_out_cycles,hw_rc,speedup_x100,mismatches\n");

  srand(3);
  for (int t = 0; t < ntests; t++) {
    const gemm_case_t tc = tests[t];
    for (int i = 0; i < tc.M; i++) {
      for (int k = 0; k < tc.K; k++) {
        gemm_a[i][k] = float_to_bf16(rand_float_unit());
      }
    }
    for (int k = 0; k < tc.K; k++) {
      for (int j = 0; j < tc.N; j++) {
        gemm_b[k][j] = float_to_bf16(rand_float_unit());
      }
    }

    const uint64_t sw_start = ws_read_cycles();
    sw_gemm_bf16_fixed_ref(tc.M, tc.N, tc.K);
    const uint64_t sw_cycles = ws_read_cycles() - sw_start;

    ws_gemm_stats_t stats;
    const int hw_rc = ws_gemm8_bf16(
        &gemm_a[0][0],
        GEMM_MAX_K,
        &gemm_b[0][0],
        GEMM_MAX_N,
        &gemm_hw[0][0],
        GEMM_MAX_N,
        tc.M,
        tc.N,
        tc.K,
        &workspace,
        &stats);

    int mismatches = 0;
    if (hw_rc == WS_GEMM_OK) {
      for (int i = 0; i < tc.M; i++) {
        for (int j = 0; j < tc.N; j++) {
          if (gemm_hw[i][j] != gemm_ref_bf16[i][j]) {
            if (mismatches < 4) {
              printf("GEMM_MISMATCH,case=%d,row=%d,col=%d,ref=0x%04x,hw=0x%04x\n",
                     t,
                     i,
                     j,
                     (unsigned)gemm_ref_bf16[i][j],
                     (unsigned)gemm_hw[i][j]);
            }
            mismatches++;
          }
        }
      }
    } else {
      mismatches = 1;
    }

    total_mismatches += mismatches;
    const uint64_t hw_e2e_cycles = stats.hw_e2e_cycles;
    const uint64_t speedup_x100 =
        hw_e2e_cycles == 0 ? 0 : (sw_cycles * 100u) / hw_e2e_cycles;
    printf("GEMM_DATA,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%d,%lu,%d\n",
           t,
           tc.M,
           tc.N,
           tc.K,
           (unsigned long)sw_cycles,
           (unsigned long)hw_e2e_cycles,
           (unsigned long)stats.stage.pack_a_cycles,
           (unsigned long)stats.stage.pack_b_cycles,
           (unsigned long)stats.stage.preload_cycles,
           (unsigned long)stats.stage.run_cycles,
           (unsigned long)stats.stage.copy_out_cycles,
           hw_rc,
           (unsigned long)speedup_x100,
           mismatches);
  }

  return total_mismatches;
}

static void sw_attention_reference(
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols,
    float scale) {
  for (int qr = 0; qr < q_rows; qr++) {
    for (int kv = 0; kv < kv_rows; kv++) {
      double dot = 0.0;
      for (int d = 0; d < d_k; d++) {
        dot += (double)bf16_to_float(q_matrix[qr][d]) *
               (double)bf16_to_float(k_matrix[kv][d]);
      }
      attn_scores[qr][kv] = (float)(dot * (double)scale);
    }
  }

  for (int qr = 0; qr < q_rows; qr++) {
    float max_score = -INFINITY;
    for (int kv = 0; kv < kv_rows; kv++) {
      if (attn_scores[qr][kv] > max_score) {
        max_score = attn_scores[qr][kv];
      }
    }

    double denom = 0.0;
    for (int kv = 0; kv < kv_rows; kv++) {
      denom += exp((double)(attn_scores[qr][kv] - max_score));
    }
    for (int kv = 0; kv < kv_rows; kv++) {
      attn_probs[qr][kv] =
          (float)(exp((double)(attn_scores[qr][kv] - max_score)) / denom);
    }
  }

  for (int qr = 0; qr < q_rows; qr++) {
    for (int vc = 0; vc < value_cols; vc++) {
      double acc = 0.0;
      for (int kv = 0; kv < kv_rows; kv++) {
        acc += (double)attn_probs[qr][kv] * (double)bf16_to_float(v_matrix[kv][vc]);
      }
      attn_ref[qr][vc] = (float)acc;
    }
  }
}

static int run_attention_section(void) {
  const attention_case_t tests[] = {
      {8, 8, 16, 8},
      {8, 9, 16, 8},
      {8, 16, 16, 8},
      {8, 64, 64, 8},
      {16, 64, 64, 16},
      {32, 64, 64, 32},
      {32, 128, 64, 32},
      {32, 256, 128, 64},
      {64, 128, 64, 64},
      {128, 128, 64, 64},
      {128, 256, 128, 64},
      {128, 256, 128, 128},
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));
  const fpga_safe_attention_workspace_t workspace =
      fpga_safe_attention_make_workspace(
          q_tiles,
          k_tiles,
          v_tiles,
          attn_out_words,
          ATTN_MAX_Q_ROWS,
          ATTN_MAX_KV_ROWS,
          ATTN_MAX_D_K,
          ATTN_MAX_VALUE_COLS);

  int total_mismatches = 0;
  printf("=== Dual-RoCC Attention custom1 Sanity ===\n");
  printf("ATTN_HEADER,case,q_rows,kv_rows,d_k,value_cols,sw_cycles,hw_e2e_cycles,hw_accel_cycles,q_pack_cycles,k_pack_cycles,v_pack_cycles,hw_score_cycles,hw_value_cycles,copy_out_cycles,hw_rc,raw_hw_rc,speedup_x100,max_abs_diff_x100000,mismatches\n");

  srand(17);
  for (int t = 0; t < ntests; t++) {
    const attention_case_t tc = tests[t];
    const float scale = 1.0f / sqrtf((float)tc.d_k);
    const uint16_t scale_bf16 = float_to_bf16(scale);

    for (int r = 0; r < tc.q_rows; r++) {
      for (int c = 0; c < tc.d_k; c++) {
        q_matrix[r][c] = float_to_bf16(rand_float_unit());
      }
    }
    for (int r = 0; r < tc.kv_rows; r++) {
      for (int c = 0; c < tc.d_k; c++) {
        k_matrix[r][c] = float_to_bf16(rand_float_unit());
      }
      for (int c = 0; c < tc.value_cols; c++) {
        v_matrix[r][c] = float_to_bf16(rand_float_unit());
      }
    }
    for (int r = 0; r < tc.q_rows; r++) {
      for (int c = 0; c < tc.value_cols; c++) {
        attn_hw[r][c] = 0;
      }
    }

    const uint64_t sw_start = ws_read_cycles();
    sw_attention_reference(tc.q_rows, tc.kv_rows, tc.d_k, tc.value_cols, scale);
    const uint64_t sw_cycles = ws_read_cycles() - sw_start;

    fpga_safe_attention_stats_t stats;
    const int hw_rc = fpga_safe_attention_bf16_hwpack(
        &q_matrix[0][0],
        ATTN_MAX_D_K,
        &k_matrix[0][0],
        ATTN_MAX_D_K,
        &v_matrix[0][0],
        ATTN_MAX_VALUE_COLS,
        tc.q_rows,
        tc.kv_rows,
        tc.d_k,
        tc.value_cols,
        scale_bf16,
        &attn_hw[0][0],
        ATTN_MAX_VALUE_COLS,
        &workspace,
        &stats);

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    if (hw_rc == FPGA_SAFE_ATTN_OK) {
      for (int r = 0; r < tc.q_rows; r++) {
        for (int c = 0; c < tc.value_cols; c++) {
          const float hw = bf16_to_float(attn_hw[r][c]);
          const float diff = fabsf(hw - attn_ref[r][c]);
          if (diff > max_abs_diff) {
            max_abs_diff = diff;
          }
          if (diff > ATTN_TOLERANCE) {
            if (mismatches < 4) {
              printf("ATTN_MISMATCH,case=%d,row=%d,col=%d,ref=", t, r, c);
              print_float_inline(attn_ref[r][c]);
              printf("hw=");
              print_float_inline(hw);
              printf("diff=");
              print_float_inline(diff);
              printf("\n");
            }
            mismatches++;
          }
        }
      }
    } else {
      mismatches = 1;
    }

    total_mismatches += mismatches;
    const uint64_t hw_total_cycles =
        stats.q_pack_cycles +
        stats.k_pack_cycles +
        stats.v_pack_cycles +
        stats.hw_e2e_cycles +
        stats.copy_out_cycles;
    const uint64_t speedup_x100 =
        hw_total_cycles == 0 ? 0 : (sw_cycles * 100u) / hw_total_cycles;
    printf("ATTN_DATA,%d,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%d,%lu,%lu,%lu,%d\n",
           t,
           tc.q_rows,
           tc.kv_rows,
           tc.d_k,
           tc.value_cols,
           (unsigned long)sw_cycles,
           (unsigned long)hw_total_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           (unsigned long)stats.q_pack_cycles,
           (unsigned long)stats.k_pack_cycles,
           (unsigned long)stats.v_pack_cycles,
           (unsigned long)stats.score_cycles,
           (unsigned long)stats.value_cycles,
           (unsigned long)stats.copy_out_cycles,
           hw_rc,
           (unsigned long)stats.raw_hw_rc,
           (unsigned long)speedup_x100,
           (unsigned long)(max_abs_diff * 100000.0f),
           mismatches);
  }

  return total_mismatches;
}

int main(void) {
  printf("=== FPGA-Safe Dual-RoCC Matmul + Attention Test ===\n");
  printf("OPCODE_INFO,matmul=%d,attention=%d\n", SA_MATMUL_OPCODE, SA_ATTN_OPCODE);

  const int gemm_mismatches = run_gemm_section();
  const int attention_mismatches = run_attention_section();
  const int total_mismatches = gemm_mismatches + attention_mismatches;

  if (total_mismatches == 0) {
    printf("PASS: dual-RoCC matmul and attention sanity tests passed.\n");
    return 0;
  }

  printf("FAIL: dual-RoCC total mismatches = %d\n", total_mismatches);
  return 1;
}
