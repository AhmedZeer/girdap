#include "bfloat16_utils.h"
#include "ws_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_M 8
#define TILE_N 8
#define MAX_K 256

typedef struct {
  int K;
} fused_case_t;

static uint16_t A[TILE_M][MAX_K] __attribute__((aligned(64)));
static uint16_t B[MAX_K][TILE_N] __attribute__((aligned(64)));
static uint16_t C_hw[TILE_M][TILE_N] __attribute__((aligned(64)));
static uint16_t C_ref[TILE_M][TILE_N];
static int64_t scores_fixed[TILE_M][TILE_N];
static uint16_t scores_bf16[TILE_M][TILE_N];
static uint64_t a_tiles[WS_GEMM8_A_TILE_WORDS(TILE_M, MAX_K)] __attribute__((aligned(64)));
static uint64_t b_tiles[WS_GEMM8_B_TILE_WORDS(TILE_N, MAX_K)] __attribute__((aligned(64)));
static uint64_t c_words[WS_GEMM8_C_TILE_WORDS] __attribute__((aligned(64)));

static float rand_float_bf16_range(void) {
  return ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
}

static int64_t bf16_to_fixed_s64_sat_local(uint16_t value, int int_bits, int frac_bits) {
  const int total_bits = int_bits + frac_bits;
  const uint32_t sign = (uint32_t)(value >> 15);
  const uint32_t exp = (uint32_t)((value >> 7) & 0xFFu);
  const uint32_t mantissa = (uint32_t)(value & 0x7Fu);

  if ((value & 0x7FFFu) == 0) {
    return 0;
  }

  const uint32_t ext_mantissa = (exp == 0u) ? mantissa : (0x80u | mantissa);
  const int shift_amt = (((exp == 0u) ? 1 : (int)exp) - 127) - 7 + frac_bits;

  uint64_t shifted_mag;
  if (shift_amt >= 0) {
    shifted_mag = (shift_amt >= 63) ? UINT64_MAX : ((uint64_t)ext_mantissa << shift_amt);
  } else {
    const int right_shift = -shift_amt;
    shifted_mag = (right_shift >= 64) ? 0u : ((uint64_t)ext_mantissa >> right_shift);
  }

  const uint64_t max_pos = (1ULL << (total_bits - 1)) - 1ULL;
  const uint64_t max_neg = 1ULL << (total_bits - 1);
  if (sign == 0u) {
    return (int64_t)(shifted_mag > max_pos ? max_pos : shifted_mag);
  }
  const uint64_t mag = shifted_mag > max_neg ? max_neg : shifted_mag;
  return -(int64_t)mag;
}

static void sw_score_tile_fixed(int K, uint16_t scale_bf16) {
  const int64_t scale_fixed =
      bf16_to_fixed_s64_sat_local(scale_bf16, 8, WS_GEMM_BF16_ACC_FRAC_BITS);

  for (int i = 0; i < TILE_M; i++) {
    for (int j = 0; j < TILE_N; j++) {
      int64_t acc = 0;
      for (int k = 0; k < K; k++) {
        const int16_t a_fixed = bf16_to_fixed_s16_sat(A[i][k], WS_GEMM_BF16_FIXED_FRAC_BITS);
        const int16_t b_fixed = bf16_to_fixed_s16_sat(B[k][j], WS_GEMM_BF16_FIXED_FRAC_BITS);
        acc += (int64_t)a_fixed * (int64_t)b_fixed;
      }
      scores_fixed[i][j] = acc;
      const int64_t scaled_acc = (acc * scale_fixed) >> WS_GEMM_BF16_ACC_FRAC_BITS;
      scores_bf16[i][j] = fixed_s64_to_bf16_sat(scaled_acc, WS_GEMM_BF16_ACC_FRAC_BITS);
    }
  }
}

static void sw_softmax_scores_bf16(void) {
  for (int i = 0; i < TILE_M; i++) {
    float max_val = bf16_to_float(scores_bf16[i][0]);
    for (int j = 1; j < TILE_N; j++) {
      const float value = bf16_to_float(scores_bf16[i][j]);
      if (value > max_val) {
        max_val = value;
      }
    }

    double denom = 0.0;
    for (int j = 0; j < TILE_N; j++) {
      denom += exp((double)(bf16_to_float(scores_bf16[i][j]) - max_val));
    }
    for (int j = 0; j < TILE_N; j++) {
      const double prob = exp((double)(bf16_to_float(scores_bf16[i][j]) - max_val)) / denom;
      C_ref[i][j] = float_to_bf16((float)prob);
    }
  }
}

int main(void) {
  const fused_case_t tests[] = {
      {16},
      {64},
      {100},
      {128},
      {256},
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));
  const ws_gemm_workspace_t workspace =
      ws_gemm8_make_workspace(a_tiles, b_tiles, c_words, TILE_M, TILE_N, MAX_K);

  srand(11);
  printf("=== Fused 8x8 Matmul + Row Softmax BF16 Test ===\n");
  printf("CSV_HEADER,case,M,N,K,sw_cycles,hw_e2e_cycles,hw_rc,max_abs_diff_x100000,mismatches\n");
  printf("STAGE_HEADER,case,pack_a_cycles,pack_b_cycles,preload_cycles,run_cycles,copy_out_cycles\n");

  int total_mismatches = 0;
  for (int t = 0; t < ntests; t++) {
    const int K = tests[t].K;
    const float inv_sqrt_k = 1.0f / sqrtf((float)K);
    const uint16_t scale_bf16 = float_to_bf16(inv_sqrt_k);
    ws_gemm_stats_t stats;

    for (int i = 0; i < TILE_M; i++) {
      for (int k = 0; k < K; k++) {
        A[i][k] = float_to_bf16(rand_float_bf16_range());
      }
    }
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < TILE_N; j++) {
        B[k][j] = float_to_bf16(rand_float_bf16_range());
      }
    }

    const uint64_t sw_start = ws_read_cycles();
    sw_score_tile_fixed(K, scale_bf16);
    sw_softmax_scores_bf16();
    const uint64_t sw_cycles = ws_read_cycles() - sw_start;

    const int hw_rc = ws_gemm8_matmul_softmax_tile_bf16(
        &A[0][0], MAX_K,
        &B[0][0], TILE_N,
        &C_hw[0][0], TILE_N,
        TILE_M, TILE_N, K,
        scale_bf16,
        &workspace,
        &stats);

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    if (hw_rc == WS_GEMM_OK) {
      for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
          const float ref = bf16_to_float(C_ref[i][j]);
          const float hw = bf16_to_float(C_hw[i][j]);
          const float diff = fabsf(ref - hw);
          if (diff > max_abs_diff) {
            max_abs_diff = diff;
          }
          if (diff > 0.10f) {
            if (mismatches < 8) {
              printf("Mismatch case=%d at (%d,%d): ref=", t, i, j);
              print_float_inline(ref);
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
      printf("Hardware return code error on case %d: rc=%d\n", t, hw_rc);
      mismatches++;
    }

    total_mismatches += mismatches;
    printf("CSV_DATA,%d,%d,%d,%d,%lu,%lu,%d,%lu,%d\n",
           t,
           TILE_M,
           TILE_N,
           K,
           (unsigned long)sw_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           hw_rc,
           (unsigned long)(max_abs_diff * 100000.0f),
           mismatches);
    printf("STAGE_DATA,%d,%lu,%lu,%lu,%lu,%lu\n",
           t,
           (unsigned long)stats.stage.pack_a_cycles,
           (unsigned long)stats.stage.pack_b_cycles,
           (unsigned long)stats.stage.preload_cycles,
           (unsigned long)stats.stage.run_cycles,
           (unsigned long)stats.stage.copy_out_cycles);
  }

  if (total_mismatches == 0) {
    printf("PASS: fused 8x8 matmul-softmax matched the software tolerance.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }
  return total_mismatches == 0 ? 0 : 1;
}
