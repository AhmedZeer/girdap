// SPDX-License-Identifier: Apache-2.0

#include "atik.h"
#include "atik_bench.h"
#include "atik_log.h"
#include "atik_ref.h"
#include "atik_types.h"

#include <stdint.h>
#include <stdio.h>

typedef struct {
  const char *name;
  int q_rows;
  int kv_rows;
  int d_k;
  int value_cols;
  int ldq;
  int ldk;
  int ldv;
  int ldo;
  uint16_t scale_bf16;
  uint32_t seed;
  float min_value;
  float max_value;
  int causal;
  int tolerance_x10000;
} atik_attention_case_t;

#define MAX_Q_ROWS 128
#define MAX_KV_ROWS 128
#define MAX_D_K 128
#define MAX_VALUE_COLS 128
#define SAMPLE_COUNT 10

static uint16_t attn_Q[MAX_Q_ROWS * MAX_D_K] __attribute__((aligned(64)));
static uint16_t attn_K[MAX_KV_ROWS * MAX_D_K] __attribute__((aligned(64)));
static uint16_t attn_V[MAX_KV_ROWS * MAX_VALUE_COLS] __attribute__((aligned(64)));
static uint16_t attn_O_sw[MAX_Q_ROWS * MAX_VALUE_COLS] __attribute__((aligned(64)));
static uint16_t attn_O_hw[MAX_Q_ROWS * MAX_VALUE_COLS] __attribute__((aligned(64)));

static const atik_attention_case_t cases[] = {
    {
        .name = "attention_q2_kv2_d2_v2_zero_scores",
        .q_rows = 2,
        .kv_rows = 2,
        .d_k = 2,
        .value_cols = 2,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3f35,
        .seed = 1,
        .min_value = 0.0f,
        .max_value = 0.0f,
        .causal = 0,
        .tolerance_x10000 = 800,
    },
    {
        .name = "attention_q4_kv4_d4_v4_rng",
        .q_rows = 4,
        .kv_rows = 4,
        .d_k = 4,
        .value_cols = 4,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3f00,
        .seed = 2,
        .min_value = -0.5f,
        .max_value = 0.5f,
        .causal = 0,
        .tolerance_x10000 = 4500,
    },
    {
        .name = "causal_attention_q4_kv4_d4_v4_rng",
        .q_rows = 4,
        .kv_rows = 4,
        .d_k = 4,
        .value_cols = 4,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3f00,
        .seed = 3,
        .min_value = -0.5f,
        .max_value = 0.5f,
        .causal = 1,
        .tolerance_x10000 = 4500,
    },
    {
        .name = "attention_q8_kv8_d8_v8_rng",
        .q_rows = 8,
        .kv_rows = 8,
        .d_k = 8,
        .value_cols = 8,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3eb5,
        .seed = 4,
        .min_value = -0.5f,
        .max_value = 0.5f,
        .causal = 0,
        .tolerance_x10000 = 5500,
    },
    {
        .name = "attention_q16_kv16_d16_v16_rng",
        .q_rows = 16,
        .kv_rows = 16,
        .d_k = 16,
        .value_cols = 16,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e80,
        .seed = 5,
        .min_value = -0.5f,
        .max_value = 0.5f,
        .causal = 0,
        .tolerance_x10000 = 7500,
    },
    {
        .name = "causal_attention_q16_kv16_d16_v16_rng",
        .q_rows = 16,
        .kv_rows = 16,
        .d_k = 16,
        .value_cols = 16,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e80,
        .seed = 6,
        .min_value = -0.5f,
        .max_value = 0.5f,
        .causal = 1,
        .tolerance_x10000 = 7500,
    },
    {
        .name = "attention_q32_kv32_d32_v32_rng",
        .q_rows = 32,
        .kv_rows = 32,
        .d_k = 32,
        .value_cols = 32,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e35,
        .seed = 7,
        .min_value = -0.375f,
        .max_value = 0.375f,
        .causal = 0,
        .tolerance_x10000 = 8500,
    },
    {
        .name = "attention_q64_kv64_d64_v64_rng",
        .q_rows = 64,
        .kv_rows = 64,
        .d_k = 64,
        .value_cols = 64,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e00,
        .seed = 8,
        .min_value = -0.25f,
        .max_value = 0.25f,
        .causal = 0,
        .tolerance_x10000 = 9500,
    },
    {
        .name = "attention_q128_kv128_d128_v128_rng",
        .q_rows = 128,
        .kv_rows = 128,
        .d_k = 128,
        .value_cols = 128,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3db5,
        .seed = 9,
        .min_value = -0.25f,
        .max_value = 0.25f,
        .causal = 0,
        .tolerance_x10000 = 11000,
    },
    {
        .name = "causal_attention_q128_kv128_d64_v16_rng",
        .q_rows = 128,
        .kv_rows = 128,
        .d_k = 64,
        .value_cols = 16,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e00,
        .seed = 10,
        .min_value = -0.25f,
        .max_value = 0.25f,
        .causal = 1,
        .tolerance_x10000 = 11000,
    },
    {
        .name = "attention_q8_kv32_d16_v64_rng",
        .q_rows = 8,
        .kv_rows = 32,
        .d_k = 16,
        .value_cols = 64,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e80,
        .seed = 11,
        .min_value = -0.5f,
        .max_value = 0.5f,
        .causal = 0,
        .tolerance_x10000 = 8000,
    },
    {
        .name = "attention_q32_kv8_d64_v16_rng",
        .q_rows = 32,
        .kv_rows = 8,
        .d_k = 64,
        .value_cols = 16,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e00,
        .seed = 12,
        .min_value = -0.375f,
        .max_value = 0.375f,
        .causal = 0,
        .tolerance_x10000 = 8500,
    },
    {
        .name = "attention_q16_kv128_d32_v8_rng",
        .q_rows = 16,
        .kv_rows = 128,
        .d_k = 32,
        .value_cols = 8,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e35,
        .seed = 13,
        .min_value = -0.25f,
        .max_value = 0.25f,
        .causal = 0,
        .tolerance_x10000 = 9000,
    },
    {
        .name = "attention_q64_kv16_d8_v128_rng",
        .q_rows = 64,
        .kv_rows = 16,
        .d_k = 8,
        .value_cols = 128,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3eb5,
        .seed = 14,
        .min_value = -0.5f,
        .max_value = 0.5f,
        .causal = 0,
        .tolerance_x10000 = 9000,
    },
    {
        .name = "attention_q128_kv64_d16_v32_rng",
        .q_rows = 128,
        .kv_rows = 64,
        .d_k = 16,
        .value_cols = 32,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3e80,
        .seed = 15,
        .min_value = -0.375f,
        .max_value = 0.375f,
        .causal = 0,
        .tolerance_x10000 = 9500,
    },
    {
        .name = "attention_q32_kv128_d128_v64_rng",
        .q_rows = 32,
        .kv_rows = 128,
        .d_k = 128,
        .value_cols = 64,
        .ldq = MAX_D_K,
        .ldk = MAX_D_K,
        .ldv = MAX_VALUE_COLS,
        .ldo = MAX_VALUE_COLS,
        .scale_bf16 = 0x3db5,
        .seed = 16,
        .min_value = -0.25f,
        .max_value = 0.25f,
        .causal = 0,
        .tolerance_x10000 = 11000,
    },
};

static void make_shape(char *shape, int size, const atik_attention_case_t *test_case) {
  snprintf(
      shape,
      size,
      "q=%d kv=%d d=%d value=%d causal=%d",
      test_case->q_rows,
      test_case->kv_rows,
      test_case->d_k,
      test_case->value_cols,
      test_case->causal);
}

static void fill_inputs(const atik_attention_case_t *test_case) {
  uint32_t seed = test_case->seed;
  atik_bench_fill_bf16(
      attn_Q,
      test_case->q_rows,
      test_case->d_k,
      test_case->ldq,
      &seed,
      test_case->min_value,
      test_case->max_value);
  atik_bench_fill_bf16(
      attn_K,
      test_case->kv_rows,
      test_case->d_k,
      test_case->ldk,
      &seed,
      test_case->min_value,
      test_case->max_value);
  atik_bench_fill_bf16(
      attn_V,
      test_case->kv_rows,
      test_case->value_cols,
      test_case->ldv,
      &seed,
      -1.0f,
      1.0f);
}

static uint64_t run_case(int case_index, const atik_attention_case_t *test_case) {
  char shape[64];
  int32_t max_diff_x10000 = 0;
  int32_t sw_sample[SAMPLE_COUNT];
  int32_t hw_sample[SAMPLE_COUNT];
  int32_t diff_sample[SAMPLE_COUNT];

  fill_inputs(test_case);
  atik_bench_zero_bf16(attn_O_sw, test_case->q_rows, test_case->value_cols, test_case->ldo);
  atik_bench_zero_bf16(attn_O_hw, test_case->q_rows, test_case->value_cols, test_case->ldo);

  const uint64_t cpu_start = atik_bench_read_cycles();
  atik_ref_attention_bf16(
      attn_Q,
      test_case->q_rows,
      test_case->d_k,
      test_case->ldq,
      attn_K,
      test_case->kv_rows,
      test_case->ldk,
      attn_V,
      test_case->value_cols,
      test_case->ldv,
      test_case->scale_bf16,
      attn_O_sw,
      test_case->ldo,
      test_case->causal);
  const uint64_t cpu_cycles = atik_bench_read_cycles() - cpu_start;

  atik_clear_counters();
  const uint64_t hw_start = atik_bench_read_cycles();
  int hw_rc;
  if (test_case->causal) {
    hw_rc = atik_causal_attention_bf16(
        attn_Q,
        test_case->q_rows,
        test_case->d_k,
        test_case->ldq,
        attn_K,
        test_case->kv_rows,
        test_case->ldk,
        attn_V,
        test_case->value_cols,
        test_case->ldv,
        test_case->scale_bf16,
        attn_O_hw,
        test_case->ldo);
  } else {
    hw_rc = atik_attention_bf16(
        attn_Q,
        test_case->q_rows,
        test_case->d_k,
        test_case->ldq,
        attn_K,
        test_case->kv_rows,
        test_case->ldk,
        attn_V,
        test_case->value_cols,
        test_case->ldv,
        test_case->scale_bf16,
        attn_O_hw,
        test_case->ldo);
  }
  const uint64_t hw_e2e_cycles = atik_bench_read_cycles() - hw_start;
  asm volatile("fence rw, rw" ::: "memory");

  int mismatches = atik_bench_compare_bf16(
      attn_O_sw,
      attn_O_hw,
      test_case->q_rows,
      test_case->value_cols,
      test_case->ldo,
      test_case->ldo,
      test_case->tolerance_x10000,
      &max_diff_x10000);
  const char *status = (hw_rc == ATIK_OK && mismatches == 0) ? "PASS" : "FAIL";
  const int total = test_case->q_rows * test_case->value_cols;
  const int sample_count = total < SAMPLE_COUNT ? total : SAMPLE_COUNT;

  atik_bench_sample_bf16(
      attn_O_sw,
      attn_O_hw,
      test_case->q_rows,
      test_case->value_cols,
      test_case->ldo,
      test_case->ldo,
      0,
      sample_count,
      sw_sample,
      hw_sample,
      diff_sample);
  make_shape(shape, sizeof(shape), test_case);

  const atik_log_stage_t stages[] = {
      {"ref_attention", cpu_cycles},
      {"atik_attention_e2e", hw_e2e_cycles},
      {"atik_attention", atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES)},
      {"compute", atik_read_counter(ATIK_COUNTER_COMPUTE_CYCLES)},
      {"dma_read", atik_read_counter(ATIK_COUNTER_DMA_READ_CYCLES)},
      {"dma_write", atik_read_counter(ATIK_COUNTER_DMA_WRITE_CYCLES)},
      {"bytes_read", atik_read_counter(ATIK_COUNTER_BYTES_READ)},
      {"bytes_written", atik_read_counter(ATIK_COUNTER_BYTES_WRITTEN)},
      {"softmax", atik_read_counter(ATIK_COUNTER_SOFTMAX_CYCLES)},
      {"tiles_loaded", atik_read_counter(ATIK_COUNTER_TILES_LOADED)},
      {"tiles_computed", atik_read_counter(ATIK_COUNTER_TILES_COMPUTED)},
      {"dma_stall", atik_read_counter(ATIK_COUNTER_DMA_STALL_CYCLES)},
  };
  const atik_log_result_t result = {
      .workload = "attention-benchmark",
      .case_index = case_index,
      .name = test_case->name,
      .shape = shape,
      .mode = test_case->causal ? "causal-attention" : "attention",
      .status = status,
      .hw_rc = (uint64_t)hw_rc,
      .raw_hw_rc = (uint64_t)hw_rc,
      .mismatches = (uint64_t)mismatches,
      .cpu_cycles = cpu_cycles,
      .hw_cycles = atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES),
      .stages = stages,
      .stage_count = (int)(sizeof(stages) / sizeof(stages[0])),
      .hw_max_abs_diff_x100000 = max_diff_x10000 * 10,
      .cpu_ref_max_abs_diff_x100000 = 0,
  };
  const atik_log_sample_t sample = {
      .workload = "attention-benchmark",
      .case_index = case_index,
      .name = test_case->name,
      .tensor = "O",
      .start = 0,
      .count = sample_count,
      .total = total,
      .hw_rc = (uint64_t)hw_rc,
      .sw_x10000 = sw_sample,
      .hw_x10000 = hw_sample,
      .diff_x10000 = diff_sample,
  };

  atik_log_result_human(&result);
  atik_log_sample(&sample);
  atik_log_result_json(&result);
  return (hw_rc == ATIK_OK && mismatches == 0) ? 0u : 1u;
}

int main(void) {
  uint64_t failures = 0;
  const int case_count = (int)(sizeof(cases) / sizeof(cases[0]));

  for (int i = 0; i < case_count; i++) {
    failures += run_case(i, &cases[i]);
  }

  return failures == 0 ? 0 : 1;
}
