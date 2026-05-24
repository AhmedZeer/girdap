// SPDX-License-Identifier: Apache-2.0

#include "atik.h"
#include "atik_bench.h"
#include "atik_log.h"
#include "atik_ref.h"
#include "atik_types.h"

#include <stdint.h>
#include <stdio.h>

#define MAX_M 16
#define MAX_N 16
#define MAX_K 16
#define SAMPLE_COUNT 10

static uint16_t mat_A[MAX_M * MAX_K] __attribute__((aligned(64)));
static uint16_t mat_B[MAX_K * MAX_N] __attribute__((aligned(64)));
static uint16_t mat_C_sw[MAX_M * MAX_N] __attribute__((aligned(64)));
static uint16_t mat_C_hw[MAX_M * MAX_N] __attribute__((aligned(64)));

static const atik_matmul_case_t cases[] = {
    {
        .name = "matmul_m2_n2_k2_rng",
        .m = 2,
        .n = 2,
        .k = 2,
        .lda = MAX_K,
        .ldb = MAX_N,
        .ldc = MAX_N,
        .seed = 1,
        .min_value = -2.0f,
        .max_value = 2.0f,
        .tolerance_x10000 = 800,
    },
    {
        .name = "matmul_m4_n4_k4_rng",
        .m = 4,
        .n = 4,
        .k = 4,
        .lda = MAX_K,
        .ldb = MAX_N,
        .ldc = MAX_N,
        .seed = 2,
        .min_value = -1.0f,
        .max_value = 1.0f,
        .tolerance_x10000 = 1200,
    },
    {
        .name = "matmul_m8_n8_k8_rng",
        .m = 8,
        .n = 8,
        .k = 8,
        .lda = MAX_K,
        .ldb = MAX_N,
        .ldc = MAX_N,
        .seed = 3,
        .min_value = -1.0f,
        .max_value = 1.0f,
        .tolerance_x10000 = 1800,
    },
    {
        .name = "matmul_m16_n16_k16_rng",
        .m = 16,
        .n = 16,
        .k = 16,
        .lda = MAX_K,
        .ldb = MAX_N,
        .ldc = MAX_N,
        .seed = 3,
        .min_value = -1.5f,
        .max_value = 1.5f,
        .tolerance_x10000 = 1800,
    },
};

static void make_shape(char *shape, int size, const atik_matmul_case_t *test_case) {
  snprintf(shape, size, "M=%d N=%d K=%d", test_case->m, test_case->n, test_case->k);
}

static uint64_t run_case(int case_index, const atik_matmul_case_t *test_case) {
  uint32_t seed = test_case->seed;
  char shape[48];
  int32_t max_diff_x10000 = 0;
  int32_t sw_sample[SAMPLE_COUNT];
  int32_t hw_sample[SAMPLE_COUNT];
  int32_t diff_sample[SAMPLE_COUNT];

  atik_bench_fill_bf16(
      mat_A,
      test_case->m,
      test_case->k,
      test_case->lda,
      &seed,
      test_case->min_value,
      test_case->max_value);
  atik_bench_fill_bf16(
      mat_B,
      test_case->k,
      test_case->n,
      test_case->ldb,
      &seed,
      test_case->min_value,
      test_case->max_value);
  atik_bench_zero_bf16(mat_C_sw, test_case->m, test_case->n, test_case->ldc);
  atik_bench_zero_bf16(mat_C_hw, test_case->m, test_case->n, test_case->ldc);

  const uint64_t cpu_start = atik_bench_read_cycles();
  atik_ref_matmul_bf16(
      mat_A,
      test_case->m,
      test_case->k,
      test_case->lda,
      mat_B,
      test_case->n,
      test_case->ldb,
      mat_C_sw,
      test_case->ldc);
  const uint64_t cpu_cycles = atik_bench_read_cycles() - cpu_start;

  atik_clear_counters();
  const uint64_t hw_start = atik_bench_read_cycles();
  int hw_rc = atik_matmul_bf16(
      mat_A,
      test_case->m,
      test_case->k,
      test_case->lda,
      mat_B,
      test_case->n,
      test_case->ldb,
      mat_C_hw,
      test_case->ldc);
  const uint64_t hw_e2e_cycles = atik_bench_read_cycles() - hw_start;
  asm volatile("fence rw, rw" ::: "memory");

  int mismatches = atik_bench_compare_bf16(
      mat_C_sw,
      mat_C_hw,
      test_case->m,
      test_case->n,
      test_case->ldc,
      test_case->ldc,
      test_case->tolerance_x10000,
      &max_diff_x10000);
  const char *status = (hw_rc == ATIK_OK && mismatches == 0) ? "PASS" : "FAIL";
  const int total = test_case->m * test_case->n;
  const int sample_count = total < SAMPLE_COUNT ? total : SAMPLE_COUNT;

  atik_bench_sample_bf16(
      mat_C_sw,
      mat_C_hw,
      test_case->m,
      test_case->n,
      test_case->ldc,
      test_case->ldc,
      0,
      sample_count,
      sw_sample,
      hw_sample,
      diff_sample);
  make_shape(shape, sizeof(shape), test_case);

  const atik_log_stage_t stages[] = {
      {"ref_matmul", cpu_cycles},
      {"atik_matmul_e2e", hw_e2e_cycles},
      {"atik_matmul", atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES)},
      {"compute", atik_read_counter(ATIK_COUNTER_COMPUTE_CYCLES)},
      {"dma_read", atik_read_counter(ATIK_COUNTER_DMA_READ_CYCLES)},
      {"dma_write", atik_read_counter(ATIK_COUNTER_DMA_WRITE_CYCLES)},
      {"bytes_read", atik_read_counter(ATIK_COUNTER_BYTES_READ)},
      {"bytes_written", atik_read_counter(ATIK_COUNTER_BYTES_WRITTEN)},
      {"tiles_loaded", atik_read_counter(ATIK_COUNTER_TILES_LOADED)},
      {"tiles_computed", atik_read_counter(ATIK_COUNTER_TILES_COMPUTED)},
      {"dma_stall", atik_read_counter(ATIK_COUNTER_DMA_STALL_CYCLES)},
      {"sram_stall", atik_read_counter(ATIK_COUNTER_SRAM_STALL_CYCLES)},
  };
  const atik_log_result_t result = {
      .workload = "matmul-benchmark",
      .case_index = case_index,
      .name = test_case->name,
      .shape = shape,
      .mode = "matmul",
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
      .workload = "matmul-benchmark",
      .case_index = case_index,
      .name = test_case->name,
      .tensor = "C",
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
