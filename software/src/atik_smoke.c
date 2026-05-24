// SPDX-License-Identifier: Apache-2.0

#include "atik.h"
#include "atik_log.h"
#include "atik_ref.h"
#include "atik_types.h"

#include <stdint.h>

static int32_t fixed_x10000(uint16_t bf16) {
  float value = atik_bf16_to_float(bf16);
  return (int32_t)(value * 10000.0f + (value >= 0.0f ? 0.5f : -0.5f));
}

static int32_t abs_i32(int32_t value) {
  return value < 0 ? -value : value;
}

static uint64_t read_cpu_cycles(void) {
  uint64_t cycles;
  asm volatile("fence; rdcycle %0" : "=r"(cycles) :: "memory");
  return cycles;
}

static const uint16_t smoke_A[4] __attribute__((aligned(64))) = {
    0x3f80,
    0x4000,
    0x4040,
    0x4080,
};

static const uint16_t smoke_B[4] __attribute__((aligned(64))) = {
    0x3f00,
    0xbf80,
    0x3fc0,
    0x4000,
};

static uint16_t smoke_hw_C[32] __attribute__((aligned(64)));

int main(void) {
  uint16_t sw_C[4] = {0, 0, 0, 0};

  const uint64_t ref_start = read_cpu_cycles();
  atik_ref_matmul_bf16(smoke_A, 2, 2, 2, smoke_B, 2, 2, sw_C, 2);
  const uint64_t ref_cycles = read_cpu_cycles() - ref_start;

  int hw_rc = atik_matmul_bf16(smoke_A, 2, 2, 2, smoke_B, 2, 2, smoke_hw_C, 2);
  asm volatile("fence rw, rw" ::: "memory");

  int32_t sw[4];
  int32_t hw[4];
  int32_t diff[4];
  int32_t max_diff = 0;
  int mismatches = 0;
  for (int i = 0; i < 4; i++) {
    sw[i] = fixed_x10000(sw_C[i]);
    hw[i] = fixed_x10000(smoke_hw_C[i]);
    diff[i] = abs_i32(hw[i] - sw[i]);
    if (diff[i] > max_diff) {
      max_diff = diff[i];
    }
    if (diff[i] > 500) {
      mismatches++;
    }
  }

  const char *status = (hw_rc == ATIK_OK && mismatches == 0) ? "PASS" : "FAIL";
  const atik_log_stage_t stages[] = {
      {"ref_matmul", ref_cycles},
      {"atik_matmul", atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES)},
      {"bytes_read", atik_read_counter(ATIK_COUNTER_BYTES_READ)},
      {"bytes_written", atik_read_counter(ATIK_COUNTER_BYTES_WRITTEN)},
  };
  const atik_log_result_t result = {
      .workload = "atik-smoke",
      .case_index = 0,
      .name = "matmul_2x2_hw_smoke",
      .shape = "M=2 N=2 K=2",
      .mode = "matmul",
      .status = status,
      .hw_rc = hw_rc,
      .raw_hw_rc = hw_rc,
      .mismatches = mismatches,
      .cpu_cycles = ref_cycles,
      .hw_cycles = atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES),
      .stages = stages,
      .stage_count = 4,
      .hw_max_abs_diff_x100000 = max_diff * 10,
      .cpu_ref_max_abs_diff_x100000 = 0,
  };
  const atik_log_sample_t sample = {
      .workload = "atik-smoke",
      .case_index = 0,
      .name = "matmul_2x2_hw_smoke",
      .tensor = "C",
      .start = 0,
      .count = 4,
      .total = 4,
      .hw_rc = hw_rc,
      .sw_x10000 = sw,
      .hw_x10000 = hw,
      .diff_x10000 = diff,
  };

  atik_log_result_human(&result);
  atik_log_sample(&sample);
  atik_log_result_json(&result);
  return (hw_rc == ATIK_OK && mismatches == 0) ? 0 : 1;
}
