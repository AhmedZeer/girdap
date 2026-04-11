#include "bfloat16_utils.h"
#include "online_softmax.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef MAX_TEST_SIZE
#define MAX_TEST_SIZE 2048
#endif

#define SAMPLE_WIDTH 10

static float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
}

static void sw_softmax_reference(
    const uint16_t *input,
    uint64_t length,
    float *output) {
  float max_val = -INFINITY;
  for (uint64_t i = 0; i < length; i++) {
    const float value = bf16_to_float(input[i]);
    if (value > max_val) {
      max_val = value;
    }
  }

  double denom = 0.0;
  for (uint64_t i = 0; i < length; i++) {
    const float value = bf16_to_float(input[i]);
    denom += exp((double)(value - max_val));
  }

  for (uint64_t i = 0; i < length; i++) {
    const float value = bf16_to_float(input[i]);
    output[i] = (float)(exp((double)(value - max_val)) / denom);
  }
}

static uint16_t bf16_array[MAX_TEST_SIZE] __attribute__((aligned(64)));
static uint16_t out_array[MAX_TEST_SIZE] __attribute__((aligned(64)));
static float sw_softmax[MAX_TEST_SIZE];
static float hw_softmax[MAX_TEST_SIZE];

static void print_softmax_samples(
    const uint16_t *input,
    const float *sw_output,
    const uint16_t *hw_output,
    uint64_t length) {
  print_bf16_array_samples("Input BF16 Array", input, length, SAMPLE_WIDTH);
  print_float_array_samples("Software Softmax", sw_output, length, SAMPLE_WIDTH);
  print_bf16_array_samples("Hardware Softmax", hw_output, length, SAMPLE_WIDTH);
}

int main(void) {
  const uint64_t test_sizes[] = {15, 16, 31, 255, 256, 511, 512, 1023, 1024, 2047, 2048};
  const int num_tests = (int)(sizeof(test_sizes) / sizeof(test_sizes[0]));
  int test_failed = 0;

  srand(1);
  printf("CSV_HEADER,ArraySize,SW_Cycles,HW_E2E_Cycles,HW_Pass1_Cycles,HW_Pass2_Cycles,Speedup_X100,Mismatches\n");

  for (int t = 0; t < num_tests; t++) {
    const uint64_t current_size = test_sizes[t];
    if (current_size > MAX_TEST_SIZE) {
      printf("[!] MAX_TEST_SIZE too small for %lu\n", (unsigned long)current_size);
      return 1;
    }

    for (uint64_t i = 0; i < current_size; i++) {
      bf16_array[i] = float_to_bf16(rand_float());
      out_array[i] = 0;
    }

    const uint64_t sw_start = online_softmax_read_cycles();
    sw_softmax_reference(bf16_array, current_size, sw_softmax);
    const uint64_t sw_cycles = online_softmax_read_cycles() - sw_start;

    online_softmax_stats_t stats;
    const int rc = online_softmax_bf16(
        bf16_array,
        (int)current_size,
        out_array,
        &stats);

    if (rc != ONLINE_SOFTMAX_OK) {
      printf("FAIL: online_softmax_bf16 returned rc=%d for size %lu\n",
             rc,
             (unsigned long)current_size);
      return 1;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    for (uint64_t i = 0; i < current_size; i++) {
      const float hw_value = bf16_to_float(out_array[i]);
      const float diff = fabsf(hw_value - sw_softmax[i]);
      hw_softmax[i] = hw_value;
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
      }
      if (diff > 0.02f) {
        if (mismatches < 8) {
          printf("Mismatch size=%lu idx=%lu sw=", (unsigned long)current_size, (unsigned long)i);
          print_float_inline(sw_softmax[i]);
          printf("hw=");
          print_float_inline(hw_value);
          printf("diff=");
          print_float_inline(diff);
          printf("\n");
        }
        mismatches++;
      }
    }

    const unsigned long speedup_x100 =
        stats.hw_e2e_cycles != 0
            ? (unsigned long)((100ULL * sw_cycles) / stats.hw_e2e_cycles)
            : 0UL;

    printf("CSV_DATA,%lu,%lu,%lu,%lu,%lu,%lu,%d\n",
           (unsigned long)current_size,
           (unsigned long)sw_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           (unsigned long)stats.stage.pass1_cycles,
           (unsigned long)stats.stage.pass2_cycles,
           speedup_x100,
           mismatches);
    print_softmax_samples(bf16_array, sw_softmax, out_array, current_size);
    print_float("Max abs diff: ", max_abs_diff);

    if (mismatches != 0) {
      test_failed = 1;
      break;
    }
  }

  return test_failed ? 1 : 0;
}
