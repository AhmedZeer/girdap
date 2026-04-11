#include "bfloat16_utils.h"
#include "online_softmax.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_ROWS 2
#define MAX_COLS 512
#define INPUT_STRIDE 512
#define OUTPUT_STRIDE 512
#define SAMPLE_WIDTH 10

typedef struct {
  int rows;
  int cols;
} softmax_case_t;

static uint16_t input_matrix[MAX_ROWS][INPUT_STRIDE] __attribute__((aligned(64)));
static uint16_t output_matrix[MAX_ROWS][OUTPUT_STRIDE] __attribute__((aligned(64)));
static float sw_row[MAX_COLS];

static float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX) * 6.0f - 3.0f;
}

static void sw_softmax_reference_row(
    const uint16_t *input,
    int cols,
    float *output) {
  float max_val = -INFINITY;
  for (int c = 0; c < cols; c++) {
    const float value = bf16_to_float(input[c]);
    if (value > max_val) {
      max_val = value;
    }
  }

  double denom = 0.0;
  for (int c = 0; c < cols; c++) {
    const float value = bf16_to_float(input[c]);
    denom += exp((double)(value - max_val));
  }

  for (int c = 0; c < cols; c++) {
    const float value = bf16_to_float(input[c]);
    output[c] = (float)(exp((double)(value - max_val)) / denom);
  }
}

static void print_row_samples(
    int case_id,
    int row,
    const uint16_t *input,
    const uint16_t *hw_output,
    int cols) {
  sw_softmax_reference_row(input, cols, sw_row);
  printf("\n=== Case %d Row %d Samples ===\n", case_id, row);
  print_bf16_array_samples("Input BF16 Row", input, (size_t)cols, SAMPLE_WIDTH);
  print_float_array_samples("Software Softmax Row", sw_row, (size_t)cols, SAMPLE_WIDTH);
  print_bf16_array_samples("Hardware Softmax Row", hw_output, (size_t)cols, SAMPLE_WIDTH);
}

int main(void) {
  const softmax_case_t tests[] = {
      {1, 32}, // 32
      {1, 64}, // 64
      {1, 128}, // 128
      {1, 256}, // 256
      {1, 512}, // 512
  };
  const int ntests = (int)(sizeof(tests) / sizeof(tests[0]));

  srand(7);
  printf("=== Online Softmax Row Test (BF16 Output) ===\n");
  printf("CSV_HEADER,case,rows,cols,sw_cycles,hw_e2e_cycles,pass1_cycles,pass2_cycles,speedup_x100,mismatches\n");

  int total_mismatches = 0;

  for (int t = 0; t < ntests; t++) {
    const int rows = tests[t].rows;
    const int cols = tests[t].cols;

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < INPUT_STRIDE; c++) {
        input_matrix[r][c] = 0;
        output_matrix[r][c] = 0;
      }
      for (int c = 0; c < cols; c++) {
        input_matrix[r][c] = float_to_bf16(rand_float());
      }
    }

    const uint64_t sw_start = online_softmax_read_cycles();
    for (int r = 0; r < rows; r++) {
      sw_softmax_reference_row(&input_matrix[r][0], cols, sw_row);
    }
    const uint64_t sw_cycles = online_softmax_read_cycles() - sw_start;

    online_softmax_stats_t stats;
    const int rc = online_softmax_bf16_rows(
        &input_matrix[0][0],
        rows,
        cols,
        INPUT_STRIDE,
        &output_matrix[0][0],
        OUTPUT_STRIDE,
        &stats);

    if (rc != ONLINE_SOFTMAX_OK) {
      printf("FAIL: online_softmax_bf16_rows rc=%d on case %d\n", rc, t);
      return 1;
    }

    int mismatches = 0;
    for (int r = 0; r < rows; r++) {
      sw_softmax_reference_row(&input_matrix[r][0], cols, sw_row);
      for (int c = 0; c < cols; c++) {
        const float hw_value = bf16_to_float(output_matrix[r][c]);
        const float diff = fabsf(hw_value - sw_row[c]);
        if (diff > 0.08f) {
          if (mismatches < 8) {
            printf("Mismatch case=%d row=%d col=%d sw=", t, r, c);
            print_float_inline(sw_row[c]);
            printf("hw=");
            print_float_inline(hw_value);
            printf("diff=");
            print_float_inline(diff);
            printf("\n");
          }
          mismatches++;
        }
      }
    }

    total_mismatches += mismatches;
    const unsigned long speedup_x100 =
        stats.hw_e2e_cycles != 0
            ? (unsigned long)((100ULL * sw_cycles) / stats.hw_e2e_cycles)
            : 0UL;
    printf("CSV_DATA,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%d\n",
           t,
           rows,
           cols,
           (unsigned long)sw_cycles,
           (unsigned long)stats.hw_e2e_cycles,
           (unsigned long)stats.stage.pass1_cycles,
           (unsigned long)stats.stage.pass2_cycles,
           speedup_x100,
           mismatches);
    // print_row_samples(t, 0, &input_matrix[0][0], &output_matrix[0][0], cols);
  }

  if (total_mismatches == 0) {
    printf("PASS: all row-wise BF16 softmax cases matched the software reference.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }

  return total_mismatches == 0 ? 0 : 1;
}
