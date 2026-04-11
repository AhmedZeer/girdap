#include "online_softmax.h"

#include <string.h>

static int validate_vector_dims(int length) {
  return length >= 0;
}

static int validate_row_dims(int rows, int cols, int input_stride, int output_stride) {
  if (rows < 0 || cols < 0) {
    return 0;
  }
  if (rows == 0 || cols == 0) {
    return 1;
  }
  if (input_stride < cols || output_stride < cols) {
    return 0;
  }
  return 1;
}

static int run_online_softmax_bf16(
    const uint16_t *input,
    int length,
    uint16_t *output,
    online_softmax_stats_t *stats) {
  const int measure = (stats != 0);

  uint64_t total_start = 0;
  uint64_t pass1_start = 0;
  uint64_t pass1_end = 0;
  uint64_t pass2_start = 0;
  uint64_t pass2_end = 0;

  if (measure) {
    total_start = online_softmax_read_cycles();
    pass1_start = total_start;
  }
  (void)online_softmax_accumulate_denominator(input, (uint64_t)length);
  if (measure) {
    pass1_end = online_softmax_read_cycles();
    stats->stage.pass1_cycles += pass1_end - pass1_start;
    pass2_start = pass1_end;
  }
  (void)online_softmax_write_bf16(output);
  if (measure) {
    pass2_end = online_softmax_read_cycles();
    stats->stage.pass2_cycles += pass2_end - pass2_start;
    stats->hw_e2e_cycles += pass2_end - total_start;
  }

  return ONLINE_SOFTMAX_OK;
}

void online_softmax_clear_stats(online_softmax_stats_t *stats) {
  if (stats != 0) {
    memset(stats, 0, sizeof(*stats));
  }
}

int online_softmax_bf16(
    const uint16_t *input,
    int length,
    uint16_t *output,
    online_softmax_stats_t *stats) {
  if (!validate_vector_dims(length)) {
    return ONLINE_SOFTMAX_ERR_BAD_DIMS;
  }
  if (length > 0 && (input == 0 || output == 0)) {
    return ONLINE_SOFTMAX_ERR_NULL;
  }

  online_softmax_clear_stats(stats);
  if (length == 0) {
    return ONLINE_SOFTMAX_OK;
  }
  return run_online_softmax_bf16(input, length, output, stats);
}

int online_softmax_bf16_rows(
    const uint16_t *input,
    int rows,
    int cols,
    int input_stride,
    uint16_t *output,
    int output_stride,
    online_softmax_stats_t *stats) {
  if (!validate_row_dims(rows, cols, input_stride, output_stride)) {
    return ONLINE_SOFTMAX_ERR_BAD_DIMS;
  }
  if (rows > 0 && cols > 0 && (input == 0 || output == 0)) {
    return ONLINE_SOFTMAX_ERR_NULL;
  }

  online_softmax_clear_stats(stats);
  if (rows == 0 || cols == 0) {
    return ONLINE_SOFTMAX_OK;
  }

  for (int r = 0; r < rows; r++) {
    const uint16_t *row_in = input + (r * input_stride);
    uint16_t *row_out = output + (r * output_stride);
    const int rc = run_online_softmax_bf16(row_in, cols, row_out, stats);
    if (rc != ONLINE_SOFTMAX_OK) {
      return rc;
    }
  }

  return ONLINE_SOFTMAX_OK;
}
