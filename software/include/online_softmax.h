#ifndef TOYROCC_ONLINE_SOFTMAX_H
#define TOYROCC_ONLINE_SOFTMAX_H

#include "rocc.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SOFTMAX_OPCODE 0
#define SOFTMAX_FUNCT_ACCUMULATE 0
#define SOFTMAX_FUNCT_NORMALIZE_BF16 1

typedef enum {
  ONLINE_SOFTMAX_OK = 0,
  ONLINE_SOFTMAX_ERR_BAD_DIMS = 1,
  ONLINE_SOFTMAX_ERR_NULL = 2,
} online_softmax_status_t;

typedef struct {
  uint64_t pass1_cycles;
  uint64_t pass2_cycles;
} online_softmax_stage_stats_t;

typedef struct {
  uint64_t hw_e2e_cycles;
  online_softmax_stage_stats_t stage;
} online_softmax_stats_t;

static inline uint64_t online_softmax_read_cycles(void) {
  uint64_t cycles;
  asm volatile("fence; rdcycle %0" : "=r"(cycles));
  return cycles;
}

static inline uint64_t online_softmax_accumulate_denominator(
    const uint16_t *input,
    uint64_t length) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SOFTMAX_OPCODE, rd, input, length, SOFTMAX_FUNCT_ACCUMULATE);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t online_softmax_write_bf16(uint16_t *output) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DS(SOFTMAX_OPCODE, rd, output, SOFTMAX_FUNCT_NORMALIZE_BF16);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

void online_softmax_clear_stats(online_softmax_stats_t *stats);

int online_softmax_bf16(
    const uint16_t *input,
    int length,
    uint16_t *output,
    online_softmax_stats_t *stats);

int online_softmax_bf16_rows(
    const uint16_t *input,
    int rows,
    int cols,
    int input_stride,
    uint16_t *output,
    int output_stride,
    online_softmax_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif
