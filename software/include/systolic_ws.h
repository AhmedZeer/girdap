#ifndef TOYROCC_SYSTOLIC_WS_H
#define TOYROCC_SYSTOLIC_WS_H

#include "rocc.h"

#include <stdint.h>
#include <stdio.h>

#define SA_OPCODE 1
#define SA_FUNCT_CONFIG 0
#define SA_FUNCT_RUN 1
#define SA_FUNCT_STATUS 2
#define SA_FUNCT_PRELOAD_WEIGHTS 3
#define SA_FUNCT_READ_COUNTER 4
#define SA_FUNCT_CLEAR_COUNTERS 5
#define SA_FUNCT_RUN_PRELOADED 6
#define SA_FUNCT_SET_SCALE_BF16 7

enum {
  WS_PERF_BUSY_CYCLES = 0,
  WS_PERF_RUN_CMDS,
  WS_PERF_PRELOAD_CMDS,
  WS_PERF_PRELOAD_REUSE_HITS,
  WS_PERF_CHUNKS_STARTED,
  WS_PERF_FEED_CYCLES,
  WS_PERF_CAPTURE_ROWS,
  WS_PERF_TL_B_READS,
  WS_PERF_TL_A_READS,
  WS_PERF_TL_C_WRITES,
  WS_PERF_WAIT_FILL_B_CYCLES,
  WS_PERF_WAIT_FILL_A_CYCLES,
  WS_PERF_LOAD_WEIGHT_CYCLES,
  WS_PERF_WAIT_CHUNK_OUT_CYCLES,
  WS_PERF_WAIT_PUT_CYCLES,
  WS_PERF_COUNT
};

static const char *const ws_perf_counter_names[WS_PERF_COUNT] = {
    "busy_cycles",
    "run_cmds",
    "preload_cmds",
    "preload_reuse_hits",
    "chunks_started",
    "feed_cycles",
    "capture_rows",
    "tl_b_reads",
    "tl_a_reads",
    "tl_c_writes",
    "wait_fill_b_cycles",
    "wait_fill_a_cycles",
    "load_weight_cycles",
    "wait_chunk_out_cycles",
    "wait_put_cycles",
};

static inline uint64_t ws_read_cycles(void) {
  uint64_t cycles;
  asm volatile("fence; rdcycle %0" : "=r"(cycles));
  return cycles;
}

static inline uint64_t ws_pack4_u16(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3) {
  return ((uint64_t)v3 << 48) |
         ((uint64_t)v2 << 32) |
         ((uint64_t)v1 << 16) |
         (uint64_t)v0;
}

static inline uint64_t ws_sa_config(const uint64_t *a_stream, const uint64_t *b_stream) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, a_stream, b_stream, SA_FUNCT_CONFIG);
  return rd;
}

static inline uint64_t ws_sa_run(uint64_t *c_words, uint64_t k) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, c_words, k, SA_FUNCT_RUN);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_sa_run_preloaded(const uint64_t *a_stream, uint64_t *c_words) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, a_stream, c_words, SA_FUNCT_RUN_PRELOADED);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_sa_preload_weights(const uint64_t *b_stream, uint64_t k) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, b_stream, k, SA_FUNCT_PRELOAD_WEIGHTS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_sa_set_scale_bf16(uint16_t scale_bf16) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, (uint64_t)scale_bf16, 0, SA_FUNCT_SET_SCALE_BF16);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_sa_read_counter(uint64_t counter_id) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, counter_id, 0, SA_FUNCT_READ_COUNTER);
  return rd;
}

static inline void ws_sa_clear_counters(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, 0, 0, SA_FUNCT_CLEAR_COUNTERS);
  asm volatile("fence rw, rw" ::: "memory");
}

static inline void ws_read_perf_counters(uint64_t counters[WS_PERF_COUNT]) {
  for (int i = 0; i < WS_PERF_COUNT; i++) {
    counters[i] = ws_sa_read_counter((uint64_t)i);
  }
}

static inline void ws_print_perf_counters(const uint64_t counters[WS_PERF_COUNT]) {
  for (int i = 0; i < WS_PERF_COUNT; i++) {
    printf("PERF,%s,%lu\n", ws_perf_counter_names[i], (unsigned long)counters[i]);
  }
}

#endif
