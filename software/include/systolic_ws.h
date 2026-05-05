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
#define SA_FUNCT_ATTN_SET_QK_ADDRS 8
#define SA_FUNCT_ATTN_SET_VOUT_ADDRS 9
#define SA_FUNCT_ATTN_SET_DIMS 10
#define SA_FUNCT_ATTN_SET_SCALE_BF16 11
#define SA_FUNCT_ATTN_RUN 12
#define SA_FUNCT_ATTN_PRECOMPUTE_SCORES 13
#define SA_FUNCT_ATTN_APPLY_CACHED 14
#define SA_FUNCT_ATTN_PACK_SET_ADDRS 15
#define SA_FUNCT_ATTN_PACK_SET_DIMS 16
#define SA_FUNCT_ATTN_PACK_RUN 17
#define SA_FUNCT_ATTN_DEBUG_READ_SCORE 18
#define SA_FUNCT_ATTN_DEBUG_READ_PROB 19

#define SA_ATTN_PACK_MODE_ROW_MAJOR_TILES 0
#define SA_ATTN_PACK_MODE_COLUMN_TILES 1

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

static inline uint64_t ws_attn_set_qk_addrs(
    const uint64_t *q_tiles,
    const uint64_t *k_tiles) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, q_tiles, k_tiles, SA_FUNCT_ATTN_SET_QK_ADDRS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_set_vout_addrs(
    const uint64_t *v_tiles,
    uint64_t *out_words) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, v_tiles, out_words, SA_FUNCT_ATTN_SET_VOUT_ADDRS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_set_dims(
    int q_rows,
    int kv_rows,
    int d_k,
    int value_cols) {
  const uint64_t packed_dims =
      (((uint64_t)(uint16_t)value_cols) << 48) |
      (((uint64_t)(uint16_t)d_k) << 32) |
      (((uint64_t)(uint16_t)kv_rows) << 16) |
      (uint64_t)(uint16_t)q_rows;
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, packed_dims, 0, SA_FUNCT_ATTN_SET_DIMS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_set_scale_bf16(uint16_t scale_bf16) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, (uint64_t)scale_bf16, 0, SA_FUNCT_ATTN_SET_SCALE_BF16);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_run(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, 0, 0, SA_FUNCT_ATTN_RUN);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_precompute_scores(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, 0, 0, SA_FUNCT_ATTN_PRECOMPUTE_SCORES);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_apply_cached(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, 0, 0, SA_FUNCT_ATTN_APPLY_CACHED);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_pack_set_addrs(
    const uint16_t *src,
    uint64_t *dst) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, src, dst, SA_FUNCT_ATTN_PACK_SET_ADDRS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_pack_set_dims(
    int rows,
    int cols,
    int ld,
    int out_stride,
    int mode) {
  const uint64_t packed_dims =
      (((uint64_t)(uint16_t)mode) << 48) |
      (((uint64_t)(uint16_t)ld) << 32) |
      (((uint64_t)(uint16_t)cols) << 16) |
      (uint64_t)(uint16_t)rows;
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(
      SA_OPCODE, rd, packed_dims, (uint64_t)(uint16_t)out_stride, SA_FUNCT_ATTN_PACK_SET_DIMS);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_pack_run(void) {
  uint64_t rd = 0;
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, 0, 0, SA_FUNCT_ATTN_PACK_RUN);
  asm volatile("fence rw, rw" ::: "memory");
  return rd;
}

static inline uint64_t ws_attn_debug_pack_index(int row, int col) {
  return (((uint64_t)(uint16_t)col) << 16) | (uint64_t)(uint16_t)row;
}

static inline int64_t ws_attn_debug_read_score(int row, int col) {
  uint64_t rd = 0;
  const uint64_t packed_index = ws_attn_debug_pack_index(row, col);
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, packed_index, 0, SA_FUNCT_ATTN_DEBUG_READ_SCORE);
  asm volatile("fence rw, rw" ::: "memory");
  return (int64_t)rd;
}

static inline uint64_t ws_attn_debug_read_prob(int row, int col) {
  uint64_t rd = 0;
  const uint64_t packed_index = ws_attn_debug_pack_index(row, col);
  asm volatile("fence rw, rw" ::: "memory");
  ROCC_INSTRUCTION_DSS(SA_OPCODE, rd, packed_index, 0, SA_FUNCT_ATTN_DEBUG_READ_PROB);
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
