#ifndef ATIK_ROCC_H
#define ATIK_ROCC_H

#include "atik_types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ATIK_CMD_SET_DESC       0
#define ATIK_CMD_RUN            1
#define ATIK_CMD_STATUS         2
#define ATIK_CMD_READ_COUNTER   3
#define ATIK_CMD_CLEAR_COUNTERS 4
#define ATIK_CMD_RESET          5

#ifndef ATIK_CUSTOM_OPCODE_VALUE
#define ATIK_CUSTOM_OPCODE_VALUE 0x0b
#endif

#define ATIK_ROCC_INSN(rd, rs1, rs2, funct) \
  asm volatile( \
      ".insn r %3, 7, %4, %0, %1, %2" \
      : "=r"(rd) \
      : "r"((uint64_t)(rs1)), "r"((uint64_t)(rs2)), \
        "i"(ATIK_CUSTOM_OPCODE_VALUE), "i"(funct))

static inline uint64_t atik_rocc_set_desc(const atik_desc_t *desc) {
  uint64_t rd;
  ATIK_ROCC_INSN(rd, (uint64_t)(uintptr_t)desc, 0, ATIK_CMD_SET_DESC);
  return rd;
}

static inline uint64_t atik_rocc_run(void) {
  uint64_t rd;
  ATIK_ROCC_INSN(rd, 0, 0, ATIK_CMD_RUN);
  return rd;
}

static inline uint64_t atik_rocc_status(void) {
  uint64_t rd;
  ATIK_ROCC_INSN(rd, 0, 0, ATIK_CMD_STATUS);
  return rd;
}

static inline uint64_t atik_rocc_read_counter(uint64_t index) {
  uint64_t rd;
  ATIK_ROCC_INSN(rd, index, 0, ATIK_CMD_READ_COUNTER);
  return rd;
}

static inline uint64_t atik_rocc_clear_counters(void) {
  uint64_t rd;
  ATIK_ROCC_INSN(rd, 0, 0, ATIK_CMD_CLEAR_COUNTERS);
  return rd;
}

static inline uint64_t atik_rocc_reset(void) {
  uint64_t rd;
  ATIK_ROCC_INSN(rd, 0, 0, ATIK_CMD_RESET);
  return rd;
}

#ifdef __cplusplus
}
#endif

#endif
