#ifndef ATIK_TYPES_H
#define ATIK_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ATIK_OP_MATMUL = 0,
  ATIK_OP_ATTENTION = 1,
  ATIK_OP_CAUSAL_ATTENTION = 2,
} atik_op_t;

typedef enum {
  ATIK_OK = 0,
  ATIK_ERR_BAD_OP = 1,
  ATIK_ERR_BAD_DIMS = 2,
  ATIK_ERR_BAD_ADDR = 3,
  ATIK_ERR_UNSUPPORTED_CONFIG = 4,
  ATIK_ERR_DMA = 5,
  ATIK_ERR_INTERNAL = 6,
  ATIK_ERR_TIMEOUT = 7,
} atik_status_t;

typedef struct {
  uint32_t op;
  uint32_t reserved0;

  uint64_t a_addr;
  uint64_t b_addr;
  uint64_t v_addr;
  uint64_t out_addr;

  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t kv_len;

  uint32_t lda;
  uint32_t ldb;
  uint32_t ldv;
  uint32_t ldout;

  uint16_t scale_bf16;
  uint16_t reserved1;
  uint32_t reserved2;
} atik_desc_t;

typedef enum {
  ATIK_COUNTER_TOTAL_CYCLES = 0,
  ATIK_COUNTER_COMPUTE_CYCLES = 1,
  ATIK_COUNTER_DMA_READ_CYCLES = 2,
  ATIK_COUNTER_DMA_WRITE_CYCLES = 3,
  ATIK_COUNTER_MESH_ACTIVE_CYCLES = 4,
  ATIK_COUNTER_MESH_IDLE_CYCLES = 5,
  ATIK_COUNTER_BYTES_READ = 6,
  ATIK_COUNTER_BYTES_WRITTEN = 7,
  ATIK_COUNTER_SOFTMAX_CYCLES = 8,
  ATIK_COUNTER_TILES_LOADED = 9,
  ATIK_COUNTER_TILES_COMPUTED = 10,
  ATIK_COUNTER_DMA_STALL_CYCLES = 11,
  ATIK_COUNTER_SRAM_STALL_CYCLES = 12,
  ATIK_COUNTER_COUNT = 13,
} atik_counter_t;

#define ATIK_STATUS_BUSY_MASK  0x1u
#define ATIK_STATUS_DONE_MASK  0x2u
#define ATIK_STATUS_ERROR_MASK 0x4u
#define ATIK_STATUS_ERROR_SHIFT 8u
#define ATIK_STATUS_ERROR_MASK_FIELD 0xff00u

#ifdef __cplusplus
}
#endif

#endif
