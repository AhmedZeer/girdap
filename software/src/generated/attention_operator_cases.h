#ifndef TOYROCC_GENERATED_ATTENTION_OPERATOR_CASES_H
#define TOYROCC_GENERATED_ATTENTION_OPERATOR_CASES_H

#include <stdint.h>

#define ATTN_OPERATOR_MAX_Q_ROWS 256
#define ATTN_OPERATOR_MAX_KV_ROWS 256
#define ATTN_OPERATOR_MAX_D_K 256
#define ATTN_OPERATOR_MAX_VALUE_COLS 128

typedef struct {
  const char *name;
  int q_rows;
  int kv_rows;
  int d_k;
  int value_cols;
  uint16_t scale_bf16;
  uint32_t tolerance_x100000;
  const uint16_t *q;
  const uint16_t *k;
  const uint16_t *v;
  const uint16_t *expected;
} attention_operator_case_t;

extern const attention_operator_case_t attention_operator_cases[];
extern const int attention_operator_case_count;

#endif
