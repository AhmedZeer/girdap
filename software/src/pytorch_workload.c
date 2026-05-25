// SPDX-License-Identifier: Apache-2.0

#include "atik.h"
#include "atik_bench.h"
#include "atik_log.h"
#include "atik_ref.h"
#include "atik_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(PYTORCH_WORKLOAD_ATTENTION_OPERATOR)
#include "generated/attention_operator_cases.h"
#elif defined(PYTORCH_WORKLOAD_GPT2_PREFILL)
#include "generated/gpt2_prefill_cases.h"
#elif defined(PYTORCH_WORKLOAD_TINY_BERT)
#include "generated/tiny_bert_cases.h"
#elif defined(PYTORCH_WORKLOAD_VIT)
#include "generated/vit_cases.h"
#endif

#define SAMPLE_COUNT 10


#ifndef ATIK_PROFILE_PROGRESS
#define ATIK_PROFILE_PROGRESS 0
#endif

#if ATIK_PROFILE_PROGRESS
#define ATIK_PROGRESS(...)        \
  do {                            \
    printf(__VA_ARGS__);          \
    printf("\n");                \
  } while (0)
#else
#define ATIK_PROGRESS(...) do { } while (0)
#endif

static int32_t abs_i32(int32_t value) {
  return value < 0 ? -value : value;
}

static void update_max_i32(int32_t *dst, int32_t value) {
  if (dst != 0 && value > *dst) *dst = value;
}

static uint64_t hybrid_cycles(uint64_t cpu_total, uint64_t cpu_replaced, uint64_t hw_replacement) {
  return cpu_total >= cpu_replaced ? (cpu_total - cpu_replaced + hw_replacement) : hw_replacement;
}

static int32_t bf16_x100000(uint16_t value) {
  float f = atik_bf16_to_float(value);
  return (int32_t)(f * 100000.0f + (f >= 0.0f ? 0.5f : -0.5f));
}

static int compare_bf16_x100000(
    const uint16_t *expected,
    const uint16_t *actual,
    int rows,
    int cols,
    int ld_expected,
    int ld_actual,
    int tolerance_x100000,
    int32_t *max_diff_x100000) {
  int mismatches = 0;
  int32_t max_diff = 0;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int32_t ref = bf16_x100000(expected[r * ld_expected + c]);
      int32_t got = bf16_x100000(actual[r * ld_actual + c]);
      int32_t diff = abs_i32(got - ref);
      if (diff > max_diff) max_diff = diff;
      if (diff > tolerance_x100000) mismatches++;
    }
  }
  if (max_diff_x100000 != 0) *max_diff_x100000 = max_diff;
  return mismatches;
}

static void sample_bf16_flat(
    const uint16_t *expected,
    const uint16_t *actual,
    int rows,
    int cols,
    int ld_expected,
    int ld_actual,
    int start,
    int count,
    int32_t *sw_x10000,
    int32_t *hw_x10000,
    int32_t *diff_x10000) {
  const int total = rows * cols;
  for (int i = 0; i < count; i++) {
    int flat = start + i;
    if (flat >= total) {
      sw_x10000[i] = 0;
      hw_x10000[i] = 0;
      diff_x10000[i] = 0;
      continue;
    }
    int row = flat / cols;
    int col = flat - row * cols;
    sw_x10000[i] = atik_bench_bf16_x10000(expected[row * ld_expected + col]);
    hw_x10000[i] = atik_bench_bf16_x10000(actual[row * ld_actual + col]);
    diff_x10000[i] = abs_i32(hw_x10000[i] - sw_x10000[i]);
  }
}

static void add_bias_bf16(uint16_t *x, int rows, int cols, int ld, const uint16_t *bias) {
  if (bias == 0) return;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float v = atik_bf16_to_float(x[r * ld + c]) + atik_bf16_to_float(bias[c]);
      x[r * ld + c] = atik_float_to_bf16(v);
    }
  }
}

static void add_matrix_bf16(
    const uint16_t *a,
    int lda,
    const uint16_t *b,
    int ldb,
    uint16_t *out,
    int ldout,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float v = atik_bf16_to_float(a[r * lda + c]) + atik_bf16_to_float(b[r * ldb + c]);
      out[r * ldout + c] = atik_float_to_bf16(v);
    }
  }
}

static void layer_norm_bf16(
    const uint16_t *x,
    int rows,
    int cols,
    int ldx,
    const uint16_t *gamma,
    const uint16_t *beta,
    uint16_t *out,
    int ldout) {
  for (int r = 0; r < rows; r++) {
    double mean = 0.0;
    for (int c = 0; c < cols; c++) mean += (double)atik_bf16_to_float(x[r * ldx + c]);
    mean /= (double)cols;
    double var = 0.0;
    for (int c = 0; c < cols; c++) {
      double d = (double)atik_bf16_to_float(x[r * ldx + c]) - mean;
      var += d * d;
    }
    double inv_std = 1.0 / sqrt((var / (double)cols) + 1.0e-5);
    for (int c = 0; c < cols; c++) {
      double y = ((double)atik_bf16_to_float(x[r * ldx + c]) - mean) * inv_std;
      y = y * (double)atik_bf16_to_float(gamma[c]) + (double)atik_bf16_to_float(beta[c]);
      out[r * ldout + c] = atik_float_to_bf16((float)y);
    }
  }
}

static void gelu_bf16(const uint16_t *x, uint16_t *out, int rows, int cols, int ldx, int ldout) {
  const float inv_sqrt2 = 0.7071067811865475f;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float v = atik_bf16_to_float(x[r * ldx + c]);
      out[r * ldout + c] = atik_float_to_bf16(0.5f * v * (1.0f + erff(v * inv_sqrt2)));
    }
  }
}

static void tanh_bf16(const uint16_t *x, uint16_t *out, int rows, int cols, int ldx, int ldout) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      out[r * ldout + c] = atik_float_to_bf16(tanhf(atik_bf16_to_float(x[r * ldx + c])));
    }
  }
}

static void linear_ref_bf16(
    const uint16_t *x,
    int rows,
    int in_cols,
    int ldx,
    const uint16_t *weight,
    int out_cols,
    const uint16_t *bias,
    uint16_t *out,
    int ldout) {
  atik_ref_matmul_bf16(x, rows, in_cols, ldx, weight, out_cols, out_cols, out, ldout);
  add_bias_bf16(out, rows, out_cols, ldout, bias);
}

static int linear_atik_bf16(
    const uint16_t *x,
    int rows,
    int in_cols,
    int ldx,
    const uint16_t *weight,
    int out_cols,
    const uint16_t *bias,
    uint16_t *out,
    int ldout,
    uint64_t *e2e_cycles) {
  const uint64_t start = atik_bench_read_cycles();
  int rc = atik_matmul_bf16(x, rows, in_cols, ldx, weight, out_cols, out_cols, out, ldout);
  asm volatile("fence rw, rw" ::: "memory");
  if (rc == ATIK_OK) add_bias_bf16(out, rows, out_cols, ldout, bias);
  if (e2e_cycles != 0) *e2e_cycles += atik_bench_read_cycles() - start;
  return rc;
}

static void account_cpu_cycles(uint64_t cycles, uint64_t *stage_cycles, uint64_t *cpu_total_cycles) {
  if (stage_cycles != 0) *stage_cycles += cycles;
  if (cpu_total_cycles != 0) *cpu_total_cycles += cycles;
}

static int profile_linear_replay_bf16(
    const char *op_name,
    const uint16_t *x,
    int rows,
    int in_cols,
    int ldx,
    const uint16_t *weight,
    int out_cols,
    const uint16_t *bias,
    uint16_t *cpu_out,
    int cpu_ldout,
    uint16_t *hw_out,
    int hw_ldout,
    int tolerance_x100000,
    uint64_t *cpu_stage_cycles,
    uint64_t *hw_stage_cycles,
    uint64_t *cpu_total_cycles,
    uint64_t *cpu_replaced_cycles,
    uint64_t *hw_replacement_cycles,
    int32_t *hw_max_diff_x100000,
    int *hw_mismatches,
    uint64_t *raw_hw_rc) {
  ATIK_PROGRESS("[cpu] begin linear %s rows=%d in=%d out=%d", op_name, rows, in_cols, out_cols);
  uint64_t start = atik_bench_read_cycles();
  linear_ref_bf16(x, rows, in_cols, ldx, weight, out_cols, bias, cpu_out, cpu_ldout);
  uint64_t cpu_cycles = atik_bench_read_cycles() - start;
  account_cpu_cycles(cpu_cycles, cpu_stage_cycles, cpu_total_cycles);
  if (cpu_replaced_cycles != 0) *cpu_replaced_cycles += cpu_cycles;
  ATIK_PROGRESS("[cpu] finished linear %s cycles=%llu", op_name, (unsigned long long)cpu_cycles);

  ATIK_PROGRESS("[hw] begin linear %s rows=%d in=%d out=%d", op_name, rows, in_cols, out_cols);
  uint64_t before_hw = hw_stage_cycles != 0 ? *hw_stage_cycles : 0;
  int rc = linear_atik_bf16(x, rows, in_cols, ldx, weight, out_cols, bias, hw_out, hw_ldout, hw_stage_cycles);
  uint64_t hw_cycles = hw_stage_cycles != 0 ? (*hw_stage_cycles - before_hw) : 0;
  if (hw_replacement_cycles != 0) *hw_replacement_cycles += hw_cycles;
  if (rc != ATIK_OK) {
    ATIK_PROGRESS("[hw] finished linear %s rc=%d cycles=%llu", op_name, rc, (unsigned long long)hw_cycles);
    if (raw_hw_rc != 0 && *raw_hw_rc == ATIK_OK) *raw_hw_rc = (uint64_t)rc;
    if (hw_mismatches != 0) *hw_mismatches += 1;
    return rc;
  }

  int32_t op_max_diff = 0;
  int op_mismatches = compare_bf16_x100000(cpu_out, hw_out, rows, out_cols, cpu_ldout, hw_ldout,
                                           tolerance_x100000, &op_max_diff);
  update_max_i32(hw_max_diff_x100000, op_max_diff);
  if (hw_mismatches != 0) *hw_mismatches += op_mismatches;
  ATIK_PROGRESS("[hw] finished linear %s rc=%d cycles=%llu mismatches=%d max_diff_x100000=%d",
                op_name, rc, (unsigned long long)hw_cycles, op_mismatches, op_max_diff);
  return rc;
}

static int profile_attention_replay_bf16(
    const char *op_name,
    const uint16_t *q,
    int q_rows,
    int d_k,
    int ldq,
    const uint16_t *k,
    int kv_rows,
    int ldk,
    const uint16_t *v,
    int value_cols,
    int ldv,
    uint16_t scale_bf16,
    uint16_t *cpu_out,
    int cpu_ldout,
    uint16_t *hw_out,
    int hw_ldout,
    int causal,
    int tolerance_x100000,
    uint64_t *cpu_stage_cycles,
    uint64_t *hw_stage_cycles,
    uint64_t *cpu_total_cycles,
    uint64_t *cpu_replaced_cycles,
    uint64_t *hw_replacement_cycles,
    int32_t *hw_max_diff_x100000,
    int *hw_mismatches,
    uint64_t *raw_hw_rc) {
  ATIK_PROGRESS("[cpu] begin attention %s q=%d kv=%d d=%d value=%d causal=%d",
                op_name, q_rows, kv_rows, d_k, value_cols, causal);
  uint64_t start = atik_bench_read_cycles();
  atik_ref_attention_bf16(q, q_rows, d_k, ldq, k, kv_rows, ldk, v, value_cols, ldv,
                          scale_bf16, cpu_out, cpu_ldout, causal);
  uint64_t cpu_cycles = atik_bench_read_cycles() - start;
  account_cpu_cycles(cpu_cycles, cpu_stage_cycles, cpu_total_cycles);
  if (cpu_replaced_cycles != 0) *cpu_replaced_cycles += cpu_cycles;
  ATIK_PROGRESS("[cpu] finished attention %s cycles=%llu", op_name, (unsigned long long)cpu_cycles);

  ATIK_PROGRESS("[hw] begin attention %s q=%d kv=%d d=%d value=%d causal=%d",
                op_name, q_rows, kv_rows, d_k, value_cols, causal);
  start = atik_bench_read_cycles();
  int rc = causal ? atik_causal_attention_bf16(q, q_rows, d_k, ldq, k, kv_rows, ldk, v, value_cols, ldv,
                                               scale_bf16, hw_out, hw_ldout)
                  : atik_attention_bf16(q, q_rows, d_k, ldq, k, kv_rows, ldk, v, value_cols, ldv,
                                        scale_bf16, hw_out, hw_ldout);
  asm volatile("fence rw, rw" ::: "memory");
  uint64_t hw_cycles = atik_bench_read_cycles() - start;
  if (hw_stage_cycles != 0) *hw_stage_cycles += hw_cycles;
  if (hw_replacement_cycles != 0) *hw_replacement_cycles += hw_cycles;
  if (rc != ATIK_OK) {
    ATIK_PROGRESS("[hw] finished attention %s rc=%d cycles=%llu", op_name, rc, (unsigned long long)hw_cycles);
    if (raw_hw_rc != 0 && *raw_hw_rc == ATIK_OK) *raw_hw_rc = (uint64_t)rc;
    if (hw_mismatches != 0) *hw_mismatches += 1;
    return rc;
  }

  int32_t op_max_diff = 0;
  int op_mismatches = compare_bf16_x100000(cpu_out, hw_out, q_rows, value_cols, cpu_ldout, hw_ldout,
                                           tolerance_x100000, &op_max_diff);
  update_max_i32(hw_max_diff_x100000, op_max_diff);
  if (hw_mismatches != 0) *hw_mismatches += op_mismatches;
  ATIK_PROGRESS("[hw] finished attention %s rc=%d cycles=%llu mismatches=%d max_diff_x100000=%d",
                op_name, rc, (unsigned long long)hw_cycles, op_mismatches, op_max_diff);
  return rc;
}

#if defined(PYTORCH_WORKLOAD_ATTENTION_OPERATOR)
static uint16_t attention_out[ATTN_OPERATOR_MAX_Q_ROWS * ATTN_OPERATOR_MAX_VALUE_COLS] __attribute__((aligned(64)));
static uint16_t attention_cpu_ref[ATTN_OPERATOR_MAX_Q_ROWS * ATTN_OPERATOR_MAX_VALUE_COLS] __attribute__((aligned(64)));

static uint64_t run_attention_case(int case_index, const attention_operator_case_t *test_case) {
  char shape[96];
  int32_t hw_max_diff = 0;
  int32_t cpu_max_diff = 0;
  int32_t sw_sample[SAMPLE_COUNT];
  int32_t hw_sample[SAMPLE_COUNT];
  int32_t diff_sample[SAMPLE_COUNT];
  memset(attention_out, 0, sizeof(attention_out));
  memset(attention_cpu_ref, 0, sizeof(attention_cpu_ref));
  const uint64_t cpu_start = atik_bench_read_cycles();
  atik_ref_attention_bf16(test_case->q, test_case->q_rows, test_case->d_k, test_case->d_k,
                          test_case->k, test_case->kv_rows, test_case->d_k,
                          test_case->v, test_case->value_cols, test_case->value_cols,
                          test_case->scale_bf16, attention_cpu_ref, test_case->value_cols, 0);
  const uint64_t cpu_cycles = atik_bench_read_cycles() - cpu_start;
  compare_bf16_x100000(test_case->expected, attention_cpu_ref, test_case->q_rows, test_case->value_cols,
                       test_case->value_cols, test_case->value_cols, (int)test_case->tolerance_x100000,
                       &cpu_max_diff);
  atik_clear_counters();
  const uint64_t hw_start = atik_bench_read_cycles();
  int hw_rc = atik_attention_bf16(test_case->q, test_case->q_rows, test_case->d_k, test_case->d_k,
                                  test_case->k, test_case->kv_rows, test_case->d_k,
                                  test_case->v, test_case->value_cols, test_case->value_cols,
                                  test_case->scale_bf16, attention_out, test_case->value_cols);
  const uint64_t hw_e2e_cycles = atik_bench_read_cycles() - hw_start;
  asm volatile("fence rw, rw" ::: "memory");
  int mismatches = 1;
  if (hw_rc == ATIK_OK) {
    mismatches = compare_bf16_x100000(test_case->expected, attention_out, test_case->q_rows,
                                      test_case->value_cols, test_case->value_cols, test_case->value_cols,
                                      (int)test_case->tolerance_x100000, &hw_max_diff);
  }
  snprintf(shape, sizeof(shape), "q=%d kv=%d d=%d value=%d", test_case->q_rows, test_case->kv_rows,
           test_case->d_k, test_case->value_cols);
  const int total = test_case->q_rows * test_case->value_cols;
  const int sample_count = total < SAMPLE_COUNT ? total : SAMPLE_COUNT;
  sample_bf16_flat(test_case->expected, attention_out, test_case->q_rows, test_case->value_cols,
                   test_case->value_cols, test_case->value_cols, 0, sample_count,
                   sw_sample, hw_sample, diff_sample);
  const atik_log_stage_t stages[] = {
      {"ref_attention", cpu_cycles}, {"atik_attention_e2e", hw_e2e_cycles},
      {"atik_attention", atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES)},
      {"compute", atik_read_counter(ATIK_COUNTER_COMPUTE_CYCLES)},
      {"dma_read", atik_read_counter(ATIK_COUNTER_DMA_READ_CYCLES)},
      {"dma_write", atik_read_counter(ATIK_COUNTER_DMA_WRITE_CYCLES)},
      {"bytes_read", atik_read_counter(ATIK_COUNTER_BYTES_READ)},
      {"bytes_written", atik_read_counter(ATIK_COUNTER_BYTES_WRITTEN)},
      {"softmax", atik_read_counter(ATIK_COUNTER_SOFTMAX_CYCLES)},
      {"tiles_loaded", atik_read_counter(ATIK_COUNTER_TILES_LOADED)},
      {"tiles_computed", atik_read_counter(ATIK_COUNTER_TILES_COMPUTED)},
  };
  const char *status = (hw_rc == ATIK_OK && mismatches == 0) ? "PASS" : "FAIL";
  const atik_log_result_t result = {
      .workload = "attention-operator", .case_index = case_index, .name = test_case->name,
      .shape = shape, .mode = "attention", .status = status, .hw_rc = (uint64_t)hw_rc,
      .raw_hw_rc = (uint64_t)hw_rc, .mismatches = (uint64_t)mismatches, .cpu_cycles = cpu_cycles,
      .hw_cycles = atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES), .stages = stages,
      .stage_count = (int)(sizeof(stages) / sizeof(stages[0])), .hw_max_abs_diff_x100000 = hw_max_diff,
      .cpu_ref_max_abs_diff_x100000 = cpu_max_diff,
  };
  const atik_log_sample_t sample = {
      .workload = "attention-operator", .case_index = case_index, .name = test_case->name,
      .tensor = "O", .start = 0, .count = sample_count, .total = total, .hw_rc = (uint64_t)hw_rc,
      .sw_x10000 = sw_sample, .hw_x10000 = hw_sample, .diff_x10000 = diff_sample,
  };
  atik_log_result_human(&result);
  atik_log_sample(&sample);
  atik_log_result_json(&result);
  return (hw_rc == ATIK_OK && mismatches == 0) ? 0u : 1u;
}

int main(void) {
  uint64_t failures = 0;
  for (int i = 0; i < attention_operator_case_count; i++) failures += run_attention_case(i, &attention_operator_cases[i]);
  return failures == 0 ? 0 : 1;
}

#include "generated/attention_operator_cases.c"

#elif defined(PYTORCH_WORKLOAD_GPT2_PREFILL)
typedef struct {
  uint64_t embedding_cycles;
  uint64_t ln_cycles;
  uint64_t qkv_proj_cycles;
  uint64_t attn_e2e_cycles;
  uint64_t out_proj_cycles;
  uint64_t mlp_fc1_cycles;
  uint64_t gelu_cycles;
  uint64_t mlp_fc2_cycles;
  uint64_t lm_head_cycles;
  uint64_t bias_cycles;
  uint64_t cpu_total_cycles;
  uint64_t cpu_replaced_cycles;
  uint64_t hw_replacement_cycles;
  uint64_t raw_hw_rc;
  int32_t hw_max_diff_x100000;
  int hw_mismatches;
} gpt2_stats_t;

#define GPT2_REPLAY_COLS   ((GPT2_PREFILL_MAX_VOCAB_SIZE > GPT2_PREFILL_MAX_HIDDEN_DIM)        ? ((GPT2_PREFILL_MAX_VOCAB_SIZE > GPT2_PREFILL_MAX_D_MODEL) ? GPT2_PREFILL_MAX_VOCAB_SIZE : GPT2_PREFILL_MAX_D_MODEL)        : ((GPT2_PREFILL_MAX_HIDDEN_DIM > GPT2_PREFILL_MAX_D_MODEL) ? GPT2_PREFILL_MAX_HIDDEN_DIM : GPT2_PREFILL_MAX_D_MODEL))

static uint16_t gpt_x[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_ln[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_q[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_k[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_v[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_context[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_attn_out[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_res[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_hidden[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t gpt_act[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t gpt_ffn_out[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t gpt_logits[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_VOCAB_SIZE] __attribute__((aligned(64)));
static uint16_t gpt_sw_logits[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_PREFILL_MAX_VOCAB_SIZE] __attribute__((aligned(64)));
static uint16_t gpt_hw_replay[GPT2_PREFILL_MAX_SEQ_LEN * GPT2_REPLAY_COLS] __attribute__((aligned(64)));

static const uint16_t *gpt_layer_vec(const uint16_t *base, int layer, int d_model) {
  return base + layer * d_model;
}

static const uint16_t *gpt_layer_hidden_vec(const uint16_t *base, int layer, int hidden_dim) {
  return base + layer * hidden_dim;
}

static const uint16_t *gpt_layer_w_dxd(const uint16_t *base, int layer, int d_model) {
  return base + layer * d_model * d_model;
}

static const uint16_t *gpt_layer_w_dxh(const uint16_t *base, int layer, int d_model, int hidden_dim) {
  return base + layer * d_model * hidden_dim;
}

static const uint16_t *gpt_layer_w_hxd(const uint16_t *base, int layer, int hidden_dim, int d_model) {
  return base + layer * hidden_dim * d_model;
}

static void gpt_embeddings(const gpt2_prefill_case_t *tc, uint16_t *out) {
  for (int r = 0; r < tc->seq_len; r++) {
    int token = tc->token_ids[r];
    for (int c = 0; c < tc->d_model; c++) {
      float v = atik_bf16_to_float(tc->token_embeddings[token * tc->d_model + c]) +
                atik_bf16_to_float(tc->position_embeddings[r * tc->d_model + c]);
      out[r * GPT2_PREFILL_MAX_D_MODEL + c] = atik_float_to_bf16(v);
    }
  }
}

static void gpt_attention_heads_ref(const gpt2_prefill_case_t *tc, uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *out) {
  for (int h = 0; h < tc->n_heads; h++) {
    int base = h * tc->head_dim;
    atik_ref_attention_bf16(
        q + base, tc->seq_len, tc->head_dim, GPT2_PREFILL_MAX_D_MODEL,
        k + base, tc->seq_len, GPT2_PREFILL_MAX_D_MODEL,
        v + base, tc->head_dim, GPT2_PREFILL_MAX_D_MODEL,
        tc->scale_bf16, out + base, GPT2_PREFILL_MAX_D_MODEL, 1);
  }
}

static void gpt_attention_heads_profiled(const gpt2_prefill_case_t *tc, gpt2_stats_t *stats) {
  for (int h = 0; h < tc->n_heads; h++) {
    int base = h * tc->head_dim;
    profile_attention_replay_bf16(
        "gpt.attn.head",
        gpt_q + base, tc->seq_len, tc->head_dim, GPT2_PREFILL_MAX_D_MODEL,
        gpt_k + base, tc->seq_len, GPT2_PREFILL_MAX_D_MODEL,
        gpt_v + base, tc->head_dim, GPT2_PREFILL_MAX_D_MODEL,
        tc->scale_bf16, gpt_context + base, GPT2_PREFILL_MAX_D_MODEL,
        gpt_hw_replay, tc->head_dim, 1, (int)tc->tolerance_x100000,
        0, &stats->attn_e2e_cycles, &stats->cpu_total_cycles,
        &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
        &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
  }
}

static void gpt_forward_ref(const gpt2_prefill_case_t *tc, uint16_t *logits_out) {
  gpt_embeddings(tc, gpt_x);
  for (int layer = 0; layer < tc->n_layers; layer++) {
    layer_norm_bf16(gpt_x, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_vec(tc->ln1_gamma, layer, tc->d_model),
                    gpt_layer_vec(tc->ln1_beta, layer, tc->d_model),
                    gpt_ln, GPT2_PREFILL_MAX_D_MODEL);
    linear_ref_bf16(gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model,
                    gpt_layer_vec(tc->bq, layer, tc->d_model), gpt_q, GPT2_PREFILL_MAX_D_MODEL);
    linear_ref_bf16(gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model,
                    gpt_layer_vec(tc->bk, layer, tc->d_model), gpt_k, GPT2_PREFILL_MAX_D_MODEL);
    linear_ref_bf16(gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model,
                    gpt_layer_vec(tc->bv, layer, tc->d_model), gpt_v, GPT2_PREFILL_MAX_D_MODEL);
    gpt_attention_heads_ref(tc, gpt_q, gpt_k, gpt_v, gpt_context);
    linear_ref_bf16(gpt_context, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model,
                    gpt_layer_vec(tc->bo, layer, tc->d_model), gpt_attn_out, GPT2_PREFILL_MAX_D_MODEL);
    add_matrix_bf16(gpt_x, GPT2_PREFILL_MAX_D_MODEL, gpt_attn_out, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_res, GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layer_norm_bf16(gpt_res, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_vec(tc->ln2_gamma, layer, tc->d_model),
                    gpt_layer_vec(tc->ln2_beta, layer, tc->d_model),
                    gpt_ln, GPT2_PREFILL_MAX_D_MODEL);
    linear_ref_bf16(gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim,
                    gpt_layer_hidden_vec(tc->b1, layer, tc->hidden_dim), gpt_hidden, GPT2_PREFILL_MAX_HIDDEN_DIM);
    gelu_bf16(gpt_hidden, gpt_act, tc->seq_len, tc->hidden_dim, GPT2_PREFILL_MAX_HIDDEN_DIM, GPT2_PREFILL_MAX_HIDDEN_DIM);
    linear_ref_bf16(gpt_act, tc->seq_len, tc->hidden_dim, GPT2_PREFILL_MAX_HIDDEN_DIM,
                    gpt_layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model,
                    gpt_layer_vec(tc->b2, layer, tc->d_model), gpt_ffn_out, GPT2_PREFILL_MAX_D_MODEL);
    add_matrix_bf16(gpt_res, GPT2_PREFILL_MAX_D_MODEL, gpt_ffn_out, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_x, GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  }
  layer_norm_bf16(gpt_x, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                  tc->final_ln_gamma, tc->final_ln_beta, gpt_ln, GPT2_PREFILL_MAX_D_MODEL);
  linear_ref_bf16(gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                  tc->lm_head, tc->vocab_size, 0, logits_out, GPT2_PREFILL_MAX_VOCAB_SIZE);
}

static void gpt_forward_profiled(const gpt2_prefill_case_t *tc, gpt2_stats_t *stats, uint16_t *logits_out) {
  memset(stats, 0, sizeof(*stats));
  stats->raw_hw_rc = ATIK_OK;
  atik_clear_counters();
  ATIK_PROGRESS("[case] begin gpt2 %s seq=%d d=%d layers=%d", tc->name, tc->seq_len, tc->d_model, tc->n_layers);

  uint64_t start = atik_bench_read_cycles();
  gpt_embeddings(tc, gpt_x);
  account_cpu_cycles(atik_bench_read_cycles() - start, &stats->embedding_cycles, &stats->cpu_total_cycles);
  ATIK_PROGRESS("[cpu] finished gpt.embedding");

  for (int layer = 0; layer < tc->n_layers; layer++) {
    ATIK_PROGRESS("[layer] begin gpt layer=%d", layer);
    start = atik_bench_read_cycles();
    layer_norm_bf16(gpt_x, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_vec(tc->ln1_gamma, layer, tc->d_model),
                    gpt_layer_vec(tc->ln1_beta, layer, tc->d_model), gpt_ln, GPT2_PREFILL_MAX_D_MODEL);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished gpt.ln1 layer=%d", layer);

    profile_linear_replay_bf16("gpt.q", gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model,
                               gpt_layer_vec(tc->bq, layer, tc->d_model), gpt_q, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_hw_replay, GPT2_PREFILL_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
    profile_linear_replay_bf16("gpt.k", gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model,
                               gpt_layer_vec(tc->bk, layer, tc->d_model), gpt_k, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_hw_replay, GPT2_PREFILL_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
    profile_linear_replay_bf16("gpt.v", gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model,
                               gpt_layer_vec(tc->bv, layer, tc->d_model), gpt_v, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_hw_replay, GPT2_PREFILL_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    gpt_attention_heads_profiled(tc, stats);

    profile_linear_replay_bf16("gpt.out", gpt_context, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model,
                               gpt_layer_vec(tc->bo, layer, tc->d_model), gpt_attn_out, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_hw_replay, GPT2_PREFILL_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->out_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    add_matrix_bf16(gpt_x, GPT2_PREFILL_MAX_D_MODEL, gpt_attn_out, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_res, GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->bias_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished gpt.attn_residual layer=%d", layer);

    start = atik_bench_read_cycles();
    layer_norm_bf16(gpt_res, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_layer_vec(tc->ln2_gamma, layer, tc->d_model),
                    gpt_layer_vec(tc->ln2_beta, layer, tc->d_model), gpt_ln, GPT2_PREFILL_MAX_D_MODEL);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished gpt.ln2 layer=%d", layer);

    profile_linear_replay_bf16("gpt.fc1", gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim,
                               gpt_layer_hidden_vec(tc->b1, layer, tc->hidden_dim), gpt_hidden, GPT2_PREFILL_MAX_HIDDEN_DIM,
                               gpt_hw_replay, GPT2_PREFILL_MAX_HIDDEN_DIM, (int)tc->tolerance_x100000,
                               0, &stats->mlp_fc1_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    gelu_bf16(gpt_hidden, gpt_act, tc->seq_len, tc->hidden_dim, GPT2_PREFILL_MAX_HIDDEN_DIM, GPT2_PREFILL_MAX_HIDDEN_DIM);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->gelu_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished gpt.gelu layer=%d", layer);

    profile_linear_replay_bf16("gpt.fc2", gpt_act, tc->seq_len, tc->hidden_dim, GPT2_PREFILL_MAX_HIDDEN_DIM,
                               gpt_layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model,
                               gpt_layer_vec(tc->b2, layer, tc->d_model), gpt_ffn_out, GPT2_PREFILL_MAX_D_MODEL,
                               gpt_hw_replay, GPT2_PREFILL_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->mlp_fc2_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    add_matrix_bf16(gpt_res, GPT2_PREFILL_MAX_D_MODEL, gpt_ffn_out, GPT2_PREFILL_MAX_D_MODEL,
                    gpt_x, GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->bias_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished gpt.ffn_residual layer=%d", layer);
    ATIK_PROGRESS("[layer] finished gpt layer=%d", layer);
  }

  start = atik_bench_read_cycles();
  layer_norm_bf16(gpt_x, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                  tc->final_ln_gamma, tc->final_ln_beta, gpt_ln, GPT2_PREFILL_MAX_D_MODEL);
  account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_cycles, &stats->cpu_total_cycles);
  ATIK_PROGRESS("[cpu] finished gpt.final_ln");

  profile_linear_replay_bf16("gpt.lm_head", gpt_ln, tc->seq_len, tc->d_model, GPT2_PREFILL_MAX_D_MODEL,
                             tc->lm_head, tc->vocab_size, 0, logits_out, GPT2_PREFILL_MAX_VOCAB_SIZE,
                             gpt_hw_replay, GPT2_PREFILL_MAX_VOCAB_SIZE, (int)tc->tolerance_x100000,
                             0, &stats->lm_head_cycles, &stats->cpu_total_cycles,
                             &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                             &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
  ATIK_PROGRESS("[case] finished gpt2 %s", tc->name);
}

static uint64_t gpt_stats_total(const gpt2_stats_t *s) {
  return s->embedding_cycles + s->ln_cycles + s->qkv_proj_cycles + s->attn_e2e_cycles +
         s->out_proj_cycles + s->mlp_fc1_cycles + s->gelu_cycles + s->mlp_fc2_cycles +
         s->lm_head_cycles + s->bias_cycles;
}

static uint64_t run_gpt2_case(int case_index, const gpt2_prefill_case_t *tc) {
  char shape[128];
  int32_t hw_max_diff = 0, cpu_max_diff = 0;
  int32_t sw_sample[SAMPLE_COUNT], hw_sample[SAMPLE_COUNT], diff_sample[SAMPLE_COUNT];
  snprintf(shape, sizeof(shape), "seq=%d d_model=%d heads=%d head_dim=%d hidden=%d layers=%d vocab=%d",
           tc->seq_len, tc->d_model, tc->n_heads, tc->head_dim, tc->hidden_dim, tc->n_layers, tc->vocab_size);

  gpt2_stats_t stats;
  gpt_forward_profiled(tc, &stats, gpt_sw_logits);
  uint64_t cpu_cycles = stats.cpu_total_cycles;
  int cpu_mismatches = compare_bf16_x100000(tc->expected_logits, gpt_sw_logits, tc->seq_len, tc->vocab_size,
                                            tc->vocab_size, GPT2_PREFILL_MAX_VOCAB_SIZE,
                                            (int)tc->tolerance_x100000, &cpu_max_diff);
  memcpy(gpt_logits, gpt_sw_logits, sizeof(uint16_t) * (size_t)tc->seq_len * (size_t)GPT2_PREFILL_MAX_VOCAB_SIZE);
  hw_max_diff = stats.hw_max_diff_x100000;
  int hw_rc = (int)stats.raw_hw_rc;
  int mismatches = cpu_mismatches + stats.hw_mismatches;
  int total = tc->seq_len * tc->vocab_size;
  int sample_count = total < SAMPLE_COUNT ? total : SAMPLE_COUNT;
  sample_bf16_flat(tc->expected_logits, gpt_logits, tc->seq_len, tc->vocab_size,
                   tc->vocab_size, GPT2_PREFILL_MAX_VOCAB_SIZE, 0, sample_count,
                   sw_sample, hw_sample, diff_sample);
  uint64_t hw_cycles = hybrid_cycles(stats.cpu_total_cycles, stats.cpu_replaced_cycles, stats.hw_replacement_cycles);
  const atik_log_stage_t stages[] = {
      {"embedding", stats.embedding_cycles}, {"ln", stats.ln_cycles},
      {"qkv", stats.qkv_proj_cycles}, {"attn", stats.attn_e2e_cycles},
      {"out", stats.out_proj_cycles}, {"fc1", stats.mlp_fc1_cycles},
      {"gelu", stats.gelu_cycles}, {"fc2", stats.mlp_fc2_cycles},
      {"lm_head", stats.lm_head_cycles}, {"bias", stats.bias_cycles},
      {"cpu_replaced", stats.cpu_replaced_cycles}, {"hw_replacement", stats.hw_replacement_cycles},
      {"atik_total", atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES)},
  };
  const char *status = (hw_rc == ATIK_OK && mismatches == 0) ? "PASS" : "FAIL";
  const atik_log_result_t result = {
      .workload = "gpt2-prefill", .case_index = case_index, .name = tc->name, .shape = shape,
      .mode = "attention+matmul", .status = status, .hw_rc = (uint64_t)hw_rc, .raw_hw_rc = stats.raw_hw_rc,
      .mismatches = (uint64_t)mismatches, .cpu_cycles = cpu_cycles, .hw_cycles = hw_cycles,
      .stages = stages, .stage_count = (int)(sizeof(stages) / sizeof(stages[0])),
      .hw_max_abs_diff_x100000 = hw_max_diff, .cpu_ref_max_abs_diff_x100000 = cpu_max_diff,
  };
  const atik_log_sample_t sample = {
      .workload = "gpt2-prefill", .case_index = case_index, .name = tc->name, .tensor = "logits",
      .start = 0, .count = sample_count, .total = total, .hw_rc = (uint64_t)hw_rc,
      .sw_x10000 = sw_sample, .hw_x10000 = hw_sample, .diff_x10000 = diff_sample,
  };
  atik_log_result_human(&result);
  atik_log_sample(&sample);
  atik_log_result_json(&result);
  return (hw_rc == ATIK_OK && mismatches == 0) ? 0u : 1u;
}

int main(void) {
  uint64_t failures = 0;
  for (int i = 0; i < gpt2_prefill_case_count; i++) failures += run_gpt2_case(i, &gpt2_prefill_cases[i]);
  return failures == 0 ? 0 : 1;
}

#include "generated/gpt2_prefill_cases.c"

#elif defined(PYTORCH_WORKLOAD_TINY_BERT)
typedef struct {
  uint64_t embedding_cycles;
  uint64_t qkv_proj_cycles;
  uint64_t attn_e2e_cycles;
  uint64_t out_proj_cycles;
  uint64_t mlp_fc1_cycles;
  uint64_t gelu_cycles;
  uint64_t mlp_fc2_cycles;
  uint64_t pooler_cycles;
  uint64_t classifier_cycles;
  uint64_t ln_residual_cycles;
  uint64_t tanh_cycles;
  uint64_t bias_cycles;
  uint64_t cpu_total_cycles;
  uint64_t cpu_replaced_cycles;
  uint64_t hw_replacement_cycles;
  uint64_t raw_hw_rc;
  int32_t hw_max_diff_x100000;
  int hw_mismatches;
} tiny_stats_t;

#define TINY_BERT_REPLAY_COLS   ((TINY_BERT_MAX_HIDDEN_DIM > TINY_BERT_MAX_D_MODEL) ? TINY_BERT_MAX_HIDDEN_DIM : TINY_BERT_MAX_D_MODEL)

static uint16_t tb_x[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_q[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_k[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_v[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_context[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_attn_out[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_res[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_hidden[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t tb_act[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t tb_ffn_out[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_pool[TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t tb_logits[TINY_BERT_MAX_NUM_CLASSES] __attribute__((aligned(64)));
static uint16_t tb_sw_logits[TINY_BERT_MAX_NUM_CLASSES] __attribute__((aligned(64)));
static uint16_t tb_hw_replay[TINY_BERT_MAX_SEQ_LEN * TINY_BERT_REPLAY_COLS] __attribute__((aligned(64)));

static const uint16_t *tb_layer_vec(const uint16_t *base, int layer, int d_model) { return base + layer * d_model; }
static const uint16_t *tb_layer_hidden_vec(const uint16_t *base, int layer, int hidden_dim) { return base + layer * hidden_dim; }
static const uint16_t *tb_layer_w_dxd(const uint16_t *base, int layer, int d_model) { return base + layer * d_model * d_model; }
static const uint16_t *tb_layer_w_dxh(const uint16_t *base, int layer, int d_model, int hidden_dim) { return base + layer * d_model * hidden_dim; }
static const uint16_t *tb_layer_w_hxd(const uint16_t *base, int layer, int hidden_dim, int d_model) { return base + layer * hidden_dim * d_model; }

static void tb_embeddings(const tiny_bert_case_t *tc, uint16_t *out) {
  for (int r = 0; r < tc->seq_len; r++) {
    int token = tc->token_ids[r];
    int type_id = tc->type_ids[r];
    for (int c = 0; c < tc->d_model; c++) {
      float v = atik_bf16_to_float(tc->word_embeddings[token * tc->d_model + c]) +
                atik_bf16_to_float(tc->position_embeddings[r * tc->d_model + c]) +
                atik_bf16_to_float(tc->token_type_embeddings[type_id * tc->d_model + c]);
      tb_res[r * TINY_BERT_MAX_D_MODEL + c] = atik_float_to_bf16(v);
    }
  }
  layer_norm_bf16(tb_res, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                  tc->emb_ln_gamma, tc->emb_ln_beta, out, TINY_BERT_MAX_D_MODEL);
}

static void tb_attention_heads_ref(const tiny_bert_case_t *tc, uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *out) {
  for (int h = 0; h < tc->n_heads; h++) {
    int base = h * tc->head_dim;
    atik_ref_attention_bf16(q + base, tc->seq_len, tc->head_dim, TINY_BERT_MAX_D_MODEL,
                            k + base, tc->seq_len, TINY_BERT_MAX_D_MODEL,
                            v + base, tc->head_dim, TINY_BERT_MAX_D_MODEL,
                            tc->scale_bf16, out + base, TINY_BERT_MAX_D_MODEL, 0);
  }
}

static void tb_attention_heads_profiled(const tiny_bert_case_t *tc, tiny_stats_t *stats) {
  for (int i = 0; i < tc->seq_len; i++) {
    if (tc->attention_mask[i] == 0) {
      if (stats->raw_hw_rc == ATIK_OK) stats->raw_hw_rc = ATIK_ERR_UNSUPPORTED_CONFIG;
      stats->hw_mismatches += 1;
      return;
    }
  }
  for (int h = 0; h < tc->n_heads; h++) {
    int base = h * tc->head_dim;
    profile_attention_replay_bf16(
        "tiny.attn.head",
        tb_q + base, tc->seq_len, tc->head_dim, TINY_BERT_MAX_D_MODEL,
        tb_k + base, tc->seq_len, TINY_BERT_MAX_D_MODEL,
        tb_v + base, tc->head_dim, TINY_BERT_MAX_D_MODEL,
        tc->scale_bf16, tb_context + base, TINY_BERT_MAX_D_MODEL,
        tb_hw_replay, tc->head_dim, 0, (int)tc->tolerance_x100000,
        0, &stats->attn_e2e_cycles, &stats->cpu_total_cycles,
        &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
        &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
  }
}

static void tb_forward_ref(const tiny_bert_case_t *tc, uint16_t *logits_out) {
  tb_embeddings(tc, tb_x);
  for (int layer = 0; layer < tc->n_layers; layer++) {
    linear_ref_bf16(tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL, tb_layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model, tb_layer_vec(tc->bq, layer, tc->d_model), tb_q, TINY_BERT_MAX_D_MODEL);
    linear_ref_bf16(tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL, tb_layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model, tb_layer_vec(tc->bk, layer, tc->d_model), tb_k, TINY_BERT_MAX_D_MODEL);
    linear_ref_bf16(tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL, tb_layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model, tb_layer_vec(tc->bv, layer, tc->d_model), tb_v, TINY_BERT_MAX_D_MODEL);
    tb_attention_heads_ref(tc, tb_q, tb_k, tb_v, tb_context);
    linear_ref_bf16(tb_context, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL, tb_layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model, tb_layer_vec(tc->bo, layer, tc->d_model), tb_attn_out, TINY_BERT_MAX_D_MODEL);
    add_matrix_bf16(tb_x, TINY_BERT_MAX_D_MODEL, tb_attn_out, TINY_BERT_MAX_D_MODEL, tb_res, TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layer_norm_bf16(tb_res, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL, tb_layer_vec(tc->attn_ln_gamma, layer, tc->d_model), tb_layer_vec(tc->attn_ln_beta, layer, tc->d_model), tb_x, TINY_BERT_MAX_D_MODEL);
    linear_ref_bf16(tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL, tb_layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim, tb_layer_hidden_vec(tc->b1, layer, tc->hidden_dim), tb_hidden, TINY_BERT_MAX_HIDDEN_DIM);
    gelu_bf16(tb_hidden, tb_act, tc->seq_len, tc->hidden_dim, TINY_BERT_MAX_HIDDEN_DIM, TINY_BERT_MAX_HIDDEN_DIM);
    linear_ref_bf16(tb_act, tc->seq_len, tc->hidden_dim, TINY_BERT_MAX_HIDDEN_DIM, tb_layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model, tb_layer_vec(tc->b2, layer, tc->d_model), tb_ffn_out, TINY_BERT_MAX_D_MODEL);
    add_matrix_bf16(tb_x, TINY_BERT_MAX_D_MODEL, tb_ffn_out, TINY_BERT_MAX_D_MODEL, tb_res, TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layer_norm_bf16(tb_res, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL, tb_layer_vec(tc->ffn_ln_gamma, layer, tc->d_model), tb_layer_vec(tc->ffn_ln_beta, layer, tc->d_model), tb_x, TINY_BERT_MAX_D_MODEL);
  }
  linear_ref_bf16(tb_x, 1, tc->d_model, TINY_BERT_MAX_D_MODEL, tc->pool_w, tc->d_model, tc->pool_b, tb_pool, tc->d_model);
  tanh_bf16(tb_pool, tb_pool, 1, tc->d_model, tc->d_model, tc->d_model);
  linear_ref_bf16(tb_pool, 1, tc->d_model, tc->d_model, tc->classifier_w, tc->num_classes, tc->classifier_b, logits_out, tc->num_classes);
}

static void tb_forward_profiled(const tiny_bert_case_t *tc, tiny_stats_t *stats, uint16_t *logits_out) {
  memset(stats, 0, sizeof(*stats));
  stats->raw_hw_rc = ATIK_OK;
  atik_clear_counters();
  ATIK_PROGRESS("[case] begin tiny-bert %s seq=%d d=%d layers=%d", tc->name, tc->seq_len, tc->d_model, tc->n_layers);

  uint64_t start = atik_bench_read_cycles();
  tb_embeddings(tc, tb_x);
  account_cpu_cycles(atik_bench_read_cycles() - start, &stats->embedding_cycles, &stats->cpu_total_cycles);
  ATIK_PROGRESS("[cpu] finished tiny.embedding");

  for (int layer = 0; layer < tc->n_layers; layer++) {
    ATIK_PROGRESS("[layer] begin tiny layer=%d", layer);
    profile_linear_replay_bf16("tiny.q", tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                               tb_layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model,
                               tb_layer_vec(tc->bq, layer, tc->d_model), tb_q, TINY_BERT_MAX_D_MODEL,
                               tb_hw_replay, TINY_BERT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
    profile_linear_replay_bf16("tiny.k", tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                               tb_layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model,
                               tb_layer_vec(tc->bk, layer, tc->d_model), tb_k, TINY_BERT_MAX_D_MODEL,
                               tb_hw_replay, TINY_BERT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
    profile_linear_replay_bf16("tiny.v", tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                               tb_layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model,
                               tb_layer_vec(tc->bv, layer, tc->d_model), tb_v, TINY_BERT_MAX_D_MODEL,
                               tb_hw_replay, TINY_BERT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    tb_attention_heads_profiled(tc, stats);

    profile_linear_replay_bf16("tiny.out", tb_context, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                               tb_layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model,
                               tb_layer_vec(tc->bo, layer, tc->d_model), tb_attn_out, TINY_BERT_MAX_D_MODEL,
                               tb_hw_replay, TINY_BERT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->out_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    add_matrix_bf16(tb_x, TINY_BERT_MAX_D_MODEL, tb_attn_out, TINY_BERT_MAX_D_MODEL, tb_res,
                    TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layer_norm_bf16(tb_res, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                    tb_layer_vec(tc->attn_ln_gamma, layer, tc->d_model),
                    tb_layer_vec(tc->attn_ln_beta, layer, tc->d_model), tb_x, TINY_BERT_MAX_D_MODEL);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_residual_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished tiny.attn_residual_ln layer=%d", layer);

    profile_linear_replay_bf16("tiny.fc1", tb_x, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                               tb_layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim,
                               tb_layer_hidden_vec(tc->b1, layer, tc->hidden_dim), tb_hidden, TINY_BERT_MAX_HIDDEN_DIM,
                               tb_hw_replay, TINY_BERT_MAX_HIDDEN_DIM, (int)tc->tolerance_x100000,
                               0, &stats->mlp_fc1_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    gelu_bf16(tb_hidden, tb_act, tc->seq_len, tc->hidden_dim, TINY_BERT_MAX_HIDDEN_DIM, TINY_BERT_MAX_HIDDEN_DIM);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->gelu_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished tiny.gelu layer=%d", layer);

    profile_linear_replay_bf16("tiny.fc2", tb_act, tc->seq_len, tc->hidden_dim, TINY_BERT_MAX_HIDDEN_DIM,
                               tb_layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model,
                               tb_layer_vec(tc->b2, layer, tc->d_model), tb_ffn_out, TINY_BERT_MAX_D_MODEL,
                               tb_hw_replay, TINY_BERT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->mlp_fc2_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    add_matrix_bf16(tb_x, TINY_BERT_MAX_D_MODEL, tb_ffn_out, TINY_BERT_MAX_D_MODEL, tb_res,
                    TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layer_norm_bf16(tb_res, tc->seq_len, tc->d_model, TINY_BERT_MAX_D_MODEL,
                    tb_layer_vec(tc->ffn_ln_gamma, layer, tc->d_model),
                    tb_layer_vec(tc->ffn_ln_beta, layer, tc->d_model), tb_x, TINY_BERT_MAX_D_MODEL);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_residual_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished tiny.ffn_residual_ln layer=%d", layer);
    ATIK_PROGRESS("[layer] finished tiny layer=%d", layer);
  }

  profile_linear_replay_bf16("tiny.pooler", tb_x, 1, tc->d_model, TINY_BERT_MAX_D_MODEL,
                             tc->pool_w, tc->d_model, tc->pool_b, tb_pool, tc->d_model,
                             tb_hw_replay, tc->d_model, (int)tc->tolerance_x100000,
                             0, &stats->pooler_cycles, &stats->cpu_total_cycles,
                             &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                             &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

  start = atik_bench_read_cycles();
  tanh_bf16(tb_pool, tb_pool, 1, tc->d_model, tc->d_model, tc->d_model);
  account_cpu_cycles(atik_bench_read_cycles() - start, &stats->tanh_cycles, &stats->cpu_total_cycles);
  ATIK_PROGRESS("[cpu] finished tiny.tanh");

  profile_linear_replay_bf16("tiny.classifier", tb_pool, 1, tc->d_model, tc->d_model,
                             tc->classifier_w, tc->num_classes, tc->classifier_b, logits_out, tc->num_classes,
                             tb_hw_replay, tc->num_classes, (int)tc->tolerance_x100000,
                             0, &stats->classifier_cycles, &stats->cpu_total_cycles,
                             &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                             &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
  ATIK_PROGRESS("[case] finished tiny-bert %s", tc->name);
}

static uint64_t tb_stats_total(const tiny_stats_t *s) {
  return s->embedding_cycles + s->qkv_proj_cycles + s->attn_e2e_cycles + s->out_proj_cycles +
         s->mlp_fc1_cycles + s->gelu_cycles + s->mlp_fc2_cycles + s->pooler_cycles +
         s->classifier_cycles + s->ln_residual_cycles + s->tanh_cycles + s->bias_cycles;
}

static uint64_t run_tiny_bert_case(int case_index, const tiny_bert_case_t *tc) {
  char shape[128];
  int32_t hw_max_diff = 0, cpu_max_diff = 0;
  int32_t sw_sample[SAMPLE_COUNT], hw_sample[SAMPLE_COUNT], diff_sample[SAMPLE_COUNT];
  snprintf(shape, sizeof(shape), "seq=%d d_model=%d heads=%d head_dim=%d hidden=%d layers=%d vocab=%d classes=%d",
           tc->seq_len, tc->d_model, tc->n_heads, tc->head_dim, tc->hidden_dim, tc->n_layers, tc->vocab_size, tc->num_classes);
  tiny_stats_t stats;
  tb_forward_profiled(tc, &stats, tb_sw_logits);
  uint64_t cpu_cycles = stats.cpu_total_cycles;
  int cpu_mismatches = compare_bf16_x100000(tc->expected_logits, tb_sw_logits, 1, tc->num_classes,
                                            tc->num_classes, tc->num_classes,
                                            (int)tc->tolerance_x100000, &cpu_max_diff);
  memcpy(tb_logits, tb_sw_logits, sizeof(uint16_t) * (size_t)tc->num_classes);
  hw_max_diff = stats.hw_max_diff_x100000;
  int hw_rc = (int)stats.raw_hw_rc;
  int mismatches = cpu_mismatches + stats.hw_mismatches;
  int sample_count = tc->num_classes < SAMPLE_COUNT ? tc->num_classes : SAMPLE_COUNT;
  sample_bf16_flat(tc->expected_logits, tb_logits, 1, tc->num_classes, tc->num_classes, tc->num_classes, 0, sample_count, sw_sample, hw_sample, diff_sample);
  uint64_t hw_cycles = hybrid_cycles(stats.cpu_total_cycles, stats.cpu_replaced_cycles, stats.hw_replacement_cycles);
  const atik_log_stage_t stages[] = {
      {"embedding", stats.embedding_cycles}, {"qkv", stats.qkv_proj_cycles},
      {"attn", stats.attn_e2e_cycles}, {"out", stats.out_proj_cycles},
      {"fc1", stats.mlp_fc1_cycles}, {"gelu", stats.gelu_cycles},
      {"fc2", stats.mlp_fc2_cycles}, {"pooler", stats.pooler_cycles},
      {"classifier", stats.classifier_cycles}, {"ln_residual", stats.ln_residual_cycles},
      {"tanh", stats.tanh_cycles}, {"cpu_replaced", stats.cpu_replaced_cycles},
      {"hw_replacement", stats.hw_replacement_cycles}, {"atik_total", atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES)},
  };
  const char *status = (hw_rc == ATIK_OK && mismatches == 0) ? "PASS" : "FAIL";
  const atik_log_result_t result = {
      .workload = "tiny-bert", .case_index = case_index, .name = tc->name, .shape = shape,
      .mode = "attention+matmul", .status = status, .hw_rc = (uint64_t)hw_rc, .raw_hw_rc = stats.raw_hw_rc,
      .mismatches = (uint64_t)mismatches, .cpu_cycles = cpu_cycles, .hw_cycles = hw_cycles,
      .stages = stages, .stage_count = (int)(sizeof(stages) / sizeof(stages[0])),
      .hw_max_abs_diff_x100000 = hw_max_diff, .cpu_ref_max_abs_diff_x100000 = cpu_max_diff,
  };
  const atik_log_sample_t sample = {
      .workload = "tiny-bert", .case_index = case_index, .name = tc->name, .tensor = "logits",
      .start = 0, .count = sample_count, .total = tc->num_classes, .hw_rc = (uint64_t)hw_rc,
      .sw_x10000 = sw_sample, .hw_x10000 = hw_sample, .diff_x10000 = diff_sample,
  };
  atik_log_result_human(&result);
  atik_log_sample(&sample);
  atik_log_result_json(&result);
  return (hw_rc == ATIK_OK && mismatches == 0) ? 0u : 1u;
}

int main(void) {
  uint64_t failures = 0;
  for (int i = 0; i < tiny_bert_case_count; i++) failures += run_tiny_bert_case(i, &tiny_bert_cases[i]);
  return failures == 0 ? 0 : 1;
}

#include "generated/tiny_bert_cases.c"

#elif defined(PYTORCH_WORKLOAD_VIT)
typedef struct {
  uint64_t patch_embed_cycles;
  uint64_t ln_cycles;
  uint64_t qkv_proj_cycles;
  uint64_t attn_e2e_cycles;
  uint64_t out_proj_cycles;
  uint64_t mlp_fc1_cycles;
  uint64_t gelu_cycles;
  uint64_t mlp_fc2_cycles;
  uint64_t classifier_cycles;
  uint64_t residual_cycles;
  uint64_t cpu_total_cycles;
  uint64_t cpu_replaced_cycles;
  uint64_t hw_replacement_cycles;
  uint64_t raw_hw_rc;
  int32_t hw_max_diff_x100000;
  int hw_mismatches;
} vit_stats_t;

#define VIT_REPLAY_COLS   ((VIT_MAX_HIDDEN_DIM > VIT_MAX_D_MODEL) ? VIT_MAX_HIDDEN_DIM : VIT_MAX_D_MODEL)

static uint16_t vit_patch_tokens[VIT_MAX_PATCH_COUNT * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_x[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_ln[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_q[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_k[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_v[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_context[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_attn_out[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_hidden[VIT_MAX_SEQ_LEN * VIT_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t vit_act[VIT_MAX_SEQ_LEN * VIT_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t vit_ffn_out[VIT_MAX_SEQ_LEN * VIT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t vit_logits[VIT_MAX_NUM_CLASSES] __attribute__((aligned(64)));
static uint16_t vit_sw_logits[VIT_MAX_NUM_CLASSES] __attribute__((aligned(64)));
static uint16_t vit_hw_replay[VIT_MAX_SEQ_LEN * VIT_REPLAY_COLS] __attribute__((aligned(64)));

static const uint16_t *vit_layer_vec(const uint16_t *base, int layer, int d_model) { return base + layer * d_model; }
static const uint16_t *vit_layer_hidden_vec(const uint16_t *base, int layer, int hidden_dim) { return base + layer * hidden_dim; }
static const uint16_t *vit_layer_w_dxd(const uint16_t *base, int layer, int d_model) { return base + layer * d_model * d_model; }
static const uint16_t *vit_layer_w_dxh(const uint16_t *base, int layer, int d_model, int hidden_dim) { return base + layer * d_model * hidden_dim; }
static const uint16_t *vit_layer_w_hxd(const uint16_t *base, int layer, int hidden_dim, int d_model) { return base + layer * hidden_dim * d_model; }

static void vit_add_position(const vit_case_t *tc) {
  for (int c = 0; c < tc->d_model; c++) {
    float v = atik_bf16_to_float(tc->cls_token[c]) + atik_bf16_to_float(tc->position_embeddings[c]);
    vit_x[c] = atik_float_to_bf16(v);
  }
  for (int r = 0; r < tc->patch_count; r++) {
    for (int c = 0; c < tc->d_model; c++) {
      float v = atik_bf16_to_float(vit_patch_tokens[r * VIT_MAX_D_MODEL + c]) +
                atik_bf16_to_float(tc->position_embeddings[(r + 1) * tc->d_model + c]);
      vit_x[(r + 1) * VIT_MAX_D_MODEL + c] = atik_float_to_bf16(v);
    }
  }
}

static void vit_attention_heads_ref(const vit_case_t *tc, uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *out) {
  for (int h = 0; h < tc->n_heads; h++) {
    int base = h * tc->head_dim;
    atik_ref_attention_bf16(q + base, tc->seq_len, tc->head_dim, VIT_MAX_D_MODEL,
                            k + base, tc->seq_len, VIT_MAX_D_MODEL,
                            v + base, tc->head_dim, VIT_MAX_D_MODEL,
                            tc->scale_bf16, out + base, VIT_MAX_D_MODEL, 0);
  }
}

static void vit_attention_heads_profiled(const vit_case_t *tc, vit_stats_t *stats) {
  for (int h = 0; h < tc->n_heads; h++) {
    int base = h * tc->head_dim;
    profile_attention_replay_bf16(
        "vit.attn.head",
        vit_q + base, tc->seq_len, tc->head_dim, VIT_MAX_D_MODEL,
        vit_k + base, tc->seq_len, VIT_MAX_D_MODEL,
        vit_v + base, tc->head_dim, VIT_MAX_D_MODEL,
        tc->scale_bf16, vit_context + base, VIT_MAX_D_MODEL,
        vit_hw_replay, tc->head_dim, 0, (int)tc->tolerance_x100000,
        0, &stats->attn_e2e_cycles, &stats->cpu_total_cycles,
        &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
        &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
  }
}

static void vit_forward_ref(const vit_case_t *tc, uint16_t *logits_out) {
  linear_ref_bf16(tc->patches, tc->patch_count, tc->patch_dim, tc->patch_dim,
                  tc->patch_w, tc->d_model, tc->patch_b, vit_patch_tokens, VIT_MAX_D_MODEL);
  vit_add_position(tc);
  for (int layer = 0; layer < tc->n_layers; layer++) {
    layer_norm_bf16(vit_x, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                    vit_layer_vec(tc->ln1_gamma, layer, tc->d_model),
                    vit_layer_vec(tc->ln1_beta, layer, tc->d_model), vit_ln, VIT_MAX_D_MODEL);
    linear_ref_bf16(vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL, vit_layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model, vit_layer_vec(tc->bq, layer, tc->d_model), vit_q, VIT_MAX_D_MODEL);
    linear_ref_bf16(vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL, vit_layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model, vit_layer_vec(tc->bk, layer, tc->d_model), vit_k, VIT_MAX_D_MODEL);
    linear_ref_bf16(vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL, vit_layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model, vit_layer_vec(tc->bv, layer, tc->d_model), vit_v, VIT_MAX_D_MODEL);
    vit_attention_heads_ref(tc, vit_q, vit_k, vit_v, vit_context);
    linear_ref_bf16(vit_context, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL, vit_layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model, vit_layer_vec(tc->bo, layer, tc->d_model), vit_attn_out, VIT_MAX_D_MODEL);
    add_matrix_bf16(vit_x, VIT_MAX_D_MODEL, vit_attn_out, VIT_MAX_D_MODEL, vit_x, VIT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layer_norm_bf16(vit_x, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                    vit_layer_vec(tc->ln2_gamma, layer, tc->d_model),
                    vit_layer_vec(tc->ln2_beta, layer, tc->d_model), vit_ln, VIT_MAX_D_MODEL);
    linear_ref_bf16(vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL, vit_layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim, vit_layer_hidden_vec(tc->b1, layer, tc->hidden_dim), vit_hidden, VIT_MAX_HIDDEN_DIM);
    gelu_bf16(vit_hidden, vit_act, tc->seq_len, tc->hidden_dim, VIT_MAX_HIDDEN_DIM, VIT_MAX_HIDDEN_DIM);
    linear_ref_bf16(vit_act, tc->seq_len, tc->hidden_dim, VIT_MAX_HIDDEN_DIM, vit_layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model, vit_layer_vec(tc->b2, layer, tc->d_model), vit_ffn_out, VIT_MAX_D_MODEL);
    add_matrix_bf16(vit_x, VIT_MAX_D_MODEL, vit_ffn_out, VIT_MAX_D_MODEL, vit_x, VIT_MAX_D_MODEL, tc->seq_len, tc->d_model);
  }
  layer_norm_bf16(vit_x, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                  tc->final_ln_gamma, tc->final_ln_beta, vit_ln, VIT_MAX_D_MODEL);
  linear_ref_bf16(vit_ln, 1, tc->d_model, VIT_MAX_D_MODEL, tc->classifier_w, tc->num_classes,
                  tc->classifier_b, logits_out, tc->num_classes);
}

static void vit_forward_profiled(const vit_case_t *tc, vit_stats_t *stats, uint16_t *logits_out) {
  memset(stats, 0, sizeof(*stats));
  stats->raw_hw_rc = ATIK_OK;
  atik_clear_counters();
  ATIK_PROGRESS("[case] begin vit %s seq=%d patch_dim=%d d=%d layers=%d", tc->name, tc->seq_len, tc->patch_dim, tc->d_model, tc->n_layers);

  profile_linear_replay_bf16("vit.patch", tc->patches, tc->patch_count, tc->patch_dim, tc->patch_dim,
                             tc->patch_w, tc->d_model, tc->patch_b, vit_patch_tokens, VIT_MAX_D_MODEL,
                             vit_hw_replay, VIT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                             0, &stats->patch_embed_cycles, &stats->cpu_total_cycles,
                             &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                             &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

  uint64_t start = atik_bench_read_cycles();
  vit_add_position(tc);
  account_cpu_cycles(atik_bench_read_cycles() - start, &stats->residual_cycles, &stats->cpu_total_cycles);
  ATIK_PROGRESS("[cpu] finished vit.cls_position");

  for (int layer = 0; layer < tc->n_layers; layer++) {
    ATIK_PROGRESS("[layer] begin vit layer=%d", layer);
    start = atik_bench_read_cycles();
    layer_norm_bf16(vit_x, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                    vit_layer_vec(tc->ln1_gamma, layer, tc->d_model),
                    vit_layer_vec(tc->ln1_beta, layer, tc->d_model), vit_ln, VIT_MAX_D_MODEL);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished vit.ln1 layer=%d", layer);

    profile_linear_replay_bf16("vit.q", vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                               vit_layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model,
                               vit_layer_vec(tc->bq, layer, tc->d_model), vit_q, VIT_MAX_D_MODEL,
                               vit_hw_replay, VIT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
    profile_linear_replay_bf16("vit.k", vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                               vit_layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model,
                               vit_layer_vec(tc->bk, layer, tc->d_model), vit_k, VIT_MAX_D_MODEL,
                               vit_hw_replay, VIT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
    profile_linear_replay_bf16("vit.v", vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                               vit_layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model,
                               vit_layer_vec(tc->bv, layer, tc->d_model), vit_v, VIT_MAX_D_MODEL,
                               vit_hw_replay, VIT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->qkv_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    vit_attention_heads_profiled(tc, stats);

    profile_linear_replay_bf16("vit.out", vit_context, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                               vit_layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model,
                               vit_layer_vec(tc->bo, layer, tc->d_model), vit_attn_out, VIT_MAX_D_MODEL,
                               vit_hw_replay, VIT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->out_proj_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    add_matrix_bf16(vit_x, VIT_MAX_D_MODEL, vit_attn_out, VIT_MAX_D_MODEL,
                    vit_x, VIT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->residual_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished vit.attn_residual layer=%d", layer);

    start = atik_bench_read_cycles();
    layer_norm_bf16(vit_x, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                    vit_layer_vec(tc->ln2_gamma, layer, tc->d_model),
                    vit_layer_vec(tc->ln2_beta, layer, tc->d_model), vit_ln, VIT_MAX_D_MODEL);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished vit.ln2 layer=%d", layer);

    profile_linear_replay_bf16("vit.fc1", vit_ln, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                               vit_layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim,
                               vit_layer_hidden_vec(tc->b1, layer, tc->hidden_dim), vit_hidden, VIT_MAX_HIDDEN_DIM,
                               vit_hw_replay, VIT_MAX_HIDDEN_DIM, (int)tc->tolerance_x100000,
                               0, &stats->mlp_fc1_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    gelu_bf16(vit_hidden, vit_act, tc->seq_len, tc->hidden_dim, VIT_MAX_HIDDEN_DIM, VIT_MAX_HIDDEN_DIM);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->gelu_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished vit.gelu layer=%d", layer);

    profile_linear_replay_bf16("vit.fc2", vit_act, tc->seq_len, tc->hidden_dim, VIT_MAX_HIDDEN_DIM,
                               vit_layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model,
                               vit_layer_vec(tc->b2, layer, tc->d_model), vit_ffn_out, VIT_MAX_D_MODEL,
                               vit_hw_replay, VIT_MAX_D_MODEL, (int)tc->tolerance_x100000,
                               0, &stats->mlp_fc2_cycles, &stats->cpu_total_cycles,
                               &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                               &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);

    start = atik_bench_read_cycles();
    add_matrix_bf16(vit_x, VIT_MAX_D_MODEL, vit_ffn_out, VIT_MAX_D_MODEL,
                    vit_x, VIT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    account_cpu_cycles(atik_bench_read_cycles() - start, &stats->residual_cycles, &stats->cpu_total_cycles);
    ATIK_PROGRESS("[cpu] finished vit.ffn_residual layer=%d", layer);
    ATIK_PROGRESS("[layer] finished vit layer=%d", layer);
  }

  start = atik_bench_read_cycles();
  layer_norm_bf16(vit_x, tc->seq_len, tc->d_model, VIT_MAX_D_MODEL,
                  tc->final_ln_gamma, tc->final_ln_beta, vit_ln, VIT_MAX_D_MODEL);
  account_cpu_cycles(atik_bench_read_cycles() - start, &stats->ln_cycles, &stats->cpu_total_cycles);
  ATIK_PROGRESS("[cpu] finished vit.final_ln");

  profile_linear_replay_bf16("vit.classifier", vit_ln, 1, tc->d_model, VIT_MAX_D_MODEL,
                             tc->classifier_w, tc->num_classes, tc->classifier_b, logits_out, tc->num_classes,
                             vit_hw_replay, tc->num_classes, (int)tc->tolerance_x100000,
                             0, &stats->classifier_cycles, &stats->cpu_total_cycles,
                             &stats->cpu_replaced_cycles, &stats->hw_replacement_cycles,
                             &stats->hw_max_diff_x100000, &stats->hw_mismatches, &stats->raw_hw_rc);
  ATIK_PROGRESS("[case] finished vit %s", tc->name);
}

static uint64_t run_vit_case(int case_index, const vit_case_t *tc) {
  char shape[128];
  int32_t hw_max_diff = 0, cpu_max_diff = 0;
  int32_t sw_sample[SAMPLE_COUNT], hw_sample[SAMPLE_COUNT], diff_sample[SAMPLE_COUNT];
  snprintf(shape, sizeof(shape), "seq=%d patches=%d patch_dim=%d d_model=%d heads=%d head_dim=%d hidden=%d layers=%d classes=%d",
           tc->seq_len, tc->patch_count, tc->patch_dim, tc->d_model, tc->n_heads, tc->head_dim, tc->hidden_dim, tc->n_layers, tc->num_classes);
  vit_stats_t stats;
  vit_forward_profiled(tc, &stats, vit_sw_logits);
  uint64_t cpu_cycles = stats.cpu_total_cycles;
  int cpu_mismatches = compare_bf16_x100000(tc->expected_logits, vit_sw_logits, 1, tc->num_classes,
                                            tc->num_classes, tc->num_classes,
                                            (int)tc->tolerance_x100000, &cpu_max_diff);
  memcpy(vit_logits, vit_sw_logits, sizeof(uint16_t) * (size_t)tc->num_classes);
  hw_max_diff = stats.hw_max_diff_x100000;
  int hw_rc = (int)stats.raw_hw_rc;
  int mismatches = cpu_mismatches + stats.hw_mismatches;
  int sample_count = tc->num_classes < SAMPLE_COUNT ? tc->num_classes : SAMPLE_COUNT;
  sample_bf16_flat(tc->expected_logits, vit_logits, 1, tc->num_classes, tc->num_classes, tc->num_classes, 0, sample_count, sw_sample, hw_sample, diff_sample);
  uint64_t hw_cycles = hybrid_cycles(stats.cpu_total_cycles, stats.cpu_replaced_cycles, stats.hw_replacement_cycles);
  const atik_log_stage_t stages[] = {
      {"patch_embed", stats.patch_embed_cycles}, {"ln", stats.ln_cycles},
      {"qkv", stats.qkv_proj_cycles}, {"attn", stats.attn_e2e_cycles},
      {"out", stats.out_proj_cycles}, {"fc1", stats.mlp_fc1_cycles},
      {"gelu", stats.gelu_cycles}, {"fc2", stats.mlp_fc2_cycles},
      {"classifier", stats.classifier_cycles}, {"residual", stats.residual_cycles},
      {"cpu_replaced", stats.cpu_replaced_cycles}, {"hw_replacement", stats.hw_replacement_cycles},
      {"atik_total", atik_read_counter(ATIK_COUNTER_TOTAL_CYCLES)},
  };
  const char *status = (hw_rc == ATIK_OK && mismatches == 0) ? "PASS" : "FAIL";
  const atik_log_result_t result = {
      .workload = "vit", .case_index = case_index, .name = tc->name, .shape = shape,
      .mode = "attention+matmul", .status = status, .hw_rc = (uint64_t)hw_rc, .raw_hw_rc = stats.raw_hw_rc,
      .mismatches = (uint64_t)mismatches, .cpu_cycles = cpu_cycles, .hw_cycles = hw_cycles,
      .stages = stages, .stage_count = (int)(sizeof(stages) / sizeof(stages[0])),
      .hw_max_abs_diff_x100000 = hw_max_diff, .cpu_ref_max_abs_diff_x100000 = cpu_max_diff,
  };
  const atik_log_sample_t sample = {
      .workload = "vit", .case_index = case_index, .name = tc->name, .tensor = "logits",
      .start = 0, .count = sample_count, .total = tc->num_classes, .hw_rc = (uint64_t)hw_rc,
      .sw_x10000 = sw_sample, .hw_x10000 = hw_sample, .diff_x10000 = diff_sample,
  };
  atik_log_result_human(&result);
  atik_log_sample(&sample);
  atik_log_result_json(&result);
  return (hw_rc == ATIK_OK && mismatches == 0) ? 0u : 1u;
}

int main(void) {
  uint64_t failures = 0;
  for (int i = 0; i < vit_case_count; i++) failures += run_vit_case(i, &vit_cases[i]);
  return failures == 0 ? 0 : 1;
}

#include "generated/vit_cases.c"

#else
int main(void) {
  const atik_log_result_t result = {
      .workload = "pytorch-workload",
      .case_index = 0,
      .name = "missing-workload-selection",
      .shape = "set PYTORCH_WORKLOAD=attention-operator|gpt2-prefill|tiny-bert|vit",
      .mode = "none",
      .status = "FAIL",
      .hw_rc = ATIK_ERR_BAD_DIMS,
      .raw_hw_rc = ATIK_ERR_BAD_DIMS,
      .mismatches = 1,
      .cpu_cycles = 0,
      .hw_cycles = 0,
      .stages = 0,
      .stage_count = 0,
      .hw_max_abs_diff_x100000 = 0,
      .cpu_ref_max_abs_diff_x100000 = 0,
  };
  atik_log_result(&result);
  return 1;
}
#endif
