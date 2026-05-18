// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026

#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "generated/encoder_model_cases.h"
#include "online_softmax.h"
#include "ws_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LN_EPSILON 0.00001f

#ifndef GIRDAP_HW_MATMUL
#define GIRDAP_HW_MATMUL 1
#endif

#ifndef GIRDAP_HW_ATTENTION
#define GIRDAP_HW_ATTENTION 1
#endif

#ifndef GIRDAP_HW_SOFTMAX
#define GIRDAP_HW_SOFTMAX 0
#endif

#if GIRDAP_HW_ATTENTION && GIRDAP_HW_SOFTMAX
#error "Use either fused attention hardware or softmax hardware for this benchmark, not both."
#endif

#if GIRDAP_HW_ATTENTION && GIRDAP_HW_MATMUL
#define GIRDAP_BENCHMARK_MODE "attention+matmul"
#elif GIRDAP_HW_ATTENTION
#define GIRDAP_BENCHMARK_MODE "attention-only"
#elif GIRDAP_HW_MATMUL
#define GIRDAP_BENCHMARK_MODE "matmul-only"
#elif GIRDAP_HW_SOFTMAX
#define GIRDAP_BENCHMARK_MODE "softmax-only"
#else
#define GIRDAP_BENCHMARK_MODE "software-only"
#endif

static uint16_t sw_buf0[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_buf1[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t hw_buf0[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t hw_buf1[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));

static uint16_t sw_ln1_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_q_proj[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_k_proj[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_v_proj[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_context[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_attn_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_res1[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_ln2_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static uint16_t sw_mlp_hidden[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_HIDDEN_DIM];
static uint16_t sw_mlp_act[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_HIDDEN_DIM];
static uint16_t sw_mlp_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL];
static float sw_scores[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_SEQ_LEN];
static float sw_probs[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_SEQ_LEN];
static uint16_t attn_scores_bf16[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_SEQ_LEN]
    __attribute__((aligned(64)));
static uint16_t attn_probs_bf16[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_SEQ_LEN]
    __attribute__((aligned(64)));

static uint16_t ln1_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t q_proj[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t k_proj[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t v_proj[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t context[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t attn_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t res1[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t ln2_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t mlp_hidden[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t mlp_act[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t mlp_out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL] __attribute__((aligned(64)));

static uint64_t gemm_a_tiles[
    WS_GEMM8_A_TILE_WORDS(ENCODER_MODEL_MAX_SEQ_LEN, ENCODER_MODEL_MAX_HIDDEN_DIM)]
    __attribute__((aligned(64)));
static uint64_t gemm_b_tiles[
    WS_GEMM8_B_TILE_WORDS(ENCODER_MODEL_MAX_HIDDEN_DIM, ENCODER_MODEL_MAX_HIDDEN_DIM)]
    __attribute__((aligned(64)));
static uint64_t gemm_c_words[WS_GEMM8_C_TILE_WORDS] __attribute__((aligned(64)));

static uint64_t q_tiles[
    FPGA_SAFE_ATTN_Q_TILE_WORDS(ENCODER_MODEL_MAX_SEQ_LEN, ENCODER_MODEL_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t k_tiles[
    FPGA_SAFE_ATTN_K_TILE_WORDS(ENCODER_MODEL_MAX_SEQ_LEN, ENCODER_MODEL_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t v_tiles[
    FPGA_SAFE_ATTN_V_TILE_WORDS(ENCODER_MODEL_MAX_HEAD_DIM, ENCODER_MODEL_MAX_SEQ_LEN)]
    __attribute__((aligned(64)));
static uint64_t attn_out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

typedef struct {
  uint64_t ln_residual_cycles;
  uint64_t qkv_proj_cycles;
  uint64_t attn_e2e_cycles;
  uint64_t attn_accel_cycles;
  uint64_t attn_score_cycles;
  uint64_t attn_value_cycles;
  uint64_t out_proj_cycles;
  uint64_t mlp_fc1_cycles;
  uint64_t gelu_cycles;
  uint64_t mlp_fc2_cycles;
  uint64_t raw_hw_rc;
} encoder_hw_stats_t;

static inline uint16_t tensor_at(const uint16_t *tensor, int row, int col, int ld) {
  return tensor[row * ld + col];
}

static void print_float_value(float value) {
  if (value < 0.0f) {
    printf("-");
    value = -value;
  }
  const int whole_part = (int)value;
  const int frac_part = (int)((value - (float)whole_part) * 10000.0f);
  printf("%d.%04d", whole_part, frac_part);
}

static int sample_chunk_start(const char *name, int rows, int cols) {
  uint32_t hash = 2166136261u;
  for (const char *p = name; *p != '\0'; p++) {
    hash ^= (uint8_t)(*p);
    hash *= 16777619u;
  }
  hash ^= (uint32_t)(rows * 131 + cols * 17);
  const int total = rows * cols;
  if (total <= 10) {
    return 0;
  }
  return (int)(hash % (uint32_t)(total - 10 + 1));
}

static void print_solution_sample_chunk(
    int case_index,
    const char *case_name,
    const char *tensor_name,
    const uint16_t *sw,
    int sw_ld,
    const uint16_t *hw,
    int hw_ld,
    int rows,
    int cols,
    int hw_rc) {
  const int total = rows * cols;
  const int count = total < 10 ? total : 10;
  const int start = sample_chunk_start(case_name, rows, cols);
  printf("\n  sample %s[start=%d count=%d total=%d]\n", tensor_name, start, count, total);
  printf("    sw :");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int row = idx / cols;
    const int col = idx % cols;
    printf(" ");
    print_float_value(bf16_to_float(tensor_at(sw, row, col, sw_ld)));
  }
  printf("\n    hw :");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int row = idx / cols;
    const int col = idx % cols;
    printf(" ");
    print_float_value(bf16_to_float(tensor_at(hw, row, col, hw_ld)));
  }
  printf("\n    diff:");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int row = idx / cols;
    const int col = idx % cols;
    const float sw_val = bf16_to_float(tensor_at(sw, row, col, sw_ld));
    const float hw_val = bf16_to_float(tensor_at(hw, row, col, hw_ld));
    printf(" ");
    print_float_value(fabsf(sw_val - hw_val));
  }
  printf("\n");
  printf("SAMPLE_JSON {\"workload\":\"encoder-model\",\"case\":%d,\"name\":\"%s\",\"tensor\":\"%s\",\"start\":%d,\"count\":%d,\"total\":%d,\"hw_rc\":%d,\"sw\":[",
         case_index, case_name, tensor_name, start, count, total, hw_rc);
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int row = idx / cols;
    const int col = idx % cols;
    if (i != 0) {
      printf(",");
    }
    print_float_value(bf16_to_float(tensor_at(sw, row, col, sw_ld)));
  }
  printf("],\"hw\":[");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int row = idx / cols;
    const int col = idx % cols;
    if (i != 0) {
      printf(",");
    }
    print_float_value(bf16_to_float(tensor_at(hw, row, col, hw_ld)));
  }
  printf("],\"diff\":[");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int row = idx / cols;
    const int col = idx % cols;
    const float sw_val = bf16_to_float(tensor_at(sw, row, col, sw_ld));
    const float hw_val = bf16_to_float(tensor_at(hw, row, col, hw_ld));
    if (i != 0) {
      printf(",");
    }
    print_float_value(fabsf(sw_val - hw_val));
  }
  printf("]}\n");
}

static inline const uint16_t *layer_vec(const uint16_t *base, int layer, int d_model) {
  return base + (layer * d_model);
}

static inline const uint16_t *layer_w_dxd(const uint16_t *base, int layer, int d_model) {
  return base + (layer * d_model * d_model);
}

static inline const uint16_t *layer_w_dxh(const uint16_t *base, int layer, int d_model, int hidden_dim) {
  return base + (layer * d_model * hidden_dim);
}

static inline const uint16_t *layer_w_hxd(const uint16_t *base, int layer, int hidden_dim, int d_model) {
  return base + (layer * hidden_dim * d_model);
}

static void copy_matrix(
    const uint16_t *src,
    int ldsrc,
    uint16_t *dst,
    int lddst,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      dst[r * lddst + c] = src[r * ldsrc + c];
    }
  }
}

static void layernorm_bf16(
    const uint16_t *input,
    int ldin,
    const uint16_t *gamma,
    const uint16_t *beta,
    uint16_t *output,
    int ldout,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    double sum = 0.0;
    for (int c = 0; c < cols; c++) {
      sum += (double)bf16_to_float(tensor_at(input, r, c, ldin));
    }
    const double mean = sum / (double)cols;

    double var_sum = 0.0;
    for (int c = 0; c < cols; c++) {
      const double d = (double)bf16_to_float(tensor_at(input, r, c, ldin)) - mean;
      var_sum += d * d;
    }
    const double inv_std = 1.0 / sqrt((var_sum / (double)cols) + (double)LN_EPSILON);

    for (int c = 0; c < cols; c++) {
      const double x = (double)bf16_to_float(tensor_at(input, r, c, ldin));
      const double g = (double)bf16_to_float(gamma[c]);
      const double b = (double)bf16_to_float(beta[c]);
      output[r * ldout + c] = float_to_bf16((float)(((x - mean) * inv_std * g) + b));
    }
  }
}

static void sw_matmul_bf16(
    const uint16_t *A,
    int lda,
    const uint16_t *B,
    int ldb,
    uint16_t *C,
    int ldc,
    int M,
    int N,
    int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double acc = 0.0;
      for (int k = 0; k < K; k++) {
        acc += (double)bf16_to_float(tensor_at(A, i, k, lda)) *
               (double)bf16_to_float(tensor_at(B, k, j, ldb));
      }
      C[i * ldc + j] = float_to_bf16((float)acc);
    }
  }
}

static void residual_add_bf16(
    const uint16_t *A,
    int lda,
    const uint16_t *B,
    int ldb,
    uint16_t *C,
    int ldc,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const float v = bf16_to_float(tensor_at(A, r, c, lda)) +
                      bf16_to_float(tensor_at(B, r, c, ldb));
      C[r * ldc + c] = float_to_bf16(v);
    }
  }
}

static void gelu_bf16(
    const uint16_t *input,
    int ldin,
    uint16_t *output,
    int ldout,
    int rows,
    int cols) {
  const float inv_sqrt2 = 0.70710678118f;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const float x = bf16_to_float(tensor_at(input, r, c, ldin));
      output[r * ldout + c] = float_to_bf16(0.5f * x * (1.0f + erff(x * inv_sqrt2)));
    }
  }
}

static void sw_attention_heads(
    const encoder_model_case_t *tc,
    const uint16_t q[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    const uint16_t k[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    const uint16_t v[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    uint16_t out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL]) {
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int kv = 0; kv < tc->seq_len; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(q[qr][base + d]) *
                 (double)bf16_to_float(k[kv][base + d]);
        }
        sw_scores[qr][kv] = (float)(dot * (double)scale);
      }
    }

    for (int qr = 0; qr < tc->seq_len; qr++) {
      float max_score = -INFINITY;
      for (int kv = 0; kv < tc->seq_len; kv++) {
        if (sw_scores[qr][kv] > max_score) {
          max_score = sw_scores[qr][kv];
        }
      }
      double denom = 0.0;
      for (int kv = 0; kv < tc->seq_len; kv++) {
        denom += exp((double)(sw_scores[qr][kv] - max_score));
      }
      for (int kv = 0; kv < tc->seq_len; kv++) {
        sw_probs[qr][kv] =
            (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
      }
    }

    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int vc = 0; vc < tc->head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv < tc->seq_len; kv++) {
          acc += (double)sw_probs[qr][kv] *
                 (double)bf16_to_float(v[kv][base + vc]);
        }
        out[qr][base + vc] = float_to_bf16((float)acc);
      }
    }
  }
}

static void sw_encoder_layer(
    const encoder_model_case_t *tc,
    int layer,
    const uint16_t *input,
    int ldin,
    uint16_t *output,
    int ldout) {
  const uint16_t *ln1_gamma = layer_vec(tc->ln1_gamma, layer, tc->d_model);
  const uint16_t *ln1_beta = layer_vec(tc->ln1_beta, layer, tc->d_model);
  const uint16_t *ln2_gamma = layer_vec(tc->ln2_gamma, layer, tc->d_model);
  const uint16_t *ln2_beta = layer_vec(tc->ln2_beta, layer, tc->d_model);
  const uint16_t *wq = layer_w_dxd(tc->wq, layer, tc->d_model);
  const uint16_t *wk = layer_w_dxd(tc->wk, layer, tc->d_model);
  const uint16_t *wv = layer_w_dxd(tc->wv, layer, tc->d_model);
  const uint16_t *wo = layer_w_dxd(tc->wo, layer, tc->d_model);
  const uint16_t *w1 = layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim);
  const uint16_t *w2 = layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model);

  layernorm_bf16(input, ldin, ln1_gamma, ln1_beta,
                 &sw_ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  sw_matmul_bf16(&sw_ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 wq, tc->d_model, &sw_q_proj[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model);
  sw_matmul_bf16(&sw_ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 wk, tc->d_model, &sw_k_proj[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model);
  sw_matmul_bf16(&sw_ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 wv, tc->d_model, &sw_v_proj[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model);
  sw_attention_heads(tc, sw_q_proj, sw_k_proj, sw_v_proj, sw_context);
  sw_matmul_bf16(&sw_context[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 wo, tc->d_model, &sw_attn_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model);
  residual_add_bf16(input, ldin, &sw_attn_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                    &sw_res1[0][0], ENCODER_MODEL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  layernorm_bf16(&sw_res1[0][0], ENCODER_MODEL_MAX_D_MODEL, ln2_gamma, ln2_beta,
                 &sw_ln2_out[0][0], ENCODER_MODEL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  sw_matmul_bf16(&sw_ln2_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 w1, tc->hidden_dim, &sw_mlp_hidden[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM,
                 tc->seq_len, tc->hidden_dim, tc->d_model);
  gelu_bf16(&sw_mlp_hidden[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM,
            &sw_mlp_act[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM, tc->seq_len, tc->hidden_dim);
  sw_matmul_bf16(&sw_mlp_act[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM,
                 w2, tc->d_model, &sw_mlp_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->hidden_dim);
  residual_add_bf16(&sw_res1[0][0], ENCODER_MODEL_MAX_D_MODEL,
                    &sw_mlp_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                    output, ldout, tc->seq_len, tc->d_model);
}

static void sw_encoder_model(
    const encoder_model_case_t *tc,
    const uint16_t **final_output,
    int *final_ld) {
  copy_matrix(tc->x, tc->d_model, &sw_buf0[0][0], ENCODER_MODEL_MAX_D_MODEL,
              tc->seq_len, tc->d_model);
  uint16_t *in = &sw_buf0[0][0];
  uint16_t *out = &sw_buf1[0][0];
  for (int layer = 0; layer < tc->n_layers; layer++) {
    sw_encoder_layer(tc, layer, in, ENCODER_MODEL_MAX_D_MODEL, out, ENCODER_MODEL_MAX_D_MODEL);
    uint16_t *tmp = in;
    in = out;
    out = tmp;
  }
  *final_output = in;
  *final_ld = ENCODER_MODEL_MAX_D_MODEL;
}

static int hw_gemm(
    const uint16_t *A,
    int lda,
    const uint16_t *B,
    int ldb,
    uint16_t *C,
    int ldc,
    int M,
    int N,
    int K,
    const ws_gemm_workspace_t *workspace,
    uint64_t *cycles) {
#if GIRDAP_HW_MATMUL
  ws_gemm_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  const int rc = ws_gemm8_bf16(A, lda, B, ldb, C, ldc, M, N, K, workspace, &stats);
  *cycles += stats.hw_e2e_cycles;
  return rc;
#else
  (void)workspace;
  const uint64_t start = ws_read_cycles();
  sw_matmul_bf16(A, lda, B, ldb, C, ldc, M, N, K);
  *cycles += ws_read_cycles() - start;
  return WS_GEMM_OK;
#endif
}

static void sw_attention_heads_from_buffers(
    const encoder_model_case_t *tc,
    const uint16_t q[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    const uint16_t k[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    const uint16_t v[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    uint16_t out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    encoder_hw_stats_t *stats) {
  const uint64_t total_start = ws_read_cycles();
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    uint64_t phase_start = ws_read_cycles();
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int kv = 0; kv < tc->seq_len; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(q[qr][base + d]) *
                 (double)bf16_to_float(k[kv][base + d]);
        }
        sw_scores[qr][kv] = (float)(dot * (double)scale);
      }
    }
    stats->attn_score_cycles += ws_read_cycles() - phase_start;

    phase_start = ws_read_cycles();
    for (int qr = 0; qr < tc->seq_len; qr++) {
      float max_score = -INFINITY;
      for (int kv = 0; kv < tc->seq_len; kv++) {
        if (sw_scores[qr][kv] > max_score) {
          max_score = sw_scores[qr][kv];
        }
      }
      double denom = 0.0;
      for (int kv = 0; kv < tc->seq_len; kv++) {
        denom += exp((double)(sw_scores[qr][kv] - max_score));
      }
      for (int kv = 0; kv < tc->seq_len; kv++) {
        sw_probs[qr][kv] =
            (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
      }
    }
    stats->attn_accel_cycles += ws_read_cycles() - phase_start;

    phase_start = ws_read_cycles();
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int vc = 0; vc < tc->head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv < tc->seq_len; kv++) {
          acc += (double)sw_probs[qr][kv] *
                 (double)bf16_to_float(v[kv][base + vc]);
        }
        out[qr][base + vc] = float_to_bf16((float)acc);
      }
    }
    stats->attn_value_cycles += ws_read_cycles() - phase_start;
  }
  stats->attn_e2e_cycles += ws_read_cycles() - total_start;
}

static int softmax_hw_attention_heads(
    const encoder_model_case_t *tc,
    const uint16_t q[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    const uint16_t k[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    const uint16_t v[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    uint16_t out[ENCODER_MODEL_MAX_SEQ_LEN][ENCODER_MODEL_MAX_D_MODEL],
    encoder_hw_stats_t *stats) {
  const uint64_t total_start = ws_read_cycles();
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    uint64_t phase_start = ws_read_cycles();
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int kv = 0; kv < tc->seq_len; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(q[qr][base + d]) *
                 (double)bf16_to_float(k[kv][base + d]);
        }
        attn_scores_bf16[qr][kv] = float_to_bf16((float)(dot * (double)scale));
      }
    }
    stats->attn_score_cycles += ws_read_cycles() - phase_start;

    online_softmax_stats_t softmax_stats;
    memset(&softmax_stats, 0, sizeof(softmax_stats));
    const int softmax_rc = online_softmax_bf16_rows(
        &attn_scores_bf16[0][0],
        tc->seq_len,
        tc->seq_len,
        ENCODER_MODEL_MAX_SEQ_LEN,
        &attn_probs_bf16[0][0],
        ENCODER_MODEL_MAX_SEQ_LEN,
        &softmax_stats);
    stats->raw_hw_rc = (uint64_t)softmax_rc;
    stats->attn_accel_cycles += softmax_stats.hw_e2e_cycles;
    if (softmax_rc != ONLINE_SOFTMAX_OK) {
      stats->attn_e2e_cycles += ws_read_cycles() - total_start;
      return softmax_rc;
    }

    phase_start = ws_read_cycles();
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int vc = 0; vc < tc->head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv < tc->seq_len; kv++) {
          acc += (double)bf16_to_float(attn_probs_bf16[qr][kv]) *
                 (double)bf16_to_float(v[kv][base + vc]);
        }
        out[qr][base + vc] = float_to_bf16((float)acc);
      }
    }
    stats->attn_value_cycles += ws_read_cycles() - phase_start;
  }
  stats->attn_e2e_cycles += ws_read_cycles() - total_start;
  return WS_GEMM_OK;
}

static int hw_attention_heads(
    const encoder_model_case_t *tc,
    const fpga_safe_attention_workspace_t *workspace,
    encoder_hw_stats_t *stats) {
#if GIRDAP_HW_ATTENTION
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    fpga_safe_attention_stats_t attn_stats;
    memset(&attn_stats, 0, sizeof(attn_stats));
    const int rc = fpga_safe_attention_bf16_hwpack(
        &q_proj[0][base], ENCODER_MODEL_MAX_D_MODEL,
        &k_proj[0][base], ENCODER_MODEL_MAX_D_MODEL,
        &v_proj[0][base], ENCODER_MODEL_MAX_D_MODEL,
        tc->seq_len, tc->seq_len, tc->head_dim, tc->head_dim,
        tc->scale_bf16,
        &context[0][base], ENCODER_MODEL_MAX_D_MODEL,
        workspace,
        &attn_stats);
    stats->raw_hw_rc = attn_stats.raw_hw_rc;
    stats->attn_accel_cycles += attn_stats.hw_e2e_cycles;
    stats->attn_score_cycles += attn_stats.score_cycles;
    stats->attn_value_cycles += attn_stats.value_cycles;
    stats->attn_e2e_cycles += attn_stats.q_pack_cycles + attn_stats.k_pack_cycles +
                              attn_stats.v_pack_cycles + attn_stats.hw_e2e_cycles +
                              attn_stats.copy_out_cycles;
    if (rc != FPGA_SAFE_ATTN_OK) {
      return rc;
    }
  }
  return FPGA_SAFE_ATTN_OK;
#elif GIRDAP_HW_SOFTMAX
  (void)workspace;
  return softmax_hw_attention_heads(tc, q_proj, k_proj, v_proj, context, stats);
#else
  (void)workspace;
  sw_attention_heads_from_buffers(tc, q_proj, k_proj, v_proj, context, stats);
  return WS_GEMM_OK;
#endif
}

static int hw_encoder_layer(
    const encoder_model_case_t *tc,
    int layer,
    const uint16_t *input,
    int ldin,
    uint16_t *output,
    int ldout,
    const ws_gemm_workspace_t *gemm_workspace,
    const fpga_safe_attention_workspace_t *attn_workspace,
    encoder_hw_stats_t *stats) {
  const uint16_t *ln1_gamma = layer_vec(tc->ln1_gamma, layer, tc->d_model);
  const uint16_t *ln1_beta = layer_vec(tc->ln1_beta, layer, tc->d_model);
  const uint16_t *ln2_gamma = layer_vec(tc->ln2_gamma, layer, tc->d_model);
  const uint16_t *ln2_beta = layer_vec(tc->ln2_beta, layer, tc->d_model);
  const uint16_t *wq = layer_w_dxd(tc->wq, layer, tc->d_model);
  const uint16_t *wk = layer_w_dxd(tc->wk, layer, tc->d_model);
  const uint16_t *wv = layer_w_dxd(tc->wv, layer, tc->d_model);
  const uint16_t *wo = layer_w_dxd(tc->wo, layer, tc->d_model);
  const uint16_t *w1 = layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim);
  const uint16_t *w2 = layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model);

  uint64_t stage_start = ws_read_cycles();
  layernorm_bf16(input, ldin, ln1_gamma, ln1_beta,
                 &ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  stats->ln_residual_cycles += ws_read_cycles() - stage_start;

  int rc = hw_gemm(&ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL, wq, tc->d_model,
                   &q_proj[0][0], ENCODER_MODEL_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                   &stats->qkv_proj_cycles);
  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(&ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL, wk, tc->d_model,
                 &k_proj[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                 &stats->qkv_proj_cycles);
  }
  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(&ln1_out[0][0], ENCODER_MODEL_MAX_D_MODEL, wv, tc->d_model,
                 &v_proj[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                 &stats->qkv_proj_cycles);
  }
  if (rc == WS_GEMM_OK) {
    rc = hw_attention_heads(tc, attn_workspace, stats);
  }
  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(&context[0][0], ENCODER_MODEL_MAX_D_MODEL, wo, tc->d_model,
                 &attn_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                 &stats->out_proj_cycles);
  }

  stage_start = ws_read_cycles();
  residual_add_bf16(input, ldin, &attn_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                    &res1[0][0], ENCODER_MODEL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  layernorm_bf16(&res1[0][0], ENCODER_MODEL_MAX_D_MODEL, ln2_gamma, ln2_beta,
                 &ln2_out[0][0], ENCODER_MODEL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  stats->ln_residual_cycles += ws_read_cycles() - stage_start;

  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(&ln2_out[0][0], ENCODER_MODEL_MAX_D_MODEL, w1, tc->hidden_dim,
                 &mlp_hidden[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM,
                 tc->seq_len, tc->hidden_dim, tc->d_model, gemm_workspace,
                 &stats->mlp_fc1_cycles);
  }

  stage_start = ws_read_cycles();
  gelu_bf16(&mlp_hidden[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM,
            &mlp_act[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM, tc->seq_len, tc->hidden_dim);
  stats->gelu_cycles += ws_read_cycles() - stage_start;

  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(&mlp_act[0][0], ENCODER_MODEL_MAX_HIDDEN_DIM, w2, tc->d_model,
                 &mlp_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->hidden_dim, gemm_workspace,
                 &stats->mlp_fc2_cycles);
  }

  stage_start = ws_read_cycles();
  residual_add_bf16(&res1[0][0], ENCODER_MODEL_MAX_D_MODEL,
                    &mlp_out[0][0], ENCODER_MODEL_MAX_D_MODEL,
                    output, ldout, tc->seq_len, tc->d_model);
  stats->ln_residual_cycles += ws_read_cycles() - stage_start;

  return rc;
}

static int hw_encoder_model(
    const encoder_model_case_t *tc,
    const ws_gemm_workspace_t *gemm_workspace,
    const fpga_safe_attention_workspace_t *attn_workspace,
    const uint16_t **final_output,
    int *final_ld,
    encoder_hw_stats_t *stats) {
  memset(stats, 0, sizeof(*stats));
  copy_matrix(tc->x, tc->d_model, &hw_buf0[0][0], ENCODER_MODEL_MAX_D_MODEL,
              tc->seq_len, tc->d_model);
  uint16_t *in = &hw_buf0[0][0];
  uint16_t *out = &hw_buf1[0][0];
  int rc = WS_GEMM_OK;
  for (int layer = 0; layer < tc->n_layers; layer++) {
    rc = hw_encoder_layer(tc, layer, in, ENCODER_MODEL_MAX_D_MODEL, out,
                          ENCODER_MODEL_MAX_D_MODEL, gemm_workspace, attn_workspace, stats);
    if (rc != WS_GEMM_OK) {
      break;
    }
    uint16_t *tmp = in;
    in = out;
    out = tmp;
  }
  *final_output = in;
  *final_ld = ENCODER_MODEL_MAX_D_MODEL;
  return rc;
}

static float max_diff_vs(
    const uint16_t *expected,
    const uint16_t *actual,
    int rows,
    int cols,
    int ldactual,
    int *mismatches,
    uint32_t tolerance_x100000,
    const char *name) {
  float max_abs_diff = 0.0f;
  *mismatches = 0;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const float ref = bf16_to_float(tensor_at(expected, r, c, cols));
      const float got = bf16_to_float(tensor_at(actual, r, c, ldactual));
      const float diff = fabsf(got - ref);
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
      }
      if ((uint32_t)(diff * 100000.0f) > tolerance_x100000) {
        if (*mismatches < 8) {
          printf("Mismatch case=%s row=%d col=%d ref=", name, r, c);
          print_float_inline(ref);
          printf("got=");
          print_float_inline(got);
          printf("diff=");
          print_float_inline(diff);
          printf("\n");
        }
        (*mismatches)++;
      }
    }
  }
  return max_abs_diff;
}

static uint64_t hw_stats_total_cycles(const encoder_hw_stats_t *stats) {
  return stats->ln_residual_cycles + stats->qkv_proj_cycles +
         stats->attn_e2e_cycles + stats->out_proj_cycles +
         stats->mlp_fc1_cycles + stats->gelu_cycles + stats->mlp_fc2_cycles;
}

int main(void) {
  const ws_gemm_workspace_t gemm_workspace =
      ws_gemm8_make_workspace(gemm_a_tiles, gemm_b_tiles, gemm_c_words,
                              ENCODER_MODEL_MAX_SEQ_LEN,
                              ENCODER_MODEL_MAX_HIDDEN_DIM,
                              ENCODER_MODEL_MAX_HIDDEN_DIM);
  const fpga_safe_attention_workspace_t attn_workspace =
      fpga_safe_attention_make_workspace(q_tiles, k_tiles, v_tiles, attn_out_words,
                                         ENCODER_MODEL_MAX_SEQ_LEN,
                                         ENCODER_MODEL_MAX_SEQ_LEN,
                                         ENCODER_MODEL_MAX_HEAD_DIM,
                                         ENCODER_MODEL_MAX_HEAD_DIM);

  printf("=== Girdap Encoder Model Test ===\n");
  printf("mode: %s  hardware: matmul=%d attention=%d softmax=%d\n",
         GIRDAP_BENCHMARK_MODE, GIRDAP_HW_MATMUL, GIRDAP_HW_ATTENTION,
         GIRDAP_HW_SOFTMAX);
  printf("opcodes: matmul=%d attention=%d softmax=%d\n",
         SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);
  printf("RUN_INFO_JSON {\"workload\":\"encoder-model\",\"mode\":\"%s\",\"matmul_hw\":%d,\"attention_hw\":%d,\"softmax_hw\":%d,\"matmul_opcode\":%d,\"attention_opcode\":%d,\"softmax_opcode\":%d}\n",
         GIRDAP_BENCHMARK_MODE, GIRDAP_HW_MATMUL, GIRDAP_HW_ATTENTION,
         GIRDAP_HW_SOFTMAX, SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);

  int total_mismatches = 0;
  for (int i = 0; i < encoder_model_case_count; i++) {
    const encoder_model_case_t *tc = &encoder_model_cases[i];

    const uint16_t *sw_final = 0;
    int sw_final_ld = 0;
    const uint64_t cpu_start = ws_read_cycles();
    sw_encoder_model(tc, &sw_final, &sw_final_ld);
    const uint64_t cpu_cycles = ws_read_cycles() - cpu_start;

    int cpu_mismatches = 0;
    const float cpu_ref_diff = max_diff_vs(tc->expected, sw_final,
                                           tc->seq_len, tc->d_model, sw_final_ld,
                                           &cpu_mismatches, tc->tolerance_x100000, tc->name);

    const uint16_t *hw_final = 0;
    int hw_final_ld = 0;
    encoder_hw_stats_t stats;
    const int hw_rc = hw_encoder_model(tc, &gemm_workspace, &attn_workspace,
                                       &hw_final, &hw_final_ld, &stats);
    const uint64_t hw_e2e_cycles = hw_stats_total_cycles(&stats);
    const uint64_t speedup_x100 =
        hw_e2e_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_e2e_cycles;

    int mismatches = 1;
    float hw_diff = 0.0f;
    if (hw_rc == WS_GEMM_OK) {
      hw_diff = max_diff_vs(tc->expected, hw_final, tc->seq_len, tc->d_model,
                            hw_final_ld, &mismatches,
                            tc->tolerance_x100000, tc->name);
    }

    printf("\n[Case %d] %s\n", i, tc->name);
    printf("  shape : seq=%d d_model=%d heads=%d head_dim=%d hidden=%d layers=%d\n",
           tc->seq_len, tc->d_model, tc->n_heads, tc->head_dim,
           tc->hidden_dim, tc->n_layers);
    printf("  mode  : %s  opcodes: matmul=%d attention=%d softmax=%d\n",
           GIRDAP_BENCHMARK_MODE, SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);
    printf("  status: %s  hw_rc=%d raw_hw_rc=%lu mismatches=%d\n",
           (hw_rc == WS_GEMM_OK && mismatches == 0) ? "PASS" : "FAIL",
           hw_rc, (unsigned long)stats.raw_hw_rc, mismatches);
    printf("  cycles: cpu=%lu hw=%lu speedup=%lu.%02lux\n",
           (unsigned long)cpu_cycles, (unsigned long)hw_e2e_cycles,
           (unsigned long)(speedup_x100 / 100u), (unsigned long)(speedup_x100 % 100u));
    printf("          ln_residual=%lu qkv=%lu attn=%lu attn_accel=%lu out=%lu fc1=%lu gelu=%lu fc2=%lu\n",
           (unsigned long)stats.ln_residual_cycles,
           (unsigned long)stats.qkv_proj_cycles,
           (unsigned long)stats.attn_e2e_cycles,
           (unsigned long)stats.attn_accel_cycles,
           (unsigned long)stats.out_proj_cycles,
           (unsigned long)stats.mlp_fc1_cycles,
           (unsigned long)stats.gelu_cycles,
           (unsigned long)stats.mlp_fc2_cycles);
    printf("  error : hw_max_abs=");
    print_float_value(hw_diff);
    printf(" cpu_ref_max_abs=");
    print_float_value(cpu_ref_diff);
    printf("\n");

    print_solution_sample_chunk(i, tc->name, "final_output", sw_final,
                                sw_final_ld, hw_final, hw_final_ld,
                                tc->seq_len, tc->d_model,
                                hw_rc);
    total_mismatches += mismatches;

    printf("RESULT_JSON {\"workload\":\"encoder-model\",\"case\":%d,\"name\":\"%s\",\"status\":\"%s\",\"hw_rc\":%d,\"raw_hw_rc\":%lu,\"mismatches\":%d,\"cpu_cycles\":%lu,\"hw_cycles\":%lu,\"speedup_x100\":%lu,\"hw_max_abs_diff_x100000\":%lu,\"cpu_ref_max_abs_diff_x100000\":%lu}\n",
           i, tc->name,
           (hw_rc == WS_GEMM_OK && mismatches == 0) ? "PASS" : "FAIL",
           hw_rc,
           (unsigned long)stats.raw_hw_rc,
           mismatches,
           (unsigned long)cpu_cycles,
           (unsigned long)hw_e2e_cycles,
           (unsigned long)speedup_x100,
           (unsigned long)(hw_diff * 100000.0f),
           (unsigned long)(cpu_ref_diff * 100000.0f));
  }

  if (total_mismatches == 0) {
    printf("PASS: Girdap encoder model matched PyTorch tolerance.\n");
  } else {
    printf("FAIL: total encoder model mismatches = %d\n", total_mismatches);
  }
  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/encoder_model_cases.c"
