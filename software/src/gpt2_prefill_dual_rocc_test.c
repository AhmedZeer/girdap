// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026

#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "generated/gpt2_prefill_cases.h"
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

static uint16_t sw_x[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_ln[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_q[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_k[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_v[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_context[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_attn_out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_res[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_hidden[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_HIDDEN_DIM];
static uint16_t sw_act[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_HIDDEN_DIM];
static uint16_t sw_ffn_out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL];
static uint16_t sw_logits[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_VOCAB_SIZE];
static float sw_scores[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_SEQ_LEN];
static float sw_probs[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_SEQ_LEN];
static uint16_t attn_scores_bf16[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_SEQ_LEN]
    __attribute__((aligned(64)));
static uint16_t attn_probs_bf16[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_SEQ_LEN]
    __attribute__((aligned(64)));

static uint16_t x[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t ln_out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t q_proj[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t k_proj[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t v_proj[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t context[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t attn_out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t res[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t hidden[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t act[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t ffn_out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t logits[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_VOCAB_SIZE] __attribute__((aligned(64)));

static uint64_t gemm_a_tiles[
    WS_GEMM8_A_TILE_WORDS(GPT2_PREFILL_MAX_SEQ_LEN, GPT2_PREFILL_MAX_GEMM_K)]
    __attribute__((aligned(64)));
static uint64_t gemm_b_tiles[
    WS_GEMM8_B_TILE_WORDS(GPT2_PREFILL_MAX_GEMM_N, GPT2_PREFILL_MAX_GEMM_K)]
    __attribute__((aligned(64)));
static uint64_t gemm_c_words[WS_GEMM8_C_TILE_WORDS] __attribute__((aligned(64)));

static uint64_t q_tiles[
    FPGA_SAFE_ATTN_Q_TILE_WORDS(1, GPT2_PREFILL_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t k_tiles[
    FPGA_SAFE_ATTN_K_TILE_WORDS(GPT2_PREFILL_MAX_SEQ_LEN, GPT2_PREFILL_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t v_tiles[
    FPGA_SAFE_ATTN_V_TILE_WORDS(GPT2_PREFILL_MAX_HEAD_DIM, GPT2_PREFILL_MAX_SEQ_LEN)]
    __attribute__((aligned(64)));
static uint64_t attn_out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

typedef struct {
  uint64_t embedding_cycles;
  uint64_t ln_cycles;
  uint64_t qkv_proj_cycles;
  uint64_t attn_e2e_cycles;
  uint64_t attn_accel_cycles;
  uint64_t attn_score_cycles;
  uint64_t attn_value_cycles;
  uint64_t out_proj_cycles;
  uint64_t mlp_fc1_cycles;
  uint64_t gelu_cycles;
  uint64_t mlp_fc2_cycles;
  uint64_t lm_head_cycles;
  uint64_t bias_cycles;
  uint64_t raw_hw_rc;
} gpt2_prefill_hw_stats_t;

static inline uint16_t tensor_at(const uint16_t *tensor, int row, int col, int ld) {
  return tensor[row * ld + col];
}

static inline const uint16_t *layer_vec(const uint16_t *base, int layer, int d_model) {
  return base + (layer * d_model);
}

static inline const uint16_t *layer_hidden_vec(const uint16_t *base, int layer, int hidden_dim) {
  return base + (layer * hidden_dim);
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
      const double delta = (double)bf16_to_float(tensor_at(input, r, c, ldin)) - mean;
      var_sum += delta * delta;
    }
    const double inv_std = 1.0 / sqrt((var_sum / (double)cols) + (double)LN_EPSILON);
    for (int c = 0; c < cols; c++) {
      const double xval = (double)bf16_to_float(tensor_at(input, r, c, ldin));
      const double g = (double)bf16_to_float(gamma[c]);
      const double b = (double)bf16_to_float(beta[c]);
      output[r * ldout + c] = float_to_bf16((float)(((xval - mean) * inv_std * g) + b));
    }
  }
}

static void embeddings_bf16(
    const gpt2_prefill_case_t *tc,
    uint16_t out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL]) {
  for (int r = 0; r < tc->seq_len; r++) {
    const int token = tc->token_ids[r];
    for (int c = 0; c < tc->d_model; c++) {
      const float value =
          bf16_to_float(tc->token_embeddings[token * tc->d_model + c]) +
          bf16_to_float(tc->position_embeddings[r * tc->d_model + c]);
      out[r][c] = float_to_bf16(value);
    }
  }
}

static void sw_matmul_bias_bf16(
    const uint16_t *A,
    int lda,
    const uint16_t *B,
    int ldb,
    const uint16_t *bias,
    uint16_t *C,
    int ldc,
    int M,
    int N,
    int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double acc = bias == 0 ? 0.0 : (double)bf16_to_float(bias[j]);
      for (int k = 0; k < K; k++) {
        acc += (double)bf16_to_float(tensor_at(A, i, k, lda)) *
               (double)bf16_to_float(tensor_at(B, k, j, ldb));
      }
      C[i * ldc + j] = float_to_bf16((float)acc);
    }
  }
}

static void add_bias_inplace_bf16(
    uint16_t *matrix,
    int ld,
    const uint16_t *bias,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const float value = bf16_to_float(tensor_at(matrix, r, c, ld)) + bf16_to_float(bias[c]);
      matrix[r * ld + c] = float_to_bf16(value);
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
      const float value = bf16_to_float(tensor_at(A, r, c, lda)) +
                          bf16_to_float(tensor_at(B, r, c, ldb));
      C[r * ldc + c] = float_to_bf16(value);
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
      const float xval = bf16_to_float(tensor_at(input, r, c, ldin));
      output[r * ldout + c] = float_to_bf16(0.5f * xval * (1.0f + erff(xval * inv_sqrt2)));
    }
  }
}

static void sw_causal_attention_heads(const gpt2_prefill_case_t *tc) {
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int kv = 0; kv <= qr; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(sw_q[qr][base + d]) *
                 (double)bf16_to_float(sw_k[kv][base + d]);
        }
        sw_scores[qr][kv] = (float)(dot * (double)scale);
      }

      float max_score = -INFINITY;
      for (int kv = 0; kv <= qr; kv++) {
        if (sw_scores[qr][kv] > max_score) {
          max_score = sw_scores[qr][kv];
        }
      }
      double denom = 0.0;
      for (int kv = 0; kv <= qr; kv++) {
        denom += exp((double)(sw_scores[qr][kv] - max_score));
      }
      for (int kv = 0; kv <= qr; kv++) {
        sw_probs[qr][kv] = (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
      }

      for (int vc = 0; vc < tc->head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv <= qr; kv++) {
          acc += (double)sw_probs[qr][kv] * (double)bf16_to_float(sw_v[kv][base + vc]);
        }
        sw_context[qr][base + vc] = float_to_bf16((float)acc);
      }
    }
  }
}

static void sw_gpt2_prefill_forward(const gpt2_prefill_case_t *tc) {
  embeddings_bf16(tc, sw_x);

  for (int layer = 0; layer < tc->n_layers; layer++) {
    layernorm_bf16(&sw_x[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_vec(tc->ln1_gamma, layer, tc->d_model),
                   layer_vec(tc->ln1_beta, layer, tc->d_model),
                   &sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);

    sw_matmul_bias_bf16(&sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model,
                        layer_vec(tc->bq, layer, tc->d_model),
                        &sw_q[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);
    sw_matmul_bias_bf16(&sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model,
                        layer_vec(tc->bk, layer, tc->d_model),
                        &sw_k[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);
    sw_matmul_bias_bf16(&sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model,
                        layer_vec(tc->bv, layer, tc->d_model),
                        &sw_v[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);

    sw_causal_attention_heads(tc);
    sw_matmul_bias_bf16(&sw_context[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model,
                        layer_vec(tc->bo, layer, tc->d_model),
                        &sw_attn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);
    residual_add_bf16(&sw_x[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &sw_attn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &sw_res[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);

    layernorm_bf16(&sw_res[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_vec(tc->ln2_gamma, layer, tc->d_model),
                   layer_vec(tc->ln2_beta, layer, tc->d_model),
                   &sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    sw_matmul_bias_bf16(&sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim,
                        layer_hidden_vec(tc->b1, layer, tc->hidden_dim),
                        &sw_hidden[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM,
                        tc->seq_len, tc->hidden_dim, tc->d_model);
    gelu_bf16(&sw_hidden[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM,
              &sw_act[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM, tc->seq_len, tc->hidden_dim);
    sw_matmul_bias_bf16(&sw_act[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM,
                        layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model,
                        layer_vec(tc->b2, layer, tc->d_model),
                        &sw_ffn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->hidden_dim);
    residual_add_bf16(&sw_res[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &sw_ffn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &sw_x[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  }

  layernorm_bf16(&sw_x[0][0], GPT2_PREFILL_MAX_D_MODEL,
                 tc->final_ln_gamma, tc->final_ln_beta,
                 &sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  sw_matmul_bias_bf16(&sw_ln[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      tc->lm_head, tc->vocab_size, 0,
                      &sw_logits[0][0], GPT2_PREFILL_MAX_VOCAB_SIZE,
                      tc->seq_len, tc->vocab_size, tc->d_model);
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
  sw_matmul_bias_bf16(A, lda, B, ldb, 0, C, ldc, M, N, K);
  *cycles += ws_read_cycles() - start;
  return WS_GEMM_OK;
#endif
}

static void sw_causal_attention_heads_from_buffers(
    const gpt2_prefill_case_t *tc,
    const uint16_t q[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    const uint16_t k[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    const uint16_t v[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    uint16_t out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    gpt2_prefill_hw_stats_t *stats) {
  const uint64_t total_start = ws_read_cycles();
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    for (int qr = 0; qr < tc->seq_len; qr++) {
      uint64_t phase_start = ws_read_cycles();
      for (int kv = 0; kv <= qr; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(q[qr][base + d]) *
                 (double)bf16_to_float(k[kv][base + d]);
        }
        sw_scores[qr][kv] = (float)(dot * (double)scale);
      }
      stats->attn_score_cycles += ws_read_cycles() - phase_start;

      phase_start = ws_read_cycles();
      float max_score = -INFINITY;
      for (int kv = 0; kv <= qr; kv++) {
        if (sw_scores[qr][kv] > max_score) {
          max_score = sw_scores[qr][kv];
        }
      }
      double denom = 0.0;
      for (int kv = 0; kv <= qr; kv++) {
        denom += exp((double)(sw_scores[qr][kv] - max_score));
      }
      for (int kv = 0; kv <= qr; kv++) {
        sw_probs[qr][kv] = (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
      }
      stats->attn_accel_cycles += ws_read_cycles() - phase_start;

      phase_start = ws_read_cycles();
      for (int vc = 0; vc < tc->head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv <= qr; kv++) {
          acc += (double)sw_probs[qr][kv] *
                 (double)bf16_to_float(v[kv][base + vc]);
        }
        out[qr][base + vc] = float_to_bf16((float)acc);
      }
      stats->attn_value_cycles += ws_read_cycles() - phase_start;
    }
  }
  stats->attn_e2e_cycles += ws_read_cycles() - total_start;
}

static int softmax_hw_causal_attention_heads(
    const gpt2_prefill_case_t *tc,
    const uint16_t q[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    const uint16_t k[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    const uint16_t v[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    uint16_t out[GPT2_PREFILL_MAX_SEQ_LEN][GPT2_PREFILL_MAX_D_MODEL],
    gpt2_prefill_hw_stats_t *stats) {
  const uint64_t total_start = ws_read_cycles();
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    for (int qr = 0; qr < tc->seq_len; qr++) {
      uint64_t phase_start = ws_read_cycles();
      for (int kv = 0; kv <= qr; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(q[qr][base + d]) *
                 (double)bf16_to_float(k[kv][base + d]);
        }
        attn_scores_bf16[qr][kv] = float_to_bf16((float)(dot * (double)scale));
      }
      stats->attn_score_cycles += ws_read_cycles() - phase_start;

      online_softmax_stats_t softmax_stats;
      memset(&softmax_stats, 0, sizeof(softmax_stats));
      const int softmax_rc = online_softmax_bf16(
          &attn_scores_bf16[qr][0],
          qr + 1,
          &attn_probs_bf16[qr][0],
          &softmax_stats);
      stats->raw_hw_rc = (uint64_t)softmax_rc;
      stats->attn_accel_cycles += softmax_stats.hw_e2e_cycles;
      if (softmax_rc != ONLINE_SOFTMAX_OK) {
        stats->attn_e2e_cycles += ws_read_cycles() - total_start;
        return softmax_rc;
      }

      phase_start = ws_read_cycles();
      for (int vc = 0; vc < tc->head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv <= qr; kv++) {
          acc += (double)bf16_to_float(attn_probs_bf16[qr][kv]) *
                 (double)bf16_to_float(v[kv][base + vc]);
        }
        out[qr][base + vc] = float_to_bf16((float)acc);
      }
      stats->attn_value_cycles += ws_read_cycles() - phase_start;
    }
  }
  stats->attn_e2e_cycles += ws_read_cycles() - total_start;
  return WS_GEMM_OK;
}

static int hw_causal_attention_heads(
    const gpt2_prefill_case_t *tc,
    const fpga_safe_attention_workspace_t *workspace,
    gpt2_prefill_hw_stats_t *stats) {
#if GIRDAP_HW_ATTENTION
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    for (int qr = 0; qr < tc->seq_len; qr++) {
      fpga_safe_attention_stats_t attn_stats;
      memset(&attn_stats, 0, sizeof(attn_stats));
      const int rc = fpga_safe_attention_bf16_hwpack(
          &q_proj[qr][base], GPT2_PREFILL_MAX_D_MODEL,
          &k_proj[0][base], GPT2_PREFILL_MAX_D_MODEL,
          &v_proj[0][base], GPT2_PREFILL_MAX_D_MODEL,
          1, qr + 1, tc->head_dim, tc->head_dim,
          tc->scale_bf16,
          &context[qr][base], GPT2_PREFILL_MAX_D_MODEL,
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
  }
  return FPGA_SAFE_ATTN_OK;
#elif GIRDAP_HW_SOFTMAX
  (void)workspace;
  return softmax_hw_causal_attention_heads(tc, q_proj, k_proj, v_proj, context, stats);
#else
  (void)workspace;
  sw_causal_attention_heads_from_buffers(tc, q_proj, k_proj, v_proj, context, stats);
  return WS_GEMM_OK;
#endif
}

static int hw_gpt2_prefill_forward(
    const gpt2_prefill_case_t *tc,
    const ws_gemm_workspace_t *gemm_workspace,
    const fpga_safe_attention_workspace_t *attn_workspace,
    gpt2_prefill_hw_stats_t *stats) {
  memset(stats, 0, sizeof(*stats));

  uint64_t start = ws_read_cycles();
  embeddings_bf16(tc, x);
  stats->embedding_cycles = ws_read_cycles() - start;

  int rc = WS_GEMM_OK;
  for (int layer = 0; layer < tc->n_layers; layer++) {
    start = ws_read_cycles();
    layernorm_bf16(&x[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_vec(tc->ln1_gamma, layer, tc->d_model),
                   layer_vec(tc->ln1_beta, layer, tc->d_model),
                   &ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->ln_cycles += ws_read_cycles() - start;

    rc = hw_gemm(&ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                 layer_w_dxd(tc->wq, layer, tc->d_model), tc->d_model,
                 &q_proj[0][0], GPT2_PREFILL_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model,
                 gemm_workspace, &stats->qkv_proj_cycles);
    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_w_dxd(tc->wk, layer, tc->d_model), tc->d_model,
                   &k_proj[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->d_model,
                   gemm_workspace, &stats->qkv_proj_cycles);
    }
    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_w_dxd(tc->wv, layer, tc->d_model), tc->d_model,
                   &v_proj[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->d_model,
                   gemm_workspace, &stats->qkv_proj_cycles);
    }

    start = ws_read_cycles();
    add_bias_inplace_bf16(&q_proj[0][0], GPT2_PREFILL_MAX_D_MODEL,
                          layer_vec(tc->bq, layer, tc->d_model), tc->seq_len, tc->d_model);
    add_bias_inplace_bf16(&k_proj[0][0], GPT2_PREFILL_MAX_D_MODEL,
                          layer_vec(tc->bk, layer, tc->d_model), tc->seq_len, tc->d_model);
    add_bias_inplace_bf16(&v_proj[0][0], GPT2_PREFILL_MAX_D_MODEL,
                          layer_vec(tc->bv, layer, tc->d_model), tc->seq_len, tc->d_model);
    stats->bias_cycles += ws_read_cycles() - start;

    if (rc == WS_GEMM_OK) {
      rc = hw_causal_attention_heads(tc, attn_workspace, stats);
    }
    if (rc == FPGA_SAFE_ATTN_OK || rc == WS_GEMM_OK) {
      rc = hw_gemm(&context[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_w_dxd(tc->wo, layer, tc->d_model), tc->d_model,
                   &attn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->d_model,
                   gemm_workspace, &stats->out_proj_cycles);
    }

    start = ws_read_cycles();
    add_bias_inplace_bf16(&attn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                          layer_vec(tc->bo, layer, tc->d_model), tc->seq_len, tc->d_model);
    residual_add_bf16(&x[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &attn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &res[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->bias_cycles += ws_read_cycles() - start;

    start = ws_read_cycles();
    layernorm_bf16(&res[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_vec(tc->ln2_gamma, layer, tc->d_model),
                   layer_vec(tc->ln2_beta, layer, tc->d_model),
                   &ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->ln_cycles += ws_read_cycles() - start;

    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim), tc->hidden_dim,
                   &hidden[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM,
                   tc->seq_len, tc->hidden_dim, tc->d_model,
                   gemm_workspace, &stats->mlp_fc1_cycles);
    }

    start = ws_read_cycles();
    add_bias_inplace_bf16(&hidden[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM,
                          layer_hidden_vec(tc->b1, layer, tc->hidden_dim), tc->seq_len, tc->hidden_dim);
    gelu_bf16(&hidden[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM,
              &act[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM, tc->seq_len, tc->hidden_dim);
    stats->gelu_cycles += ws_read_cycles() - start;

    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&act[0][0], GPT2_PREFILL_MAX_HIDDEN_DIM,
                   layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model), tc->d_model,
                   &ffn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->hidden_dim,
                   gemm_workspace, &stats->mlp_fc2_cycles);
    }

    start = ws_read_cycles();
    add_bias_inplace_bf16(&ffn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                          layer_vec(tc->b2, layer, tc->d_model), tc->seq_len, tc->d_model);
    residual_add_bf16(&res[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &ffn_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                      &x[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->bias_cycles += ws_read_cycles() - start;
  }

  start = ws_read_cycles();
  layernorm_bf16(&x[0][0], GPT2_PREFILL_MAX_D_MODEL,
                 tc->final_ln_gamma, tc->final_ln_beta,
                 &ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL, tc->seq_len, tc->d_model);
  stats->ln_cycles += ws_read_cycles() - start;

  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(&ln_out[0][0], GPT2_PREFILL_MAX_D_MODEL,
                 tc->lm_head, tc->vocab_size,
                 &logits[0][0], GPT2_PREFILL_MAX_VOCAB_SIZE,
                 tc->seq_len, tc->vocab_size, tc->d_model,
                 gemm_workspace, &stats->lm_head_cycles);
  }
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

int main(void) {
  const ws_gemm_workspace_t gemm_workspace =
      ws_gemm8_make_workspace(gemm_a_tiles, gemm_b_tiles, gemm_c_words,
                              GPT2_PREFILL_MAX_SEQ_LEN,
                              GPT2_PREFILL_MAX_GEMM_N,
                              GPT2_PREFILL_MAX_GEMM_K);
  const fpga_safe_attention_workspace_t attn_workspace =
      fpga_safe_attention_make_workspace(q_tiles, k_tiles, v_tiles, attn_out_words,
                                         1,
                                         GPT2_PREFILL_MAX_SEQ_LEN,
                                         GPT2_PREFILL_MAX_HEAD_DIM,
                                         GPT2_PREFILL_MAX_HEAD_DIM);

  printf("=== Girdap GPT-2 Prefill Test ===\n");
  printf("MODE_INFO,name=%s,matmul_hw=%d,attention_hw=%d,softmax_hw=%d\n",
         GIRDAP_BENCHMARK_MODE, GIRDAP_HW_MATMUL, GIRDAP_HW_ATTENTION,
         GIRDAP_HW_SOFTMAX);
  printf("OPCODE_INFO,matmul=%d,attention=%d,softmax=%d,maxK=512,causal=row_prefix_calls\n",
         SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);
  printf("CSV_HEADER,case,name,seq_len,d_model,n_heads,head_dim,hidden_dim,n_layers,vocab_size,cpu_cycles,hw_e2e_cycles,embedding_cycles,ln_cycles,qkv_proj_cycles,attn_e2e_cycles,attn_accel_cycles,attn_score_cycles,attn_value_cycles,out_proj_cycles,mlp_fc1_cycles,gelu_cycles,mlp_fc2_cycles,lm_head_cycles,bias_cycles,hw_rc,raw_hw_rc,speedup_x100,hw_max_abs_diff_x100000,cpu_ref_max_abs_diff_x100000,mismatches\n");

  int total_mismatches = 0;
  for (int i = 0; i < gpt2_prefill_case_count; i++) {
    const gpt2_prefill_case_t *tc = &gpt2_prefill_cases[i];
    memset(logits, 0, sizeof(logits));

    const uint64_t cpu_start = ws_read_cycles();
    sw_gpt2_prefill_forward(tc);
    const uint64_t cpu_cycles = ws_read_cycles() - cpu_start;

    int cpu_mismatches = 0;
    const float cpu_ref_diff = max_diff_vs(tc->expected_logits, &sw_logits[0][0],
                                           tc->seq_len, tc->vocab_size,
                                           GPT2_PREFILL_MAX_VOCAB_SIZE,
                                           &cpu_mismatches,
                                           tc->tolerance_x100000, tc->name);

    gpt2_prefill_hw_stats_t stats;
    const int hw_rc = hw_gpt2_prefill_forward(tc, &gemm_workspace, &attn_workspace, &stats);
    const uint64_t hw_e2e_cycles = stats.embedding_cycles + stats.ln_cycles +
                                   stats.qkv_proj_cycles + stats.attn_e2e_cycles +
                                   stats.out_proj_cycles + stats.mlp_fc1_cycles +
                                   stats.gelu_cycles + stats.mlp_fc2_cycles +
                                   stats.lm_head_cycles + stats.bias_cycles;
    const uint64_t speedup_x100 =
        hw_e2e_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_e2e_cycles;

    int mismatches = 1;
    float hw_diff = 0.0f;
    if (hw_rc == WS_GEMM_OK) {
      hw_diff = max_diff_vs(tc->expected_logits, &logits[0][0],
                            tc->seq_len, tc->vocab_size,
                            GPT2_PREFILL_MAX_VOCAB_SIZE,
                            &mismatches, tc->tolerance_x100000, tc->name);
    }
    total_mismatches += mismatches;

    printf("CSV_DATA,%d,%s,%d,%d,%d,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%d,%lu,%lu,%lu,%lu,%d\n",
           i, tc->name, tc->seq_len, tc->d_model, tc->n_heads, tc->head_dim,
           tc->hidden_dim, tc->n_layers, tc->vocab_size,
           (unsigned long)cpu_cycles,
           (unsigned long)hw_e2e_cycles,
           (unsigned long)stats.embedding_cycles,
           (unsigned long)stats.ln_cycles,
           (unsigned long)stats.qkv_proj_cycles,
           (unsigned long)stats.attn_e2e_cycles,
           (unsigned long)stats.attn_accel_cycles,
           (unsigned long)stats.attn_score_cycles,
           (unsigned long)stats.attn_value_cycles,
           (unsigned long)stats.out_proj_cycles,
           (unsigned long)stats.mlp_fc1_cycles,
           (unsigned long)stats.gelu_cycles,
           (unsigned long)stats.mlp_fc2_cycles,
           (unsigned long)stats.lm_head_cycles,
           (unsigned long)stats.bias_cycles,
           hw_rc,
           (unsigned long)stats.raw_hw_rc,
           (unsigned long)speedup_x100,
           (unsigned long)(hw_diff * 100000.0f),
           (unsigned long)(cpu_ref_diff * 100000.0f),
           mismatches);
  }

  if (total_mismatches == 0) {
    printf("PASS: Girdap GPT-2 prefill matched PyTorch tolerance.\n");
  } else {
    printf("FAIL: total GPT-2 prefill mismatches = %d\n", total_mismatches);
  }
  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/gpt2_prefill_cases.c"
