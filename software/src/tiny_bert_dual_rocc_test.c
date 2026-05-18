// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026

#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "generated/tiny_bert_cases.h"
#include "online_softmax.h"
#include "ws_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LN_EPSILON 0.00001f
#define TINY_BERT_ERR_MASK_UNSUPPORTED 10

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

static uint16_t sw_x[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_q[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_k[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_v[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_context[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_attn_out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_res[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_hidden[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_HIDDEN_DIM];
static uint16_t sw_act[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_HIDDEN_DIM];
static uint16_t sw_ffn_out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL];
static uint16_t sw_pool[TINY_BERT_MAX_D_MODEL];
static uint16_t sw_logits[TINY_BERT_MAX_NUM_CLASSES];
static float sw_scores[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_SEQ_LEN];
static float sw_probs[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_SEQ_LEN];
static uint16_t attn_scores_bf16[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_SEQ_LEN]
    __attribute__((aligned(64)));
static uint16_t attn_probs_bf16[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_SEQ_LEN]
    __attribute__((aligned(64)));

static uint16_t x[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t q_proj[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t k_proj[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t v_proj[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t context[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t attn_out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t res[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t hidden[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t act[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_HIDDEN_DIM] __attribute__((aligned(64)));
static uint16_t ffn_out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t pool[TINY_BERT_MAX_D_MODEL] __attribute__((aligned(64)));
static uint16_t logits[TINY_BERT_MAX_NUM_CLASSES] __attribute__((aligned(64)));

static uint64_t gemm_a_tiles[
    WS_GEMM8_A_TILE_WORDS(TINY_BERT_MAX_SEQ_LEN, TINY_BERT_MAX_HIDDEN_DIM)]
    __attribute__((aligned(64)));
static uint64_t gemm_b_tiles[
    WS_GEMM8_B_TILE_WORDS(TINY_BERT_MAX_HIDDEN_DIM, TINY_BERT_MAX_HIDDEN_DIM)]
    __attribute__((aligned(64)));
static uint64_t gemm_c_words[WS_GEMM8_C_TILE_WORDS] __attribute__((aligned(64)));

static uint64_t q_tiles[
    FPGA_SAFE_ATTN_Q_TILE_WORDS(TINY_BERT_MAX_SEQ_LEN, TINY_BERT_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t k_tiles[
    FPGA_SAFE_ATTN_K_TILE_WORDS(TINY_BERT_MAX_SEQ_LEN, TINY_BERT_MAX_HEAD_DIM)]
    __attribute__((aligned(64)));
static uint64_t v_tiles[
    FPGA_SAFE_ATTN_V_TILE_WORDS(TINY_BERT_MAX_HEAD_DIM, TINY_BERT_MAX_SEQ_LEN)]
    __attribute__((aligned(64)));
static uint64_t attn_out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

typedef struct {
  uint64_t embedding_cycles;
  uint64_t qkv_proj_cycles;
  uint64_t attn_e2e_cycles;
  uint64_t attn_accel_cycles;
  uint64_t attn_score_cycles;
  uint64_t attn_value_cycles;
  uint64_t out_proj_cycles;
  uint64_t mlp_fc1_cycles;
  uint64_t mlp_fc2_cycles;
  uint64_t pooler_cycles;
  uint64_t classifier_cycles;
  uint64_t bias_cycles;
  uint64_t ln_residual_cycles;
  uint64_t gelu_cycles;
  uint64_t tanh_cycles;
  uint64_t raw_hw_rc;
} tiny_bert_hw_stats_t;

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
  printf("SAMPLE_JSON {\"workload\":\"tiny-bert\",\"case\":%d,\"name\":\"%s\",\"tensor\":\"%s\",\"start\":%d,\"count\":%d,\"total\":%d,\"hw_rc\":%d,\"sw\":[",
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

static int mask_all_valid(const tiny_bert_case_t *tc) {
  for (int i = 0; i < tc->seq_len; i++) {
    if (tc->attention_mask[i] == 0) {
      return 0;
    }
  }
  return 1;
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
      const double xval = (double)bf16_to_float(tensor_at(input, r, c, ldin));
      const double g = (double)bf16_to_float(gamma[c]);
      const double b = (double)bf16_to_float(beta[c]);
      output[r * ldout + c] = float_to_bf16((float)(((xval - mean) * inv_std * g) + b));
    }
  }
}

static void embeddings_bf16(
    const tiny_bert_case_t *tc,
    uint16_t out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL]) {
  for (int r = 0; r < tc->seq_len; r++) {
    const int token = tc->token_ids[r];
    const int type_id = tc->type_ids[r];
    for (int c = 0; c < tc->d_model; c++) {
      const float v = bf16_to_float(tc->word_embeddings[token * tc->d_model + c]) +
                      bf16_to_float(tc->position_embeddings[r * tc->d_model + c]) +
                      bf16_to_float(tc->token_type_embeddings[type_id * tc->d_model + c]);
      out[r][c] = float_to_bf16(v);
    }
  }
  layernorm_bf16(&out[0][0], TINY_BERT_MAX_D_MODEL, tc->emb_ln_gamma, tc->emb_ln_beta,
                 &out[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
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
      double acc = (double)bf16_to_float(bias[j]);
      for (int k = 0; k < K; k++) {
        acc += (double)bf16_to_float(tensor_at(A, i, k, lda)) *
               (double)bf16_to_float(tensor_at(B, k, j, ldb));
      }
      C[i * ldc + j] = float_to_bf16((float)acc);
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

static void add_bias_inplace_bf16(
    uint16_t *matrix,
    int ld,
    const uint16_t *bias,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const float v = bf16_to_float(tensor_at(matrix, r, c, ld)) + bf16_to_float(bias[c]);
      matrix[r * ld + c] = float_to_bf16(v);
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
      const float xval = bf16_to_float(tensor_at(input, r, c, ldin));
      output[r * ldout + c] = float_to_bf16(0.5f * xval * (1.0f + erff(xval * inv_sqrt2)));
    }
  }
}

static void tanh_bf16_vec(uint16_t *values, int count) {
  for (int i = 0; i < count; i++) {
    values[i] = float_to_bf16(tanhf(bf16_to_float(values[i])));
  }
}

static void sw_attention_heads(const tiny_bert_case_t *tc) {
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int kv = 0; kv < tc->seq_len; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(sw_q[qr][base + d]) *
                 (double)bf16_to_float(sw_k[kv][base + d]);
        }
        sw_scores[qr][kv] = tc->attention_mask[kv] ? (float)(dot * (double)scale) : -10000.0f;
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
                 (double)bf16_to_float(sw_v[kv][base + vc]);
        }
        sw_context[qr][base + vc] = float_to_bf16((float)acc);
      }
    }
  }
}

static void sw_tiny_bert_forward(const tiny_bert_case_t *tc) {
  embeddings_bf16(tc, sw_x);
  for (int layer = 0; layer < tc->n_layers; layer++) {
    const uint16_t *attn_ln_gamma = layer_vec(tc->attn_ln_gamma, layer, tc->d_model);
    const uint16_t *attn_ln_beta = layer_vec(tc->attn_ln_beta, layer, tc->d_model);
    const uint16_t *ffn_ln_gamma = layer_vec(tc->ffn_ln_gamma, layer, tc->d_model);
    const uint16_t *ffn_ln_beta = layer_vec(tc->ffn_ln_beta, layer, tc->d_model);
    const uint16_t *wq = layer_w_dxd(tc->wq, layer, tc->d_model);
    const uint16_t *wk = layer_w_dxd(tc->wk, layer, tc->d_model);
    const uint16_t *wv = layer_w_dxd(tc->wv, layer, tc->d_model);
    const uint16_t *wo = layer_w_dxd(tc->wo, layer, tc->d_model);
    const uint16_t *w1 = layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim);
    const uint16_t *w2 = layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model);

    sw_matmul_bias_bf16(&sw_x[0][0], TINY_BERT_MAX_D_MODEL, wq, tc->d_model,
                        layer_vec(tc->bq, layer, tc->d_model),
                        &sw_q[0][0], TINY_BERT_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);
    sw_matmul_bias_bf16(&sw_x[0][0], TINY_BERT_MAX_D_MODEL, wk, tc->d_model,
                        layer_vec(tc->bk, layer, tc->d_model),
                        &sw_k[0][0], TINY_BERT_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);
    sw_matmul_bias_bf16(&sw_x[0][0], TINY_BERT_MAX_D_MODEL, wv, tc->d_model,
                        layer_vec(tc->bv, layer, tc->d_model),
                        &sw_v[0][0], TINY_BERT_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);
    sw_attention_heads(tc);
    sw_matmul_bias_bf16(&sw_context[0][0], TINY_BERT_MAX_D_MODEL, wo, tc->d_model,
                        layer_vec(tc->bo, layer, tc->d_model),
                        &sw_attn_out[0][0], TINY_BERT_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->d_model);
    residual_add_bf16(&sw_x[0][0], TINY_BERT_MAX_D_MODEL,
                      &sw_attn_out[0][0], TINY_BERT_MAX_D_MODEL,
                      &sw_res[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layernorm_bf16(&sw_res[0][0], TINY_BERT_MAX_D_MODEL, attn_ln_gamma, attn_ln_beta,
                   &sw_x[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    sw_matmul_bias_bf16(&sw_x[0][0], TINY_BERT_MAX_D_MODEL, w1, tc->hidden_dim,
                        layer_hidden_vec(tc->b1, layer, tc->hidden_dim),
                        &sw_hidden[0][0], TINY_BERT_MAX_HIDDEN_DIM,
                        tc->seq_len, tc->hidden_dim, tc->d_model);
    gelu_bf16(&sw_hidden[0][0], TINY_BERT_MAX_HIDDEN_DIM,
              &sw_act[0][0], TINY_BERT_MAX_HIDDEN_DIM, tc->seq_len, tc->hidden_dim);
    sw_matmul_bias_bf16(&sw_act[0][0], TINY_BERT_MAX_HIDDEN_DIM, w2, tc->d_model,
                        layer_vec(tc->b2, layer, tc->d_model),
                        &sw_ffn_out[0][0], TINY_BERT_MAX_D_MODEL,
                        tc->seq_len, tc->d_model, tc->hidden_dim);
    residual_add_bf16(&sw_x[0][0], TINY_BERT_MAX_D_MODEL,
                      &sw_ffn_out[0][0], TINY_BERT_MAX_D_MODEL,
                      &sw_res[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layernorm_bf16(&sw_res[0][0], TINY_BERT_MAX_D_MODEL, ffn_ln_gamma, ffn_ln_beta,
                   &sw_x[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
  }
  sw_matmul_bias_bf16(&sw_x[0][0], TINY_BERT_MAX_D_MODEL, tc->pool_w, tc->d_model,
                      tc->pool_b, sw_pool, tc->d_model, 1, tc->d_model, tc->d_model);
  tanh_bf16_vec(sw_pool, tc->d_model);
  sw_matmul_bias_bf16(sw_pool, tc->d_model, tc->classifier_w, tc->num_classes,
                      tc->classifier_b, sw_logits, tc->num_classes,
                      1, tc->num_classes, tc->d_model);
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
    const tiny_bert_case_t *tc,
    const uint16_t q[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    const uint16_t k[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    const uint16_t v[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    uint16_t out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    tiny_bert_hw_stats_t *stats) {
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
        sw_scores[qr][kv] = tc->attention_mask[kv] ? (float)(dot * (double)scale) : -10000.0f;
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
    const tiny_bert_case_t *tc,
    const uint16_t q[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    const uint16_t k[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    const uint16_t v[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    uint16_t out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL],
    tiny_bert_hw_stats_t *stats) {
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
        const float score = tc->attention_mask[kv] ? (float)(dot * (double)scale) : -10000.0f;
        attn_scores_bf16[qr][kv] = float_to_bf16(score);
      }
    }
    stats->attn_score_cycles += ws_read_cycles() - phase_start;

    online_softmax_stats_t softmax_stats;
    memset(&softmax_stats, 0, sizeof(softmax_stats));
    const int softmax_rc = online_softmax_bf16_rows(
        &attn_scores_bf16[0][0],
        tc->seq_len,
        tc->seq_len,
        TINY_BERT_MAX_SEQ_LEN,
        &attn_probs_bf16[0][0],
        TINY_BERT_MAX_SEQ_LEN,
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
    const tiny_bert_case_t *tc,
    const fpga_safe_attention_workspace_t *workspace,
    tiny_bert_hw_stats_t *stats) {
#if GIRDAP_HW_ATTENTION
  if (!mask_all_valid(tc)) {
    return TINY_BERT_ERR_MASK_UNSUPPORTED;
  }
  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;
    fpga_safe_attention_stats_t attn_stats;
    memset(&attn_stats, 0, sizeof(attn_stats));
    const int rc = fpga_safe_attention_bf16_hwpack(
        &q_proj[0][base], TINY_BERT_MAX_D_MODEL,
        &k_proj[0][base], TINY_BERT_MAX_D_MODEL,
        &v_proj[0][base], TINY_BERT_MAX_D_MODEL,
        tc->seq_len, tc->seq_len, tc->head_dim, tc->head_dim,
        tc->scale_bf16,
        &context[0][base], TINY_BERT_MAX_D_MODEL,
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

static int hw_tiny_bert_forward(
    const tiny_bert_case_t *tc,
    const ws_gemm_workspace_t *gemm_workspace,
    const fpga_safe_attention_workspace_t *attn_workspace,
    tiny_bert_hw_stats_t *stats) {
  memset(stats, 0, sizeof(*stats));

  uint64_t stage_start = ws_read_cycles();
  embeddings_bf16(tc, x);
  stats->embedding_cycles += ws_read_cycles() - stage_start;

  int rc = WS_GEMM_OK;
  for (int layer = 0; layer < tc->n_layers; layer++) {
    const uint16_t *attn_ln_gamma = layer_vec(tc->attn_ln_gamma, layer, tc->d_model);
    const uint16_t *attn_ln_beta = layer_vec(tc->attn_ln_beta, layer, tc->d_model);
    const uint16_t *ffn_ln_gamma = layer_vec(tc->ffn_ln_gamma, layer, tc->d_model);
    const uint16_t *ffn_ln_beta = layer_vec(tc->ffn_ln_beta, layer, tc->d_model);
    const uint16_t *wq = layer_w_dxd(tc->wq, layer, tc->d_model);
    const uint16_t *wk = layer_w_dxd(tc->wk, layer, tc->d_model);
    const uint16_t *wv = layer_w_dxd(tc->wv, layer, tc->d_model);
    const uint16_t *wo = layer_w_dxd(tc->wo, layer, tc->d_model);
    const uint16_t *w1 = layer_w_dxh(tc->w1, layer, tc->d_model, tc->hidden_dim);
    const uint16_t *w2 = layer_w_hxd(tc->w2, layer, tc->hidden_dim, tc->d_model);

    rc = hw_gemm(&x[0][0], TINY_BERT_MAX_D_MODEL, wq, tc->d_model,
                 &q_proj[0][0], TINY_BERT_MAX_D_MODEL,
                 tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                 &stats->qkv_proj_cycles);
    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&x[0][0], TINY_BERT_MAX_D_MODEL, wk, tc->d_model,
                   &k_proj[0][0], TINY_BERT_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                   &stats->qkv_proj_cycles);
    }
    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&x[0][0], TINY_BERT_MAX_D_MODEL, wv, tc->d_model,
                   &v_proj[0][0], TINY_BERT_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                   &stats->qkv_proj_cycles);
    }

    stage_start = ws_read_cycles();
    add_bias_inplace_bf16(&q_proj[0][0], TINY_BERT_MAX_D_MODEL,
                          layer_vec(tc->bq, layer, tc->d_model), tc->seq_len, tc->d_model);
    add_bias_inplace_bf16(&k_proj[0][0], TINY_BERT_MAX_D_MODEL,
                          layer_vec(tc->bk, layer, tc->d_model), tc->seq_len, tc->d_model);
    add_bias_inplace_bf16(&v_proj[0][0], TINY_BERT_MAX_D_MODEL,
                          layer_vec(tc->bv, layer, tc->d_model), tc->seq_len, tc->d_model);
    stats->bias_cycles += ws_read_cycles() - stage_start;

    if (rc == WS_GEMM_OK) {
      rc = hw_attention_heads(tc, attn_workspace, stats);
    }
    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&context[0][0], TINY_BERT_MAX_D_MODEL, wo, tc->d_model,
                   &attn_out[0][0], TINY_BERT_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->d_model, gemm_workspace,
                   &stats->out_proj_cycles);
    }

    stage_start = ws_read_cycles();
    add_bias_inplace_bf16(&attn_out[0][0], TINY_BERT_MAX_D_MODEL,
                          layer_vec(tc->bo, layer, tc->d_model), tc->seq_len, tc->d_model);
    stats->bias_cycles += ws_read_cycles() - stage_start;

    stage_start = ws_read_cycles();
    residual_add_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL,
                      &attn_out[0][0], TINY_BERT_MAX_D_MODEL,
                      &res[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layernorm_bf16(&res[0][0], TINY_BERT_MAX_D_MODEL, attn_ln_gamma, attn_ln_beta,
                   &x[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->ln_residual_cycles += ws_read_cycles() - stage_start;

    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&x[0][0], TINY_BERT_MAX_D_MODEL, w1, tc->hidden_dim,
                   &hidden[0][0], TINY_BERT_MAX_HIDDEN_DIM,
                   tc->seq_len, tc->hidden_dim, tc->d_model, gemm_workspace,
                   &stats->mlp_fc1_cycles);
    }

    stage_start = ws_read_cycles();
    add_bias_inplace_bf16(&hidden[0][0], TINY_BERT_MAX_HIDDEN_DIM,
                          layer_hidden_vec(tc->b1, layer, tc->hidden_dim),
                          tc->seq_len, tc->hidden_dim);
    stats->bias_cycles += ws_read_cycles() - stage_start;

    stage_start = ws_read_cycles();
    gelu_bf16(&hidden[0][0], TINY_BERT_MAX_HIDDEN_DIM,
              &act[0][0], TINY_BERT_MAX_HIDDEN_DIM, tc->seq_len, tc->hidden_dim);
    stats->gelu_cycles += ws_read_cycles() - stage_start;

    if (rc == WS_GEMM_OK) {
      rc = hw_gemm(&act[0][0], TINY_BERT_MAX_HIDDEN_DIM, w2, tc->d_model,
                   &ffn_out[0][0], TINY_BERT_MAX_D_MODEL,
                   tc->seq_len, tc->d_model, tc->hidden_dim, gemm_workspace,
                   &stats->mlp_fc2_cycles);
    }

    stage_start = ws_read_cycles();
    add_bias_inplace_bf16(&ffn_out[0][0], TINY_BERT_MAX_D_MODEL,
                          layer_vec(tc->b2, layer, tc->d_model), tc->seq_len, tc->d_model);
    stats->bias_cycles += ws_read_cycles() - stage_start;

    stage_start = ws_read_cycles();
    residual_add_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL,
                      &ffn_out[0][0], TINY_BERT_MAX_D_MODEL,
                      &res[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layernorm_bf16(&res[0][0], TINY_BERT_MAX_D_MODEL, ffn_ln_gamma, ffn_ln_beta,
                   &x[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->ln_residual_cycles += ws_read_cycles() - stage_start;

    if (rc != WS_GEMM_OK) {
      break;
    }
  }

  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(&x[0][0], TINY_BERT_MAX_D_MODEL, tc->pool_w, tc->d_model,
                 pool, tc->d_model, 1, tc->d_model, tc->d_model,
                 gemm_workspace, &stats->pooler_cycles);
  }
  stage_start = ws_read_cycles();
  add_bias_inplace_bf16(pool, tc->d_model, tc->pool_b, 1, tc->d_model);
  stats->bias_cycles += ws_read_cycles() - stage_start;

  stage_start = ws_read_cycles();
  tanh_bf16_vec(pool, tc->d_model);
  stats->tanh_cycles += ws_read_cycles() - stage_start;

  if (rc == WS_GEMM_OK) {
    rc = hw_gemm(pool, tc->d_model, tc->classifier_w, tc->num_classes,
                 logits, tc->num_classes, 1, tc->num_classes, tc->d_model,
                 gemm_workspace, &stats->classifier_cycles);
  }
  stage_start = ws_read_cycles();
  add_bias_inplace_bf16(logits, tc->num_classes, tc->classifier_b, 1, tc->num_classes);
  stats->bias_cycles += ws_read_cycles() - stage_start;

  return rc;
}

static float max_diff_logits(
    const uint16_t *expected,
    const uint16_t *actual,
    int count,
    int *mismatches,
    uint32_t tolerance_x100000,
    const char *name) {
  float max_abs_diff = 0.0f;
  *mismatches = 0;
  for (int i = 0; i < count; i++) {
    const float ref = bf16_to_float(expected[i]);
    const float got = bf16_to_float(actual[i]);
    const float diff = fabsf(got - ref);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
    }
    if ((uint32_t)(diff * 100000.0f) > tolerance_x100000) {
      if (*mismatches < 8) {
        printf("Mismatch case=%s logit=%d ref=", name, i);
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
  return max_abs_diff;
}

static uint64_t hw_stats_total_cycles(const tiny_bert_hw_stats_t *stats) {
  return stats->embedding_cycles + stats->qkv_proj_cycles + stats->attn_e2e_cycles +
         stats->out_proj_cycles + stats->mlp_fc1_cycles + stats->mlp_fc2_cycles +
         stats->pooler_cycles + stats->classifier_cycles + stats->bias_cycles +
         stats->ln_residual_cycles + stats->gelu_cycles + stats->tanh_cycles;
}

int main(void) {
  const ws_gemm_workspace_t gemm_workspace =
      ws_gemm8_make_workspace(gemm_a_tiles, gemm_b_tiles, gemm_c_words,
                              TINY_BERT_MAX_SEQ_LEN,
                              TINY_BERT_MAX_HIDDEN_DIM,
                              TINY_BERT_MAX_HIDDEN_DIM);
  const fpga_safe_attention_workspace_t attn_workspace =
      fpga_safe_attention_make_workspace(q_tiles, k_tiles, v_tiles, attn_out_words,
                                         TINY_BERT_MAX_SEQ_LEN,
                                         TINY_BERT_MAX_SEQ_LEN,
                                         TINY_BERT_MAX_HEAD_DIM,
                                         TINY_BERT_MAX_HEAD_DIM);

  printf("=== Girdap Tiny-BERT Inference Test ===\n");
  printf("mode: %s  hardware: matmul=%d attention=%d softmax=%d\n",
         GIRDAP_BENCHMARK_MODE, GIRDAP_HW_MATMUL, GIRDAP_HW_ATTENTION,
         GIRDAP_HW_SOFTMAX);
  printf("opcodes: matmul=%d attention=%d softmax=%d\n",
         SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);
  printf("RUN_INFO_JSON {\"workload\":\"tiny-bert\",\"mode\":\"%s\",\"matmul_hw\":%d,\"attention_hw\":%d,\"softmax_hw\":%d,\"matmul_opcode\":%d,\"attention_opcode\":%d,\"softmax_opcode\":%d}\n",
         GIRDAP_BENCHMARK_MODE, GIRDAP_HW_MATMUL, GIRDAP_HW_ATTENTION,
         GIRDAP_HW_SOFTMAX, SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);

  int total_mismatches = 0;
  for (int i = 0; i < tiny_bert_case_count; i++) {
    const tiny_bert_case_t *tc = &tiny_bert_cases[i];

    const uint64_t cpu_start = ws_read_cycles();
    sw_tiny_bert_forward(tc);
    const uint64_t cpu_cycles = ws_read_cycles() - cpu_start;

    int cpu_mismatches = 0;
    const float cpu_ref_diff = max_diff_logits(tc->expected_logits, sw_logits,
                                               tc->num_classes, &cpu_mismatches,
                                               tc->tolerance_x100000, tc->name);

    tiny_bert_hw_stats_t stats;
    const int hw_rc = hw_tiny_bert_forward(tc, &gemm_workspace, &attn_workspace, &stats);
    const uint64_t hw_e2e_cycles = hw_stats_total_cycles(&stats);
    const uint64_t speedup_x100 =
        hw_e2e_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_e2e_cycles;

    int mismatches = 1;
    float hw_diff = 0.0f;
    if (hw_rc == WS_GEMM_OK) {
      hw_diff = max_diff_logits(tc->expected_logits, logits, tc->num_classes,
                                &mismatches, tc->tolerance_x100000, tc->name);
    }

    printf("\n[Case %d] %s\n", i, tc->name);
    printf("  shape : seq=%d d_model=%d heads=%d head_dim=%d hidden=%d layers=%d vocab=%d classes=%d\n",
           tc->seq_len, tc->d_model, tc->n_heads, tc->head_dim,
           tc->hidden_dim, tc->n_layers, tc->vocab_size, tc->num_classes);
    printf("  mode  : %s  opcodes: matmul=%d attention=%d softmax=%d\n",
           GIRDAP_BENCHMARK_MODE, SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);
    printf("  status: %s  hw_rc=%d raw_hw_rc=%lu mismatches=%d\n",
           (hw_rc == WS_GEMM_OK && mismatches == 0) ? "PASS" : "FAIL",
           hw_rc, (unsigned long)stats.raw_hw_rc, mismatches);
    printf("  cycles: cpu=%lu hw=%lu speedup=%lu.%02lux\n",
           (unsigned long)cpu_cycles, (unsigned long)hw_e2e_cycles,
           (unsigned long)(speedup_x100 / 100u), (unsigned long)(speedup_x100 % 100u));
    printf("          embedding=%lu qkv=%lu attn=%lu attn_accel=%lu out=%lu fc1=%lu fc2=%lu pooler=%lu classifier=%lu\n",
           (unsigned long)stats.embedding_cycles,
           (unsigned long)stats.qkv_proj_cycles,
           (unsigned long)stats.attn_e2e_cycles,
           (unsigned long)stats.attn_accel_cycles,
           (unsigned long)stats.out_proj_cycles,
           (unsigned long)stats.mlp_fc1_cycles,
           (unsigned long)stats.mlp_fc2_cycles,
           (unsigned long)stats.pooler_cycles,
           (unsigned long)stats.classifier_cycles);
    printf("          bias=%lu ln_residual=%lu gelu=%lu tanh=%lu\n",
           (unsigned long)stats.bias_cycles,
           (unsigned long)stats.ln_residual_cycles,
           (unsigned long)stats.gelu_cycles,
           (unsigned long)stats.tanh_cycles);
    printf("  error : hw_max_abs=");
    print_float_value(hw_diff);
    printf(" cpu_ref_max_abs=");
    print_float_value(cpu_ref_diff);
    printf("\n");

    print_solution_sample_chunk(i, tc->name, "encoder_output", &sw_x[0][0],
                                TINY_BERT_MAX_D_MODEL, &x[0][0],
                                TINY_BERT_MAX_D_MODEL, tc->seq_len,
                                tc->d_model, hw_rc);
    print_solution_sample_chunk(i, tc->name, "logits", sw_logits,
                                tc->num_classes, logits, tc->num_classes, 1,
                                tc->num_classes, hw_rc);
    total_mismatches += mismatches;

    printf("RESULT_JSON {\"workload\":\"tiny-bert\",\"case\":%d,\"name\":\"%s\",\"status\":\"%s\",\"hw_rc\":%d,\"raw_hw_rc\":%lu,\"mismatches\":%d,\"cpu_cycles\":%lu,\"hw_cycles\":%lu,\"speedup_x100\":%lu,\"hw_max_abs_diff_x100000\":%lu,\"cpu_ref_max_abs_diff_x100000\":%lu}\n",
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
    printf("PASS: Girdap Tiny-BERT inference matched PyTorch tolerance.\n");
  } else {
    printf("FAIL: total Tiny-BERT mismatches = %d\n", total_mismatches);
  }
  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/tiny_bert_cases.c"
