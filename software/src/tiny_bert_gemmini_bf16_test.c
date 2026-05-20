// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026

#include "bfloat16_utils.h"
#include "generated/tiny_bert_cases.h"
#include "include/gemmini_testutils.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LN_EPSILON 0.00001f
#define TINY_BERT_ERR_MASK_UNSUPPORTED 10
#define TINY_BERT_CLASSIFIER_STRIDE \
  ((TINY_BERT_MAX_NUM_CLASSES < DIM) ? DIM : TINY_BERT_MAX_NUM_CLASSES)

#define GEMMINI_TINY_BERT_FLOW "gemmini-matmul-only"

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

static elem_t x[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t q_proj[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t k_proj[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t v_proj[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t attn_probs[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_SEQ_LEN] row_align(1);
static elem_t context[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t attn_out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t res[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t hidden[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_HIDDEN_DIM] row_align(1);
static elem_t act[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_HIDDEN_DIM] row_align(1);
static elem_t ffn_out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t pool[TINY_BERT_MAX_D_MODEL] row_align(1);
static elem_t logits[TINY_BERT_MAX_NUM_CLASSES] row_align(1);
static elem_t classifier_out[TINY_BERT_CLASSIFIER_STRIDE] row_align(1);

static acc_t bias_d_model[TINY_BERT_MAX_D_MODEL] row_align_acc(1);
static acc_t bias_hidden[TINY_BERT_MAX_HIDDEN_DIM] row_align_acc(1);
static acc_t bias_classes[TINY_BERT_CLASSIFIER_STRIDE] row_align_acc(1);

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
  print_float_inline(value);
}

static int sample_chunk_start(const char *name, int rows, int cols) {
  uint32_t hash = 2166136261u;
  for (const char *p = name; *p != '\0'; p++) {
    hash ^= (uint8_t)(*p);
    hash *= 16777619u;
  }
  const int total = rows * cols;
  if (total <= 10) {
    return 0;
  }
  return (int)(hash % (uint32_t)(total - 10));
}

static void print_solution_sample_chunk(
    int case_idx,
    const char *case_name,
    const char *tensor_name,
    const uint16_t *ref,
    int ref_ld,
    const uint16_t *got,
    int got_ld,
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
    const int r = idx / cols;
    const int c = idx % cols;
    printf(" ");
    print_float_inline(bf16_to_float(tensor_at(ref, r, c, ref_ld)));
  }
  printf("\n    hw :");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int r = idx / cols;
    const int c = idx % cols;
    printf(" ");
    print_float_inline(bf16_to_float(tensor_at(got, r, c, got_ld)));
  }
  printf("\n    diff:");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int r = idx / cols;
    const int c = idx % cols;
    const float a = bf16_to_float(tensor_at(ref, r, c, ref_ld));
    const float b = bf16_to_float(tensor_at(got, r, c, got_ld));
    printf(" ");
    print_float_inline(fabsf(a - b));
  }
  printf("\n");

  printf("SAMPLE_JSON {\"workload\":\"tiny-bert\",\"case\":%d,\"name\":\"%s\",\"tensor\":\"%s\",\"start\":%d,\"count\":%d,\"total\":%d,\"hw_rc\":%d,\"sw\":[",
         case_idx, case_name, tensor_name, start, count, total, hw_rc);
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int r = idx / cols;
    const int c = idx % cols;
    if (i != 0) {
      printf(",");
    }
    print_float_value(bf16_to_float(tensor_at(ref, r, c, ref_ld)));
  }
  printf("],\"hw\":[");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int r = idx / cols;
    const int c = idx % cols;
    if (i != 0) {
      printf(",");
    }
    print_float_value(bf16_to_float(tensor_at(got, r, c, got_ld)));
  }
  printf("],\"diff\":[");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const int r = idx / cols;
    const int c = idx % cols;
    const float a = bf16_to_float(tensor_at(ref, r, c, ref_ld));
    const float b = bf16_to_float(tensor_at(got, r, c, got_ld));
    if (i != 0) {
      printf(",");
    }
    print_float_value(fabsf(a - b));
  }
  printf("]}\n");
}

static inline const uint16_t *layer_vec(const uint16_t *base, int layer, int d_model) {
  return base + (size_t)layer * d_model;
}

static inline const uint16_t *layer_hidden_vec(const uint16_t *base, int layer, int hidden_dim) {
  return base + (size_t)layer * hidden_dim;
}

static inline const uint16_t *layer_w_dxd(const uint16_t *base, int layer, int d_model) {
  return base + (size_t)layer * d_model * d_model;
}

static inline const uint16_t *layer_w_dxh(const uint16_t *base, int layer, int d_model, int hidden_dim) {
  return base + (size_t)layer * d_model * hidden_dim;
}

static inline const uint16_t *layer_w_hxd(const uint16_t *base, int layer, int hidden_dim, int d_model) {
  return base + (size_t)layer * hidden_dim * d_model;
}

static int mask_all_valid(const tiny_bert_case_t *tc) {
  for (int i = 0; i < tc->seq_len; i++) {
    if (tc->attention_mask[i] == 0) {
      return 0;
    }
  }
  return 1;
}

static void bf16_bias_to_acc(const uint16_t *src, acc_t *dst, int count) {
  for (int i = 0; i < count; i++) {
    dst[i] = (acc_t)bf16_to_float(src[i]);
  }
}

static void layernorm_bf16(
    const uint16_t *input,
    int input_ld,
    const uint16_t *gamma,
    const uint16_t *beta,
    uint16_t *output,
    int output_ld,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    double mean = 0.0;
    for (int c = 0; c < cols; c++) {
      mean += (double)bf16_to_float(input[r * input_ld + c]);
    }
    mean /= (double)cols;

    double var = 0.0;
    for (int c = 0; c < cols; c++) {
      const double d = (double)bf16_to_float(input[r * input_ld + c]) - mean;
      var += d * d;
    }
    var /= (double)cols;
    const double inv_std = 1.0 / sqrt(var + LN_EPSILON);

    for (int c = 0; c < cols; c++) {
      const double norm = ((double)bf16_to_float(input[r * input_ld + c]) - mean) * inv_std;
      const double scaled =
          norm * (double)bf16_to_float(gamma[c]) + (double)bf16_to_float(beta[c]);
      output[r * output_ld + c] = float_to_bf16((float)scaled);
    }
  }
}

static void embeddings_bf16(
    const tiny_bert_case_t *tc,
    uint16_t out[TINY_BERT_MAX_SEQ_LEN][TINY_BERT_MAX_D_MODEL]) {
  for (int r = 0; r < tc->seq_len; r++) {
    const int token = tc->token_ids[r];
    const int type = tc->type_ids[r];
    for (int c = 0; c < tc->d_model; c++) {
      const float word = bf16_to_float(tc->word_embeddings[token * tc->d_model + c]);
      const float pos = bf16_to_float(tc->position_embeddings[r * tc->d_model + c]);
      const float typ = bf16_to_float(tc->token_type_embeddings[type * tc->d_model + c]);
      out[r][c] = float_to_bf16(word + pos + typ);
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
      double acc = bias ? (double)bf16_to_float(bias[j]) : 0.0;
      for (int k = 0; k < K; k++) {
        acc += (double)bf16_to_float(A[i * lda + k]) *
               (double)bf16_to_float(B[k * ldb + j]);
      }
      C[i * ldc + j] = float_to_bf16((float)acc);
    }
  }
}

static void residual_add_bf16(
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
      const float sum = bf16_to_float(a[r * lda + c]) + bf16_to_float(b[r * ldb + c]);
      out[r * ldout + c] = float_to_bf16(sum);
    }
  }
}

static void gelu_bf16(
    const uint16_t *input,
    int input_ld,
    uint16_t *output,
    int output_ld,
    int rows,
    int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const float xval = bf16_to_float(input[r * input_ld + c]);
      const float y = 0.5f * xval *
                      (1.0f + tanhf(0.7978845608f * (xval + 0.044715f * xval * xval * xval)));
      output[r * output_ld + c] = float_to_bf16(y);
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

static int gemmini_matmul_bias_bf16(
    const uint16_t *A,
    int lda,
    const uint16_t *B,
    int ldb,
    const acc_t *bias,
    uint16_t *C,
    int ldc,
    int M,
    int N,
    int K,
    uint64_t *cycles) {
  gemmini_fence();
  const uint64_t start = read_cycles();
  tiled_matmul_auto(
      M,
      N,
      K,
      (const elem_t *)A,
      (const elem_t *)B,
      bias,
      (elem_t *)C,
      lda,
      ldb,
      0,
      ldc,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      NO_ACTIVATION,
      ACC_SCALE_IDENTITY,
      0,
      false,
      false,
      false,
      false,
      false,
      0,
      WS);
  gemmini_fence();
  *cycles += read_cycles() - start;
  return 0;
}

static int cpu_attention_heads_from_gemmini_qkv(
    const tiny_bert_case_t *tc,
    tiny_bert_hw_stats_t *stats) {
  const uint64_t total_start = read_cycles();
  const float scale = bf16_to_float(tc->scale_bf16);

  for (int h = 0; h < tc->n_heads; h++) {
    const int base = h * tc->head_dim;

    uint64_t stage_start = read_cycles();
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int kv = 0; kv < tc->seq_len; kv++) {
        double dot = 0.0;
        for (int d = 0; d < tc->head_dim; d++) {
          dot += (double)bf16_to_float(q_proj[qr][base + d]) *
                 (double)bf16_to_float(k_proj[kv][base + d]);
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
    stats->attn_score_cycles += read_cycles() - stage_start;

    stage_start = read_cycles();
    for (int qr = 0; qr < tc->seq_len; qr++) {
      for (int vc = 0; vc < tc->head_dim; vc++) {
        double acc = 0.0;
        for (int kv = 0; kv < tc->seq_len; kv++) {
          acc += (double)sw_probs[qr][kv] *
                 (double)bf16_to_float(v_proj[kv][base + vc]);
        }
        context[qr][base + vc] = float_to_bf16((float)acc);
      }
    }
    stats->attn_value_cycles += read_cycles() - stage_start;
  }

  stats->attn_e2e_cycles += read_cycles() - total_start;
  return 0;
}

static int gemmini_tiny_bert_forward(const tiny_bert_case_t *tc, tiny_bert_hw_stats_t *stats) {
  memset(stats, 0, sizeof(*stats));

  uint64_t stage_start = read_cycles();
  embeddings_bf16(tc, (uint16_t (*)[TINY_BERT_MAX_D_MODEL])x);
  stats->embedding_cycles += read_cycles() - stage_start;

  int rc = 0;
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

    bf16_bias_to_acc(layer_vec(tc->bq, layer, tc->d_model), bias_d_model, tc->d_model);
    printf("    gemmini: layer=%d q_proj_start\n", layer);
    rc = gemmini_matmul_bias_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL, wq, tc->d_model,
                                  bias_d_model, &q_proj[0][0], TINY_BERT_MAX_D_MODEL,
                                  tc->seq_len, tc->d_model, tc->d_model,
                                  &stats->qkv_proj_cycles);
    printf("    gemmini: layer=%d q_proj_done rc=%d\n", layer, rc);
    bf16_bias_to_acc(layer_vec(tc->bk, layer, tc->d_model), bias_d_model, tc->d_model);
    if (rc == 0) {
      printf("    gemmini: layer=%d k_proj_start\n", layer);
      rc = gemmini_matmul_bias_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL, wk, tc->d_model,
                                    bias_d_model, &k_proj[0][0], TINY_BERT_MAX_D_MODEL,
                                    tc->seq_len, tc->d_model, tc->d_model,
                                    &stats->qkv_proj_cycles);
      printf("    gemmini: layer=%d k_proj_done rc=%d\n", layer, rc);
    }
    bf16_bias_to_acc(layer_vec(tc->bv, layer, tc->d_model), bias_d_model, tc->d_model);
    if (rc == 0) {
      printf("    gemmini: layer=%d v_proj_start\n", layer);
      rc = gemmini_matmul_bias_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL, wv, tc->d_model,
                                    bias_d_model, &v_proj[0][0], TINY_BERT_MAX_D_MODEL,
                                    tc->seq_len, tc->d_model, tc->d_model,
                                    &stats->qkv_proj_cycles);
      printf("    gemmini: layer=%d v_proj_done rc=%d\n", layer, rc);
    }

    if (rc == 0) {
      printf("    cpu: layer=%d attention_start\n", layer);
      rc = cpu_attention_heads_from_gemmini_qkv(tc, stats);
      printf("    cpu: layer=%d attention_done rc=%d\n", layer, rc);
    }

    bf16_bias_to_acc(layer_vec(tc->bo, layer, tc->d_model), bias_d_model, tc->d_model);
    if (rc == 0) {
      printf("    gemmini: layer=%d out_proj_start\n", layer);
      rc = gemmini_matmul_bias_bf16(&context[0][0], TINY_BERT_MAX_D_MODEL, wo, tc->d_model,
                                    bias_d_model, &attn_out[0][0], TINY_BERT_MAX_D_MODEL,
                                    tc->seq_len, tc->d_model, tc->d_model,
                                    &stats->out_proj_cycles);
      printf("    gemmini: layer=%d out_proj_done rc=%d\n", layer, rc);
    }

    stage_start = read_cycles();
    residual_add_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL,
                      &attn_out[0][0], TINY_BERT_MAX_D_MODEL,
                      &res[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layernorm_bf16(&res[0][0], TINY_BERT_MAX_D_MODEL, attn_ln_gamma, attn_ln_beta,
                   &x[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->ln_residual_cycles += read_cycles() - stage_start;

    bf16_bias_to_acc(layer_hidden_vec(tc->b1, layer, tc->hidden_dim), bias_hidden, tc->hidden_dim);
    if (rc == 0) {
      printf("    gemmini: layer=%d mlp_fc1_start\n", layer);
      rc = gemmini_matmul_bias_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL, w1, tc->hidden_dim,
                                    bias_hidden, &hidden[0][0], TINY_BERT_MAX_HIDDEN_DIM,
                                    tc->seq_len, tc->hidden_dim, tc->d_model,
                                    &stats->mlp_fc1_cycles);
      printf("    gemmini: layer=%d mlp_fc1_done rc=%d\n", layer, rc);
    }

    stage_start = read_cycles();
    gelu_bf16(&hidden[0][0], TINY_BERT_MAX_HIDDEN_DIM,
              &act[0][0], TINY_BERT_MAX_HIDDEN_DIM, tc->seq_len, tc->hidden_dim);
    stats->gelu_cycles += read_cycles() - stage_start;

    bf16_bias_to_acc(layer_vec(tc->b2, layer, tc->d_model), bias_d_model, tc->d_model);
    if (rc == 0) {
      printf("    gemmini: layer=%d mlp_fc2_start\n", layer);
      rc = gemmini_matmul_bias_bf16(&act[0][0], TINY_BERT_MAX_HIDDEN_DIM, w2, tc->d_model,
                                    bias_d_model, &ffn_out[0][0], TINY_BERT_MAX_D_MODEL,
                                    tc->seq_len, tc->d_model, tc->hidden_dim,
                                    &stats->mlp_fc2_cycles);
      printf("    gemmini: layer=%d mlp_fc2_done rc=%d\n", layer, rc);
    }

    stage_start = read_cycles();
    residual_add_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL,
                      &ffn_out[0][0], TINY_BERT_MAX_D_MODEL,
                      &res[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    layernorm_bf16(&res[0][0], TINY_BERT_MAX_D_MODEL, ffn_ln_gamma, ffn_ln_beta,
                   &x[0][0], TINY_BERT_MAX_D_MODEL, tc->seq_len, tc->d_model);
    stats->ln_residual_cycles += read_cycles() - stage_start;

    if (rc != 0) {
      break;
    }
  }

  bf16_bias_to_acc(tc->pool_b, bias_d_model, tc->d_model);
  if (rc == 0) {
    printf("    gemmini: pooler_start\n");
    rc = gemmini_matmul_bias_bf16(&x[0][0], TINY_BERT_MAX_D_MODEL, tc->pool_w, tc->d_model,
                                  bias_d_model, pool, tc->d_model,
                                  1, tc->d_model, tc->d_model,
                                  &stats->pooler_cycles);
    printf("    gemmini: pooler_done rc=%d\n", rc);
  }

  stage_start = read_cycles();
  tanh_bf16_vec(pool, tc->d_model);
  stats->tanh_cycles += read_cycles() - stage_start;

  memset(classifier_out, 0, sizeof(classifier_out));
  memset(bias_classes, 0, sizeof(bias_classes));
  bf16_bias_to_acc(tc->classifier_b, bias_classes, tc->num_classes);
  if (rc == 0) {
    printf("    gemmini: classifier_start\n");
    rc = gemmini_matmul_bias_bf16(pool, tc->d_model, tc->classifier_w, tc->num_classes,
                                  bias_classes, classifier_out, TINY_BERT_CLASSIFIER_STRIDE,
                                  1, tc->num_classes, tc->d_model,
                                  &stats->classifier_cycles);
    printf("    gemmini: classifier_done rc=%d\n", rc);
    if (rc == 0) {
      for (int i = 0; i < tc->num_classes; i++) {
        logits[i] = classifier_out[i];
      }
    }
  }

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
  gemmini_flush(0);

  printf("=== Gemmini Tiny-BERT BF16 Inference Test ===\n");
  printf("mode: gemmini-bf16  flow=%s\n", GEMMINI_TINY_BERT_FLOW);
  printf("note: Gemmini accelerates GEMM layers only; attention score/softmax/context runs on CPU.\n");
  printf("gemmini: DIM=%d elem_bits=%d acc_bits=%d custom=%d\n",
         DIM, ELEM_T_EXP_BITS + ELEM_T_SIG_BITS, ACC_T_EXP_BITS + ACC_T_SIG_BITS,
         XCUSTOM_ACC);
  printf("RUN_INFO_JSON {\"workload\":\"tiny-bert\",\"mode\":\"gemmini-bf16\",\"backend\":\"gemmini-bf16\",\"flow\":\"%s\",\"dim\":%d,\"elem_bits\":%d,\"acc_bits\":%d,\"opcode\":%d,\"gemmini_attention\":0}\n",
         GEMMINI_TINY_BERT_FLOW,
         DIM, ELEM_T_EXP_BITS + ELEM_T_SIG_BITS,
         ACC_T_EXP_BITS + ACC_T_SIG_BITS, XCUSTOM_ACC);

  int total_mismatches = 0;
  for (int i = 0; i < tiny_bert_case_count; i++) {
    const tiny_bert_case_t *tc = &tiny_bert_cases[i];

    printf("\n[Case %d] %s\n", i, tc->name);
    printf("  shape : seq=%d d_model=%d heads=%d head_dim=%d hidden=%d layers=%d vocab=%d classes=%d\n",
           tc->seq_len, tc->d_model, tc->n_heads, tc->head_dim,
           tc->hidden_dim, tc->n_layers, tc->vocab_size, tc->num_classes);
    printf("  phase : cpu_ref_start\n");
    const uint64_t cpu_start = read_cycles();
    sw_tiny_bert_forward(tc);
    const uint64_t cpu_cycles = read_cycles() - cpu_start;
    printf("  phase : cpu_ref_done cycles=%lu\n", (unsigned long)cpu_cycles);

    int cpu_mismatches = 0;
    const float cpu_ref_diff = max_diff_logits(tc->expected_logits, sw_logits,
                                               tc->num_classes, &cpu_mismatches,
                                               tc->tolerance_x100000, tc->name);

    printf("  phase : gemmini_start\n");
    tiny_bert_hw_stats_t stats;
    const int hw_rc = gemmini_tiny_bert_forward(tc, &stats);
    printf("  phase : gemmini_done hw_rc=%d\n", hw_rc);
    const uint64_t hw_e2e_cycles = hw_stats_total_cycles(&stats);
    const uint64_t speedup_x100 =
        hw_e2e_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_e2e_cycles;

    int mismatches = 1;
    float hw_diff = 0.0f;
    if (hw_rc == 0) {
      hw_diff = max_diff_logits(tc->expected_logits, logits, tc->num_classes,
                                &mismatches, tc->tolerance_x100000, tc->name);
    }

    printf("  mode  : gemmini-bf16  backend=gemmini-bf16 flow=%s dim=%d opcode=%d\n",
           GEMMINI_TINY_BERT_FLOW, DIM, XCUSTOM_ACC);
    printf("  status: %s  hw_rc=%d raw_hw_rc=%lu mismatches=%d\n",
           (hw_rc == 0 && mismatches == 0) ? "PASS" : "FAIL",
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

    printf("RESULT_JSON {\"workload\":\"tiny-bert\",\"case\":%d,\"name\":\"%s\",\"status\":\"%s\",\"backend\":\"gemmini-bf16\",\"hw_rc\":%d,\"raw_hw_rc\":%lu,\"mismatches\":%d,\"cpu_cycles\":%lu,\"hw_cycles\":%lu,\"speedup_x100\":%lu,\"hw_max_abs_diff_x100000\":%lu,\"cpu_ref_max_abs_diff_x100000\":%lu}\n",
           i, tc->name,
           (hw_rc == 0 && mismatches == 0) ? "PASS" : "FAIL",
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
    printf("PASS: gemmini-bf16 Tiny-BERT inference matched PyTorch tolerance.\n");
  } else {
    printf("FAIL: total Tiny-BERT mismatches = %d\n", total_mismatches);
  }
  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/tiny_bert_cases.c"
