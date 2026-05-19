#include "bfloat16_utils.h"
#include "fpga_safe_attention.h"
#include "generated/attention_operator_cases.h"
#include "online_softmax.h"
#include "ws_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef GIRDAP_HW_MATMUL
#define GIRDAP_HW_MATMUL 0
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

#define ATTN_OPERATOR_MAX_GEMM_N \
  ((ATTN_OPERATOR_MAX_KV_ROWS > ATTN_OPERATOR_MAX_VALUE_COLS) ? \
       ATTN_OPERATOR_MAX_KV_ROWS : ATTN_OPERATOR_MAX_VALUE_COLS)
#define ATTN_OPERATOR_MAX_GEMM_K \
  ((ATTN_OPERATOR_MAX_D_K > ATTN_OPERATOR_MAX_KV_ROWS) ? \
       ATTN_OPERATOR_MAX_D_K : ATTN_OPERATOR_MAX_KV_ROWS)

static uint16_t out_matrix[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_VALUE_COLS]
    __attribute__((aligned(64)));
static uint16_t score_bf16[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_KV_ROWS]
    __attribute__((aligned(64)));
static uint16_t probs_bf16[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_KV_ROWS]
    __attribute__((aligned(64)));
static uint16_t k_t[ATTN_OPERATOR_MAX_D_K][ATTN_OPERATOR_MAX_KV_ROWS]
    __attribute__((aligned(64)));
static float sw_scores[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_KV_ROWS];
static float sw_probs[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_KV_ROWS];
static float sw_output[ATTN_OPERATOR_MAX_Q_ROWS][ATTN_OPERATOR_MAX_VALUE_COLS];

static uint64_t gemm_a_tiles[WS_GEMM8_A_TILE_WORDS(ATTN_OPERATOR_MAX_Q_ROWS, ATTN_OPERATOR_MAX_GEMM_K)]
    __attribute__((aligned(64)));
static uint64_t gemm_b_tiles[WS_GEMM8_B_TILE_WORDS(ATTN_OPERATOR_MAX_GEMM_N, ATTN_OPERATOR_MAX_GEMM_K)]
    __attribute__((aligned(64)));
static uint64_t gemm_c_words[WS_GEMM8_C_TILE_WORDS] __attribute__((aligned(64)));

static uint64_t q_tiles[FPGA_SAFE_ATTN_Q_TILE_WORDS(ATTN_OPERATOR_MAX_Q_ROWS, ATTN_OPERATOR_MAX_D_K)]
    __attribute__((aligned(64)));
static uint64_t k_tiles[FPGA_SAFE_ATTN_K_TILE_WORDS(ATTN_OPERATOR_MAX_KV_ROWS, ATTN_OPERATOR_MAX_D_K)]
    __attribute__((aligned(64)));
static uint64_t v_tiles[FPGA_SAFE_ATTN_V_TILE_WORDS(ATTN_OPERATOR_MAX_VALUE_COLS, ATTN_OPERATOR_MAX_KV_ROWS)]
    __attribute__((aligned(64)));
static uint64_t out_words[FPGA_SAFE_ATTN_OUT_WORDS] __attribute__((aligned(64)));

static uint16_t tensor_at(
    const uint16_t *tensor,
    int row,
    int col,
    int ld) {
  return tensor[row * ld + col];
}

typedef struct {
  uint64_t qk_cycles;
  uint64_t softmax_cycles;
  uint64_t pv_cycles;
  uint64_t q_pack_cycles;
  uint64_t k_pack_cycles;
  uint64_t v_pack_cycles;
  uint64_t attn_accel_cycles;
  uint64_t score_cycles;
  uint64_t value_cycles;
  uint64_t copy_out_cycles;
  uint64_t raw_hw_rc;
} attention_operator_hw_stats_t;

static void print_float_value(float value) {
  if (value < 0.0f) {
    printf("-");
    value = -value;
  }
  const int whole_part = (int)value;
  const int frac_part = (int)((value - (float)whole_part) * 10000.0f);
  printf("%d.%04d", whole_part, frac_part);
}

static void transpose_k(const attention_operator_case_t *tc) {
  for (int r = 0; r < tc->d_k; r++) {
    for (int c = 0; c < tc->kv_rows; c++) {
      k_t[r][c] = tensor_at(tc->k, c, r, tc->d_k);
    }
  }
}

static void scale_scores_inplace(const attention_operator_case_t *tc) {
  const float scale = bf16_to_float(tc->scale_bf16);
  for (int r = 0; r < tc->q_rows; r++) {
    for (int c = 0; c < tc->kv_rows; c++) {
      const float v = bf16_to_float(score_bf16[r][c]) * scale;
      score_bf16[r][c] = float_to_bf16(v);
    }
  }
}

static void sw_softmax_rows_bf16(const attention_operator_case_t *tc) {
  for (int qr = 0; qr < tc->q_rows; qr++) {
    float max_score = -INFINITY;
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      const float score = bf16_to_float(score_bf16[qr][kv]);
      if (score > max_score) {
        max_score = score;
      }
    }
    double denom = 0.0;
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      denom += exp((double)(bf16_to_float(score_bf16[qr][kv]) - max_score));
    }
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      const float prob =
          (float)(exp((double)(bf16_to_float(score_bf16[qr][kv]) - max_score)) / denom);
      probs_bf16[qr][kv] = float_to_bf16(prob);
    }
  }
}

static void sw_pv_bf16(const attention_operator_case_t *tc) {
  for (int qr = 0; qr < tc->q_rows; qr++) {
    for (int vc = 0; vc < tc->value_cols; vc++) {
      double acc = 0.0;
      for (int kv = 0; kv < tc->kv_rows; kv++) {
        acc += (double)bf16_to_float(probs_bf16[qr][kv]) *
               (double)bf16_to_float(tensor_at(tc->v, kv, vc, tc->value_cols));
      }
      out_matrix[qr][vc] = float_to_bf16((float)acc);
    }
  }
}

static int run_gemm(
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
  *cycles += ws_read_cycles() - start;
  return WS_GEMM_OK;
#endif
}

static void sw_attention_reference(const attention_operator_case_t *tc) {
  const float scale = bf16_to_float(tc->scale_bf16);

  for (int qr = 0; qr < tc->q_rows; qr++) {
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      double dot = 0.0;
      for (int d = 0; d < tc->d_k; d++) {
        dot += (double)bf16_to_float(tensor_at(tc->q, qr, d, tc->d_k)) *
               (double)bf16_to_float(tensor_at(tc->k, kv, d, tc->d_k));
      }
      sw_scores[qr][kv] = (float)(dot * (double)scale);
    }
  }

  for (int qr = 0; qr < tc->q_rows; qr++) {
    float max_score = -INFINITY;
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      if (sw_scores[qr][kv] > max_score) {
        max_score = sw_scores[qr][kv];
      }
    }

    double denom = 0.0;
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      denom += exp((double)(sw_scores[qr][kv] - max_score));
    }
    for (int kv = 0; kv < tc->kv_rows; kv++) {
      sw_probs[qr][kv] =
          (float)(exp((double)(sw_scores[qr][kv] - max_score)) / denom);
    }
  }

  for (int qr = 0; qr < tc->q_rows; qr++) {
    for (int vc = 0; vc < tc->value_cols; vc++) {
      double acc = 0.0;
      for (int kv = 0; kv < tc->kv_rows; kv++) {
        acc += (double)sw_probs[qr][kv] *
               (double)bf16_to_float(tensor_at(tc->v, kv, vc, tc->value_cols));
      }
      sw_output[qr][vc] = (float)acc;
    }
  }
}

static int compare_bf16_output(
    const attention_operator_case_t *tc,
    float *max_abs_diff) {
  int mismatches = 0;
  *max_abs_diff = 0.0f;

  for (int r = 0; r < tc->q_rows; r++) {
    for (int c = 0; c < tc->value_cols; c++) {
      const int idx = r * tc->value_cols + c;
      const float ref = bf16_to_float(tc->expected[idx]);
      const float hw = bf16_to_float(out_matrix[r][c]);
      const float diff = fabsf(hw - ref);
      if (diff > *max_abs_diff) {
        *max_abs_diff = diff;
      }
      if ((uint32_t)(diff * 100000.0f) > tc->tolerance_x100000) {
        if (mismatches < 8) {
          printf("Mismatch case=%s row=%d col=%d ref=", tc->name, r, c);
          print_float_inline(ref);
          printf("hw=");
          print_float_inline(hw);
          printf("diff=");
          print_float_inline(diff);
          printf("\n");
        }
        mismatches++;
      }
    }
  }

  return mismatches;
}

static float max_cpu_vs_exported_diff(const attention_operator_case_t *tc) {
  float max_abs_diff = 0.0f;
  for (int r = 0; r < tc->q_rows; r++) {
    for (int c = 0; c < tc->value_cols; c++) {
      const int idx = r * tc->value_cols + c;
      const float ref = bf16_to_float(tc->expected[idx]);
      const float diff = fabsf(sw_output[r][c] - ref);
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
      }
    }
  }
  return max_abs_diff;
}

static int run_attention_operator(
    const attention_operator_case_t *tc,
    const ws_gemm_workspace_t *gemm_workspace,
    const fpga_safe_attention_workspace_t *attn_workspace,
    attention_operator_hw_stats_t *stats) {
  memset(stats, 0, sizeof(*stats));

#if GIRDAP_HW_ATTENTION
  fpga_safe_attention_stats_t attn_stats;
  memset(&attn_stats, 0, sizeof(attn_stats));
  const int rc = fpga_safe_attention_bf16_hwpack(
      tc->q,
      tc->d_k,
      tc->k,
      tc->d_k,
      tc->v,
      tc->value_cols,
      tc->q_rows,
      tc->kv_rows,
      tc->d_k,
      tc->value_cols,
      tc->scale_bf16,
      &out_matrix[0][0],
      ATTN_OPERATOR_MAX_VALUE_COLS,
      attn_workspace,
      &attn_stats);
  stats->q_pack_cycles = attn_stats.q_pack_cycles;
  stats->k_pack_cycles = attn_stats.k_pack_cycles;
  stats->v_pack_cycles = attn_stats.v_pack_cycles;
  stats->attn_accel_cycles = attn_stats.hw_e2e_cycles;
  stats->score_cycles = attn_stats.score_cycles;
  stats->value_cycles = attn_stats.value_cycles;
  stats->copy_out_cycles = attn_stats.copy_out_cycles;
  stats->raw_hw_rc = attn_stats.raw_hw_rc;
  return rc;
#else
  transpose_k(tc);

  int rc = run_gemm(
      tc->q,
      tc->d_k,
      &k_t[0][0],
      ATTN_OPERATOR_MAX_KV_ROWS,
      &score_bf16[0][0],
      ATTN_OPERATOR_MAX_KV_ROWS,
      tc->q_rows,
      tc->kv_rows,
      tc->d_k,
      gemm_workspace,
      &stats->qk_cycles);
  if (rc != WS_GEMM_OK) {
    return rc;
  }
  scale_scores_inplace(tc);

#if GIRDAP_HW_SOFTMAX
  online_softmax_stats_t softmax_stats;
  memset(&softmax_stats, 0, sizeof(softmax_stats));
  rc = online_softmax_bf16_rows(
      &score_bf16[0][0],
      tc->q_rows,
      tc->kv_rows,
      ATTN_OPERATOR_MAX_KV_ROWS,
      &probs_bf16[0][0],
      ATTN_OPERATOR_MAX_KV_ROWS,
      &softmax_stats);
  stats->softmax_cycles += softmax_stats.hw_e2e_cycles;
  stats->raw_hw_rc = (uint64_t)rc;
  if (rc != ONLINE_SOFTMAX_OK) {
    return rc;
  }
#else
  const uint64_t softmax_start = ws_read_cycles();
  sw_softmax_rows_bf16(tc);
  stats->softmax_cycles += ws_read_cycles() - softmax_start;
#endif

  rc = run_gemm(
      &probs_bf16[0][0],
      ATTN_OPERATOR_MAX_KV_ROWS,
      tc->v,
      tc->value_cols,
      &out_matrix[0][0],
      ATTN_OPERATOR_MAX_VALUE_COLS,
      tc->q_rows,
      tc->value_cols,
      tc->kv_rows,
      gemm_workspace,
      &stats->pv_cycles);
  return rc;
#endif
}

static uint64_t hw_stats_total_cycles(const attention_operator_hw_stats_t *stats) {
  return stats->qk_cycles + stats->softmax_cycles + stats->pv_cycles +
         stats->q_pack_cycles + stats->k_pack_cycles + stats->v_pack_cycles +
         stats->attn_accel_cycles + stats->copy_out_cycles;
}

static int sample_chunk_start(const char *name, int rows, int cols) {
  uint32_t hash = 2166136261u;
  for (const char *p = name; *p != '\0'; p++) {
    hash ^= (uint8_t)(*p);
    hash *= 16777619u;
  }
  hash ^= (uint32_t)(rows * 131 + cols * 17);
  const int total = rows * cols;
  return total <= 10 ? 0 : (int)(hash % (uint32_t)(total - 10 + 1));
}

static void print_sample_chunk(int case_index, const attention_operator_case_t *tc, int hw_rc) {
  const int total = tc->q_rows * tc->value_cols;
  const int count = total < 10 ? total : 10;
  const int start = sample_chunk_start(tc->name, tc->q_rows, tc->value_cols);

  printf("\n  sample output[start=%d count=%d total=%d]\n", start, count, total);
  printf("    sw :");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    printf(" ");
    print_float_value(sw_output[idx / tc->value_cols][idx % tc->value_cols]);
  }
  printf("\n    hw :");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    printf(" ");
    print_float_value(bf16_to_float(out_matrix[idx / tc->value_cols][idx % tc->value_cols]));
  }
  printf("\n    diff:");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const float sw = sw_output[idx / tc->value_cols][idx % tc->value_cols];
    const float hw = bf16_to_float(out_matrix[idx / tc->value_cols][idx % tc->value_cols]);
    printf(" ");
    print_float_value(fabsf(sw - hw));
  }
  printf("\n");

  printf("SAMPLE_JSON {\"workload\":\"attention-operator\",\"case\":%d,\"name\":\"%s\",\"tensor\":\"output\",\"start\":%d,\"count\":%d,\"total\":%d,\"hw_rc\":%d,\"sw\":[",
         case_index, tc->name, start, count, total, hw_rc);
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    if (i != 0) {
      printf(",");
    }
    print_float_value(sw_output[idx / tc->value_cols][idx % tc->value_cols]);
  }
  printf("],\"hw\":[");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    if (i != 0) {
      printf(",");
    }
    print_float_value(bf16_to_float(out_matrix[idx / tc->value_cols][idx % tc->value_cols]));
  }
  printf("],\"diff\":[");
  for (int i = 0; i < count; i++) {
    const int idx = start + i;
    const float sw = sw_output[idx / tc->value_cols][idx % tc->value_cols];
    const float hw = bf16_to_float(out_matrix[idx / tc->value_cols][idx % tc->value_cols]);
    if (i != 0) {
      printf(",");
    }
    print_float_value(fabsf(sw - hw));
  }
  printf("]}\n");
}

int main(void) {
  const ws_gemm_workspace_t gemm_workspace =
      ws_gemm8_make_workspace(gemm_a_tiles, gemm_b_tiles, gemm_c_words,
                              ATTN_OPERATOR_MAX_Q_ROWS,
                              ATTN_OPERATOR_MAX_GEMM_N,
                              ATTN_OPERATOR_MAX_GEMM_K);
  const fpga_safe_attention_workspace_t workspace =
      fpga_safe_attention_make_workspace(
          q_tiles,
          k_tiles,
          v_tiles,
          out_words,
          ATTN_OPERATOR_MAX_Q_ROWS,
          ATTN_OPERATOR_MAX_KV_ROWS,
          ATTN_OPERATOR_MAX_D_K,
          ATTN_OPERATOR_MAX_VALUE_COLS);

  printf("=== Girdap Attention Operator Test ===\n");
  printf("mode: %s  hardware: matmul=%d attention=%d softmax=%d\n",
         GIRDAP_BENCHMARK_MODE, GIRDAP_HW_MATMUL, GIRDAP_HW_ATTENTION,
         GIRDAP_HW_SOFTMAX);
  printf("opcodes: matmul=%d attention=%d softmax=%d\n",
         SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);
  printf("RUN_INFO_JSON {\"workload\":\"attention-operator\",\"mode\":\"%s\",\"matmul_hw\":%d,\"attention_hw\":%d,\"softmax_hw\":%d,\"matmul_opcode\":%d,\"attention_opcode\":%d,\"softmax_opcode\":%d}\n",
         GIRDAP_BENCHMARK_MODE, GIRDAP_HW_MATMUL, GIRDAP_HW_ATTENTION,
         GIRDAP_HW_SOFTMAX, SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);

  int total_mismatches = 0;
  for (int i = 0; i < attention_operator_case_count; i++) {
    const attention_operator_case_t *tc = &attention_operator_cases[i];
    memset(out_matrix, 0, sizeof(out_matrix));

    const uint64_t cpu_start = ws_read_cycles();
    sw_attention_reference(tc);
    const uint64_t cpu_cycles = ws_read_cycles() - cpu_start;
    const float cpu_ref_max_abs_diff = max_cpu_vs_exported_diff(tc);

    attention_operator_hw_stats_t stats;
    const int hw_rc = run_attention_operator(tc, &gemm_workspace, &workspace, &stats);

    float hw_max_abs_diff = 0.0f;
    const int mismatches =
        (hw_rc == FPGA_SAFE_ATTN_OK) ? compare_bf16_output(tc, &hw_max_abs_diff) : 1;
    total_mismatches += mismatches;

    const uint64_t hw_total_cycles = hw_stats_total_cycles(&stats);
    const uint64_t speedup_x100 =
        hw_total_cycles == 0 ? 0 : (cpu_cycles * 100u) / hw_total_cycles;

    printf("\n[Case %d] %s\n", i, tc->name);
    printf("  shape : q_rows=%d kv_rows=%d d_k=%d value_cols=%d\n",
           tc->q_rows, tc->kv_rows, tc->d_k, tc->value_cols);
    printf("  mode  : %s  opcodes: matmul=%d attention=%d softmax=%d\n",
           GIRDAP_BENCHMARK_MODE, SA_MATMUL_OPCODE, SA_ATTN_OPCODE, SOFTMAX_OPCODE);
    printf("  status: %s  hw_rc=%d raw_hw_rc=%lu mismatches=%d\n",
           (hw_rc == FPGA_SAFE_ATTN_OK && mismatches == 0) ? "PASS" : "FAIL",
           hw_rc, (unsigned long)stats.raw_hw_rc, mismatches);
    printf("  cycles: cpu=%lu hw=%lu speedup=%lu.%02lux\n",
           (unsigned long)cpu_cycles, (unsigned long)hw_total_cycles,
           (unsigned long)(speedup_x100 / 100u), (unsigned long)(speedup_x100 % 100u));
    printf("          qk=%lu softmax=%lu pv=%lu q_pack=%lu k_pack=%lu v_pack=%lu attn_accel=%lu score=%lu value=%lu copy=%lu\n",
           (unsigned long)stats.qk_cycles,
           (unsigned long)stats.softmax_cycles,
           (unsigned long)stats.pv_cycles,
           (unsigned long)stats.q_pack_cycles,
           (unsigned long)stats.k_pack_cycles,
           (unsigned long)stats.v_pack_cycles,
           (unsigned long)stats.attn_accel_cycles,
           (unsigned long)stats.score_cycles,
           (unsigned long)stats.value_cycles,
           (unsigned long)stats.copy_out_cycles);
    printf("  error : hw_max_abs=");
    print_float_value(hw_max_abs_diff);
    printf(" cpu_ref_max_abs=");
    print_float_value(cpu_ref_max_abs_diff);
    printf("\n");
    print_sample_chunk(i, tc, hw_rc);

    printf("RESULT_JSON {\"workload\":\"attention-operator\",\"case\":%d,\"name\":\"%s\",\"status\":\"%s\",\"hw_rc\":%d,\"raw_hw_rc\":%lu,\"mismatches\":%d,\"cpu_cycles\":%lu,\"hw_cycles\":%lu,\"speedup_x100\":%lu,\"hw_max_abs_diff_x100000\":%lu,\"cpu_ref_max_abs_diff_x100000\":%lu}\n",
           i, tc->name,
           (hw_rc == FPGA_SAFE_ATTN_OK && mismatches == 0) ? "PASS" : "FAIL",
           hw_rc,
           (unsigned long)stats.raw_hw_rc,
           mismatches,
           (unsigned long)cpu_cycles,
           (unsigned long)hw_total_cycles,
           (unsigned long)speedup_x100,
           (unsigned long)(hw_max_abs_diff * 100000.0f),
           (unsigned long)(cpu_ref_max_abs_diff * 100000.0f));
  }

  if (total_mismatches == 0) {
    printf("PASS: Girdap attention operator matched PyTorch tolerance.\n");
  } else {
    printf("FAIL: total mismatches = %d\n", total_mismatches);
  }

  return total_mismatches == 0 ? 0 : 1;
}

#include "generated/attention_operator_cases.c"
