#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <tiny-bert|vit|gpt2-prefill|attention-operator> <attention-matmul|attention-only|matmul-only|softmax-only> [dual-rocc|matmul-rocc|attention-rocc|softmax-rocc]" >&2
  exit 2
fi

BENCHMARK="$1"
HW_MODE="$2"
if [ "$#" -eq 3 ]; then
  HW_PROFILE="$3"
elif [ "${HW_MODE}" = "softmax-only" ]; then
  HW_PROFILE="softmax-rocc"
else
  HW_PROFILE="dual-rocc"
fi

PYTHON_BIN="${CY_DIR:-/home/ubuntu/chipyard-f2}/.conda-env/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

case "${BENCHMARK}" in
  tiny-bert)
    GENERATE_LABEL="PyTorch Tiny-BERT inference cases"
    GENERATE_CMD=("${PYTHON_BIN}" scripts/export_tiny_bert_cases.py)
    TARGET="build/tiny_bert_dual_rocc_test.riscv"
    ;;
  vit)
    GENERATE_LABEL="PyTorch ViT encoder cases"
    GENERATE_CMD=("${PYTHON_BIN}" scripts/export_encoder_model_cases.py --suite vit)
    TARGET="build/encoder_model_dual_rocc_test.riscv"
    ;;
  gpt2-prefill)
    GENERATE_LABEL="PyTorch GPT-2 prefill cases"
    GENERATE_CMD=("${PYTHON_BIN}" scripts/export_gpt2_prefill_cases.py)
    TARGET="build/gpt2_prefill_dual_rocc_test.riscv"
    ;;
  attention-operator)
    GENERATE_LABEL="PyTorch attention operator cases"
    GENERATE_CMD=("${PYTHON_BIN}" scripts/export_attention_operator_cases.py)
    TARGET="build/attention_operator_hwpack_test.riscv"
    ;;
  *)
    echo "unknown benchmark '${BENCHMARK}'" >&2
    exit 2
    ;;
esac

case "${HW_MODE}" in
  attention-matmul)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=1 -DGIRDAP_HW_ATTENTION=1 -DGIRDAP_HW_SOFTMAX=0"
    ;;
  attention-only)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=0 -DGIRDAP_HW_ATTENTION=1 -DGIRDAP_HW_SOFTMAX=0"
    ;;
  matmul-only)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=1 -DGIRDAP_HW_ATTENTION=0 -DGIRDAP_HW_SOFTMAX=0"
    ;;
  softmax-only)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=0 -DGIRDAP_HW_ATTENTION=0 -DGIRDAP_HW_SOFTMAX=1"
    ;;
  *)
    echo "unknown hardware mode '${HW_MODE}'" >&2
    exit 2
    ;;
esac

case "${HW_PROFILE}" in
  dual-rocc)
    PROFILE_CFLAGS="-DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1 -DSOFTMAX_OPCODE=0"
    ;;
  matmul-rocc)
    PROFILE_CFLAGS="-DSA_MATMUL_OPCODE=1 -DSA_ATTN_OPCODE=1 -DSOFTMAX_OPCODE=0"
    if [ "${HW_MODE}" != "matmul-only" ]; then
      echo "hardware profile '${HW_PROFILE}' only supports matmul-only mode" >&2
      exit 2
    fi
    ;;
  attention-rocc)
    PROFILE_CFLAGS="-DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1 -DSOFTMAX_OPCODE=0"
    if [ "${HW_MODE}" != "attention-only" ]; then
      echo "hardware profile '${HW_PROFILE}' only supports attention-only mode" >&2
      exit 2
    fi
    ;;
  softmax-rocc)
    PROFILE_CFLAGS="-DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1 -DSOFTMAX_OPCODE=0"
    if [ "${HW_MODE}" != "softmax-only" ]; then
      echo "hardware profile '${HW_PROFILE}' only supports softmax-only mode" >&2
      exit 2
    fi
    ;;
  *)
    echo "unknown hardware profile '${HW_PROFILE}'" >&2
    exit 2
    ;;
esac

echo "Generating ${GENERATE_LABEL}"
"${GENERATE_CMD[@]}"

echo "Building ${BENCHMARK} for ${HW_MODE} on ${HW_PROFILE}"
make clean
make BUILD_MODE=baremetal \
  CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude ${MODE_CFLAGS} ${PROFILE_CFLAGS}" \
  "${TARGET}"
