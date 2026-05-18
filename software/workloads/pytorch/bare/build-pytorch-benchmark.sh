#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tiny-bert|vit|gpt2-prefill> <attention-matmul|attention-only|matmul-only|softmax-only>" >&2
  exit 2
fi

BENCHMARK="$1"
HW_MODE="$2"

PYTHON_BIN="${CY_DIR:-/home/ubuntu/chipyard-f2}/.conda-env/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

case "${BENCHMARK}" in
  tiny-bert)
    echo "Generating PyTorch Tiny-BERT inference cases"
    "${PYTHON_BIN}" scripts/export_tiny_bert_cases.py
    TARGET="build/tiny_bert_dual_rocc_test.riscv"
    ;;
  vit)
    echo "Generating PyTorch ViT encoder cases"
    "${PYTHON_BIN}" scripts/export_encoder_model_cases.py --suite vit
    TARGET="build/encoder_model_dual_rocc_test.riscv"
    ;;
  gpt2-prefill)
    echo "Generating PyTorch GPT-2 prefill cases"
    "${PYTHON_BIN}" scripts/export_gpt2_prefill_cases.py
    TARGET="build/gpt2_prefill_dual_rocc_test.riscv"
    ;;
  *)
    echo "unknown benchmark '${BENCHMARK}'" >&2
    exit 2
    ;;
esac

case "${HW_MODE}" in
  attention-matmul)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=1 -DGIRDAP_HW_ATTENTION=1 -DGIRDAP_HW_SOFTMAX=0 -DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1"
    ;;
  attention-only)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=0 -DGIRDAP_HW_ATTENTION=1 -DGIRDAP_HW_SOFTMAX=0 -DSA_ATTN_OPCODE=1"
    ;;
  matmul-only)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=1 -DGIRDAP_HW_ATTENTION=0 -DGIRDAP_HW_SOFTMAX=0 -DSA_MATMUL_OPCODE=1"
    ;;
  softmax-only)
    MODE_CFLAGS="-DGIRDAP_HW_MATMUL=0 -DGIRDAP_HW_ATTENTION=0 -DGIRDAP_HW_SOFTMAX=1 -DSOFTMAX_OPCODE=0"
    ;;
  *)
    echo "unknown hardware mode '${HW_MODE}'" >&2
    exit 2
    ;;
esac

echo "Building ${BENCHMARK} for ${HW_MODE}"
make clean
make BUILD_MODE=baremetal \
  CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude ${MODE_CFLAGS}" \
  "${TARGET}"
