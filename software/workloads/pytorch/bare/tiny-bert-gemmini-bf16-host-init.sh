#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PYTHON_BIN="${CY_DIR:-/home/ubuntu/chipyard-f2}/.conda-env/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

echo "Generating PyTorch Tiny-BERT inference cases"
"${PYTHON_BIN}" scripts/export_tiny_bert_cases.py

echo "Building Tiny-BERT for Gemmini BF16"
make clean
make BUILD_MODE=baremetal \
  CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude" \
  build/tiny_bert_gemmini_bf16_test.riscv
