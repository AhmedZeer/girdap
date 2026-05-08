#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PYTHON_BIN="${CY_DIR:-/home/ubuntu/chipyard-f2}/.conda-env/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

echo "Generating PyTorch Tiny-BERT inference cases"
"${PYTHON_BIN}" scripts/export_tiny_bert_cases.py

echo "Building bare-metal dual-RoCC Tiny-BERT inference binary"
make clean
make BUILD_MODE=baremetal \
  CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude -DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1" \
  build/tiny_bert_dual_rocc_test.riscv
