#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PYTHON_BIN="${CY_DIR:-/home/ubuntu/chipyard-f2}/.conda-env/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

echo "Generating PyTorch ViT-style encoder model cases"
"${PYTHON_BIN}" scripts/export_encoder_model_cases.py --suite vit

echo "Building bare-metal dual-RoCC ViT-style encoder binary"
make clean
make BUILD_MODE=baremetal \
  CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude -DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1" \
  build/encoder_model_dual_rocc_test.riscv
