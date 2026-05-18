#!/bin/bash
set -euo pipefail

PYTHON_BIN="${CY_DIR:-/home/ubuntu/chipyard-f2}/.conda-env/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

echo "Generating PyTorch transformer layer cases"
"${PYTHON_BIN}" scripts/export_transformer_layer_cases.py

echo "Building bare-metal dual-RoCC transformer layer binary"
make clean
make BUILD_MODE=baremetal \
  CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude -DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1" \
  build/transformer_layer_dual_rocc_test.riscv
