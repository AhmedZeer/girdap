#!/bin/bash
set -euo pipefail

echo "Building bare-metal dual-RoCC Girdap binary"
make clean
make BUILD_MODE=baremetal \
  CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude -DSA_MATMUL_OPCODE=0 -DSA_ATTN_OPCODE=1" \
  build/fpga_safe_matmul_attention_dual_test.riscv
