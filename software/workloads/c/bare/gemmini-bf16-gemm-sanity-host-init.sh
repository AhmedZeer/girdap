#!/bin/bash
set -euo pipefail

echo "Building Gemmini BF16 GEMM sanity bare-metal binary"
make clean
make BUILD_MODE=baremetal CFLAGS="-O2 -static -specs=htif_nano.specs -Iinclude" build/gemmini_bf16_gemm_sanity.riscv
