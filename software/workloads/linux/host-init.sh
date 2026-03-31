#!/bin/bash
set -euo pipefail

echo "Building Linux ToyRoCC binaries"
make clean
make BUILD_MODE=linux

mkdir -p overlay/root
cp -r ./build/* overlay/root/

chmod +x overlay/root/run-softmax-smoke.sh
chmod +x overlay/root/run-matmul-smoke.sh

chmod +x overlay/root/run-test.sh
chmod +x overlay/root/softmax_norm_test.riscv
chmod +x overlay/root/systolic_matmul_simple.riscv
chmod +x overlay/root/test.riscv
