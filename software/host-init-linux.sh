#!/bin/bash
set -euo pipefail

echo "Building softmax benchmark"
make clean
make BUILD_MODE=linux
# make BUILD_MODE=baremetal

mkdir -p overlay/root
cp -r ./build/* overlay/root/

if [ ! -f overlay/root/softmax_norm_test.riscv ]; then
    echo "ERROR: expected overlay/root/softmax_norm_test.riscv after build"
    exit 1
fi

if [ ! -f overlay/root/softmax_norm_fpga_smoke.riscv ]; then
    echo "ERROR: expected overlay/root/softmax_norm_fpga_smoke.riscv after build"
    exit 1
fi

chmod +x overlay/root/run-softmax-smoke.sh
chmod +x overlay/root/run-test.sh
chmod +x overlay/root/softmax_norm_test.riscv
chmod +x overlay/root/softmax_norm_fpga_smoke.riscv
chmod +x overlay/root/test.riscv
