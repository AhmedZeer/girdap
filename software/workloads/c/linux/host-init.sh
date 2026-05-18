#!/bin/bash
set -euo pipefail

echo "Building Linux Girdap binaries"
make clean
make BUILD_MODE=linux

mkdir -p overlay/root
cp -r ./build/* overlay/root/

chmod +x overlay/root/run-softmax-smoke.sh
chmod +x overlay/root/run-matmul-smoke.sh
chmod +x overlay/root/run-matmul-smoke-2x2.sh
chmod +x overlay/root/run-attention-smoke.sh
chmod +x overlay/root/run-attention-fused-smoke.sh

chmod +x overlay/root/run-test.sh
find overlay/root -maxdepth 1 -name '*.riscv' -exec chmod +x {} +
