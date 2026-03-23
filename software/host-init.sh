#!/bin/bash
set -euo pipefail

# This script will run on the host from the workload directory
# (e.g. workloads/example-fed) every time the workload is built.
# It is recommended to call into something like a makefile because
# this script may be called multiple times.
echo "Building softmax benchmark"
make clean
make
mkdir -p overlay/root
cp -r ./build/* overlay/root/

if [ ! -f overlay/root/softmax_norm_test.riscv ]; then
    echo "ERROR: expected overlay/root/softmax_norm_test.riscv after build"
    exit 1
fi

chmod +x overlay/root/run-softmax-smoke.sh
chmod +x overlay/root/softmax_norm_test.riscv
