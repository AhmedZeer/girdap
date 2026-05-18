#!/bin/bash
set -euo pipefail

echo "Building bare-metal Girdap binaries"
make clean
make BUILD_MODE=baremetal

shopt -s nullglob
binaries=(build/*.riscv)

if [ ${#binaries[@]} -eq 0 ]; then
  echo "ERROR: expected at least one bare-metal .riscv binary under build/"
  exit 1
fi
