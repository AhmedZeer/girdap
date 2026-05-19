#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
workloads/pytorch/bare/build-pytorch-benchmark.sh attention-operator matmul-only dual-rocc
