#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
workloads/pytorch/bare/build-pytorch-benchmark.sh vit attention-only dual-rocc
