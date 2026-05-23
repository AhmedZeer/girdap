#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source ../env.sh
set -u

COMMON_MAKE_ARGS=(
  tutorial=sky130-openroad
  CONFIG=LeanGemminiRocketConfig
  VLSI_TOP=RocketTile
  TOOLS_CONF=example-openroad.yml
  TECH_CONF=example-sky130.yml
  DESIGN_CONFS=example-designs/sky130-openroad-leangemmini-rockettile.yml
  VLSI_OBJ_DIR=build-leangemmini-rockettile-sky130-openroad
  'TOP_MACROCOMPILER_MODE=-l $(SMEMS_CACHE) -hir $(SMEMS_HAMMER) --mode strict --force-synflops cc_dir_ext --force-synflops rockettile_dcache_tag_array_ext --force-synflops rockettile_icache_tag_array_ext --force-synflops mem_0_ext'
)

make buildfile "${COMMON_MAKE_ARGS[@]}"
make syn "${COMMON_MAKE_ARGS[@]}"
