#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source ../env.sh
set -u

MODE="${1:-core}"
ACTION="${2:-syn}"
CONFIG=Atik4x4RoCCConfig
MESH=4x4
LABEL=atik4x4

usage() {
  cat <<USAGE
Usage: $(basename "$0") {chiptop|rockettile|core} [buildfile|syn|par|syn-par]

Modes:
  chiptop     synth/PAR full ChipTop
  rockettile  synth/PAR RocketTile with Atik RoCC attached
  core        synth/PAR standalone AtikCore

Default: core syn
USAGE
}

case "$MODE" in
  chiptop|soc)
    MODE=chiptop
    VLSI_TOP=ChipTop
    DESIGN_CONFS=example-designs/sky130-openroad-atik-chiptop.yml
    ;;
  rockettile|tile)
    MODE=rockettile
    VLSI_TOP=RocketTile
    DESIGN_CONFS=example-designs/sky130-openroad-atik-rockettile.yml
    ;;
  core|atikcore)
    MODE=core
    VLSI_TOP=AtikCore
    DESIGN_CONFS=example-designs/sky130-openroad-atik-core-${MESH}.yml
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "$ACTION" in
  buildfile|syn|par|syn-par) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

COMMON_MAKE_ARGS=(
  tutorial=sky130-openroad
  CONFIG="$CONFIG"
  VLSI_TOP="$VLSI_TOP"
  TOOLS_CONF=example-openroad.yml
  TECH_CONF=example-sky130.yml
  DESIGN_CONFS="$DESIGN_CONFS"
  VLSI_OBJ_DIR="build-${LABEL}-${MODE}-sky130-openroad"
  'TOP_MACROCOMPILER_MODE=-l $(SMEMS_CACHE) -hir $(SMEMS_HAMMER) --mode strict --force-synflops cc_dir_ext --force-synflops rockettile_dcache_tag_array_ext --force-synflops rockettile_icache_tag_array_ext --force-synflops rockettile_icache_data_arrays_0_ext'
)

echo "CONFIG=$CONFIG TOP=$VLSI_TOP MODE=$MODE ACTION=$ACTION"
make buildfile "${COMMON_MAKE_ARGS[@]}"

case "$ACTION" in
  buildfile)
    ;;
  syn)
    make syn "${COMMON_MAKE_ARGS[@]}"
    ;;
  par)
    make par "${COMMON_MAKE_ARGS[@]}"
    ;;
  syn-par)
    make syn "${COMMON_MAKE_ARGS[@]}"
    make par "${COMMON_MAKE_ARGS[@]}"
    ;;
esac
