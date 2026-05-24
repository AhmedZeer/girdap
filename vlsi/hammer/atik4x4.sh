#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -u

ACTION="${1:-syn}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [buildfile|syn|par|syn-par]

Runs standalone AtikCore only.

Default: syn
USAGE
}

case "$ACTION" in
  -h|--help|help)
    usage
    exit 0
    ;;
  core|atikcore)
    ACTION="${2:-syn}"
    ;;
esac

find_upward() {
  local name="$1"
  local dir="$SCRIPT_DIR"
  while [[ "$dir" != "/" ]]; do
    if [[ -e "$dir/$name" ]]; then
      echo "$dir/$name"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

ENV_SH="$(find_upward env.sh || true)"
if [[ -z "$ENV_SH" ]]; then
  echo "error: could not find Chipyard env.sh above $SCRIPT_DIR" >&2
  exit 1
fi
# Chipyard/conda env hooks reference optional variables while sourcing.
# Keep nounset off only for the environment setup, then restore it.
set +u
source "$ENV_SH"
set -u

CHIPYARD_ROOT="$(dirname "$ENV_SH")"
if [[ -f "$SCRIPT_DIR/../Makefile" ]]; then
  VLSI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/Makefile" ]]; then
  VLSI_DIR="$SCRIPT_DIR"
elif [[ -f "$CHIPYARD_ROOT/vlsi/Makefile" ]]; then
  VLSI_DIR="$CHIPYARD_ROOT/vlsi"
else
  echo "error: could not find VLSI Makefile near $SCRIPT_DIR or under $CHIPYARD_ROOT/vlsi" >&2
  exit 1
fi
CONF_PREFIX="$SCRIPT_DIR"

CONFIG=Atik4x4RoCCConfig
MESH=4x4
LABEL=atik4x4
VLSI_TOP=AtikCore
TOOLS_CONF="${CONF_PREFIX}/example-openroad.yml"
TECH_CONF="${CONF_PREFIX}/example-sky130.yml"
DESIGN_CONFS="${CONF_PREFIX}/example-designs/sky130-openroad-atik-core-${MESH}.yml"

case "$ACTION" in
  buildfile|syn|par|syn-par) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

COMMON_MAKE_ARGS=(
  tutorial=sky130-openroad
  CONFIG="$CONFIG"
  VLSI_TOP="$VLSI_TOP"
  TOOLS_CONF="$TOOLS_CONF"
  TECH_CONF="$TECH_CONF"
  DESIGN_CONFS="$DESIGN_CONFS"
  VLSI_OBJ_DIR="build-${LABEL}-core-sky130-openroad"
  'TOP_MACROCOMPILER_MODE=-l $(SMEMS_CACHE) -hir $(SMEMS_HAMMER) --mode strict --force-synflops cc_dir_ext --force-synflops cc_banks_0_ext --force-synflops rockettile_dcache_data_arrays_0_ext --force-synflops rockettile_dcache_tag_array_ext --force-synflops rockettile_icache_tag_array_ext --force-synflops rockettile_icache_data_arrays_0_ext --force-synflops mem_0_ext'
)

echo "CONFIG=$CONFIG TOP=$VLSI_TOP ACTION=$ACTION VLSI_DIR=$VLSI_DIR"
make -C "$VLSI_DIR" buildfile "${COMMON_MAKE_ARGS[@]}"

case "$ACTION" in
  buildfile)
    ;;
  syn)
    make -C "$VLSI_DIR" syn "${COMMON_MAKE_ARGS[@]}"
    ;;
  par)
    make -C "$VLSI_DIR" par "${COMMON_MAKE_ARGS[@]}"
    ;;
  syn-par)
    make -C "$VLSI_DIR" syn "${COMMON_MAKE_ARGS[@]}"
    make -C "$VLSI_DIR" par "${COMMON_MAKE_ARGS[@]}"
    ;;
esac
