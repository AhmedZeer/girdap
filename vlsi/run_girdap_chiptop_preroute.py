#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run pre-route OpenLane2 checks for full Girdap ChipTop designs.

This is intentionally separate from run_girdap_area_synth.py. The area script
is for accelerator-only synthesis. This script targets the full SoC top and
stops before detailed routing so we can cheaply inspect whole-chip feasibility.

The run is only useful if post-synthesis validation proves that Rocket,
DigitalTop, and the Girdap RoCC modules survived optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_girdap_area_synth as area


DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "girdap-chiptop-preroute"

FLOW_BY_STOP = {
    "synthesis": [
        "Yosys.Synthesis",
        "OpenROAD.CheckSDCFiles",
        "OpenROAD.STAPrePNR",
    ],
    "floorplan": [
        "Yosys.Synthesis",
        "OpenROAD.CheckSDCFiles",
        "OpenROAD.STAPrePNR",
        "OpenROAD.Floorplan",
        "OpenROAD.IOPlacement",
    ],
    "gpl": [
        "Yosys.Synthesis",
        "OpenROAD.CheckSDCFiles",
        "OpenROAD.STAPrePNR",
        "OpenROAD.Floorplan",
        "OpenROAD.IOPlacement",
        "OpenROAD.GlobalPlacement",
        "OpenROAD.STAMidPNR",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-ChipTop pre-route OpenLane2 analysis for Girdap configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", default="all", help="Comma-separated Girdap flavors/configs, or all.")
    parser.add_argument("--chipyard-root", type=Path, default=area.CHIPYARD_ROOT)
    parser.add_argument("--openlane2-root", type=Path, default=area.DEFAULT_OPENLANE2_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 16)))
    parser.add_argument("--pdk", default=os.environ.get("PDK", "sky130A"))
    parser.add_argument("--scl", default=os.environ.get("STD_CELL_LIBRARY", "sky130_fd_sc_hd"))
    parser.add_argument("--clock-port", default="clock_uncore")
    parser.add_argument("--clock-period", type=float, default=20.0)
    parser.add_argument("--core-util", type=int, default=20)
    parser.add_argument("--target-density", type=float, default=0.35)
    parser.add_argument("--stop-after", choices=tuple(FLOW_BY_STOP), default="gpl")
    parser.add_argument("--min-cells", type=int, default=10000, help="Below this, flag ChipTop as optimized-away/invalid.")
    parser.add_argument("--top", default="ChipTop")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-openlane", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-verilog", action="store_true")
    parser.add_argument("--no-prune-verilog", action="store_true")
    parser.add_argument("--no-synlig", action="store_true")
    parser.add_argument("--synlig-defer", action="store_true", help="Use Synlig deferred linking.")
    parser.add_argument("--reuse-runs", action="store_true", help="Allow OpenLane to resume an existing run tag.")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Allow flattening. Default keeps hierarchy so validation can prove Rocket/RoCC survived.",
    )
    parser.add_argument(
        "--tie-undefined-low",
        action="store_true",
        help="Use OpenLane/Yosys default undefined-low tying. Default leaves undriven nets untied for diagnosis.",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Extra post-synth module/token that must appear in the netlist. Can be repeated.",
    )
    parser.add_argument(
        "--verilog-cmd-template",
        default="make -C {chipyard_root}/sims/verilator CONFIG={config} verilog",
    )
    return parser.parse_args()


def write_chiptop_config(
    path: Path,
    design_name: str,
    files: list[Path],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "meta": {"version": 2, "flow": FLOW_BY_STOP[args.stop_after]},
        "DESIGN_NAME": design_name,
        "VERILOG_FILES": [str(p) for p in files],
        "CLOCK_PORT": args.clock_port,
        "CLOCK_PERIOD": args.clock_period,
        "FP_CORE_UTIL": args.core_util,
        "PL_TARGET_DENSITY": args.target_density,
        "USE_SYNLIG": not args.no_synlig,
        "SYNLIG_DEFER": args.synlig_defer,
        "SYNTH_HIERARCHY_MODE": "flatten" if args.flatten else "keep",
        "SYNTH_TIE_UNDEFINED": "low" if args.tie_undefined_low else None,
    }
    if args.pdk:
        config["PDK"] = args.pdk
    if args.scl:
        config["STD_CELL_LIBRARY"] = args.scl
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def read_metrics(run_dir: Path | None) -> dict[str, object]:
    if run_dir is None:
        return {}
    metrics_path = run_dir / "final" / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def metric(metrics: dict[str, object], *names: str) -> str:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return str(value)
    return ""


def netlist_files(run_dir: Path | None) -> list[Path]:
    if run_dir is None or not run_dir.exists():
        return []
    final = run_dir / "final"
    roots = [final, run_dir]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(p for p in root.rglob("*.v") if p.is_file())
    return sorted(set(files))


def missing_required_tokens(run_dir: Path | None, required: list[str]) -> list[str]:
    files = netlist_files(run_dir)
    if not files:
        return required
    haystack_parts: list[str] = []
    for path in files:
        try:
            haystack_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    haystack = "\n".join(haystack_parts)
    return [token for token in required if token not in haystack]


def required_tokens_for_design(
    design: area.GirdapDesign,
    files: list[Path],
    args: argparse.Namespace,
) -> list[str]:
    tokens = ["DigitalTop", "RocketTile"]
    overrides: dict[str, str] = {}
    for component in design.components:
        try:
            tokens.append(area.choose_top(design, component, files, "accelerator", overrides))
        except RuntimeError:
            pass
    tokens.extend(args.require)
    return sorted(set(tokens))


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "flavor",
        "chipyard_config",
        "description",
        "top_module",
        "stop_after",
        "generated_src",
        "openlane_config",
        "run_dir",
        "synth_area_um2",
        "cell_count",
        "sequential_area_um2",
        "sequential_area_pct",
        "unmapped_cell_count",
        "check_error_count",
        "wire_count",
        "wire_bits",
        "port_count",
        "port_bits",
        "memory_count",
        "memory_bits",
        "wns",
        "tns",
        "die_area_um2",
        "core_area_um2",
        "missing_required",
        "metrics_json",
        "report",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    chipyard_root = args.chipyard_root.resolve()
    out_dir = args.out_dir.resolve()
    openlane2_root = args.openlane2_root.resolve()
    designs = area.selected_designs(args.configs)

    if args.clean and out_dir.exists():
        if args.plan:
            print(f"[plan] would remove {out_dir}")
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for design in designs:
        print(f"\n=== {design.flavor}: {design.chipyard_config} ===")
        design_dir = out_dir / design.flavor
        logs_dir = design_dir / "logs"
        design_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_generate:
            cmd = args.verilog_cmd_template.format(
                chipyard_root=chipyard_root,
                config=design.chipyard_config,
                flavor=design.flavor,
            )
            area.run_command(cmd, cwd=chipyard_root, log_path=logs_dir / "generate-verilog.log", plan=args.plan)

        src_dir = area.generated_src_dir(chipyard_root, design.chipyard_config)
        files = area.verilog_files(src_dir)
        local_files = area.copy_or_reference_verilog(files, design_dir, args.copy_verilog)
        rtl_files = local_files if args.no_prune_verilog else area.prune_verilog_to_top(local_files, args.top)
        required = required_tokens_for_design(design, local_files, args)

        config_path = design_dir / "config.json"
        write_chiptop_config(config_path, args.top, rtl_files, args)

        run_dir = None
        status = "config-only" if args.skip_openlane or args.plan else "unknown"
        if not args.skip_openlane:
            tag = f"chiptop-{design.flavor}-{args.stop_after}"
            cmd = area.openlane_command(args, config_path, design_dir, tag)
            if args.plan:
                print(f"[cmd] {cmd}")
            else:
                existing_run = design_dir / "runs" / tag
                if existing_run.exists() and not args.reuse_runs:
                    shutil.rmtree(existing_run)
                area.run_command(cmd, cwd=design_dir, log_path=logs_dir / "openlane-chiptop-preroute.log", plan=False)
                run_dir = area.find_latest_run_dir(design_dir, tag)
                status = "done"

        report_data = area.parse_synthesis_reports(run_dir, args.top)
        metrics = read_metrics(run_dir)
        missing = missing_required_tokens(run_dir, required) if not args.skip_openlane and not args.plan else []
        cell_count = int(report_data["cells"] or metric(metrics, "design__instance__count") or 0)
        if missing:
            status = "invalid-missing-required"
        elif cell_count and cell_count < args.min_cells:
            status = "invalid-too-few-cells"

        metrics_path = str(run_dir / "final" / "metrics.json") if run_dir else ""
        row = {
            "flavor": design.flavor,
            "chipyard_config": design.chipyard_config,
            "description": design.description,
            "top_module": args.top,
            "stop_after": args.stop_after,
            "generated_src": str(src_dir),
            "openlane_config": str(config_path),
            "run_dir": str(run_dir or ""),
            "synth_area_um2": report_data["area"] or metric(metrics, "design__instance__area"),
            "cell_count": report_data["cells"] or metric(metrics, "design__instance__count"),
            "sequential_area_um2": report_data["sequential_area"],
            "sequential_area_pct": report_data["sequential_area_pct"],
            "unmapped_cell_count": metric(metrics, "design__instance_unmapped__count"),
            "check_error_count": metric(metrics, "synthesis__check_error__count"),
            "wire_count": report_data["wire_count"],
            "wire_bits": report_data["wire_bits"],
            "port_count": report_data["port_count"],
            "port_bits": report_data["port_bits"],
            "memory_count": report_data["memory_count"],
            "memory_bits": report_data["memory_bits"],
            "wns": metric(metrics, "timing__setup__ws", "design__setup__ws"),
            "tns": metric(metrics, "timing__setup__tns", "design__setup__tns"),
            "die_area_um2": metric(metrics, "design__die__area"),
            "core_area_um2": metric(metrics, "design__core__area"),
            "missing_required": ";".join(missing),
            "metrics_json": metrics_path,
            "report": report_data["report"],
            "status": status,
        }
        rows.append(row)

        print(f"[top] {args.top}")
        print(f"[rtl] {len(rtl_files)} / {len(local_files)} files")
        print(f"[required] {', '.join(required)}")
        print(f"[config] {config_path}")
        if status.startswith("invalid"):
            print(f"[status] {status}: {row['missing_required'] or 'cell count below threshold'}")

    summary = out_dir / "summary.csv"
    write_summary(summary, rows)
    print(f"\n[summary] {summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
