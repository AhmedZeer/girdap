#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the full OpenLane2 flow for full Girdap ChipTop designs.

This is the RAM/time-heavy path intended for a larger machine. It runs the
complete OpenLane Classic flow, including placement, CTS, global routing,
detailed routing, extraction/signoff checks where enabled by OpenLane.
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
import run_girdap_chiptop_preroute as preroute


DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "girdap-chiptop-full-openlane"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full OpenLane2 Classic flow for Girdap ChipTop configs.",
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
    parser.add_argument("--min-cells", type=int, default=10000)
    parser.add_argument("--top", default="ChipTop")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-openlane", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-verilog", action="store_true")
    parser.add_argument("--no-prune-verilog", action="store_true")
    parser.add_argument("--no-synlig", action="store_true")
    parser.add_argument("--synlig-defer", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--tie-undefined-low", action="store_true")
    parser.add_argument("--reuse-runs", action="store_true")
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument(
        "--disable-expensive-signoff",
        action="store_true",
        help="Disable optional extraction/IR-drop/multi-corner signoff steps while still running routing.",
    )
    parser.add_argument(
        "--verilog-cmd-template",
        default="make -C {chipyard_root}/sims/verilator CONFIG={config} verilog",
    )
    return parser.parse_args()


def write_full_config(
    path: Path,
    design_name: str,
    files: list[Path],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "meta": {"version": 2, "flow": "Classic"},
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
    if args.disable_expensive_signoff:
        config.update(
            {
                "RUN_SPEF_EXTRACTION": False,
                "RUN_MCSTA": False,
                "RUN_IRDROP_REPORT": False,
            }
        )
    if args.pdk:
        config["PDK"] = args.pdk
    if args.scl:
        config["STD_CELL_LIBRARY"] = args.scl
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def metric(metrics: dict[str, object], *names: str) -> str:
    return preroute.metric(metrics, *names)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "flavor",
        "chipyard_config",
        "description",
        "top_module",
        "generated_src",
        "openlane_config",
        "run_dir",
        "synth_area_um2",
        "cell_count",
        "unmapped_cell_count",
        "check_error_count",
        "final_area_um2",
        "die_area_um2",
        "core_area_um2",
        "wns",
        "tns",
        "routing_violations",
        "magic_drc_violations",
        "antenna_violations",
        "lvs_errors",
        "missing_required",
        "metrics_json",
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
        required = preroute.required_tokens_for_design(design, local_files, args)

        config_path = design_dir / "config.json"
        write_full_config(config_path, args.top, rtl_files, args)

        run_dir = None
        status = "config-only" if args.skip_openlane or args.plan else "unknown"
        if not args.skip_openlane:
            tag = f"chiptop-{design.flavor}-full"
            cmd = area.openlane_command(args, config_path, design_dir, tag)
            if args.plan:
                print(f"[cmd] {cmd}")
            else:
                existing_run = design_dir / "runs" / tag
                if existing_run.exists() and not args.reuse_runs:
                    shutil.rmtree(existing_run)
                area.run_command(cmd, cwd=design_dir, log_path=logs_dir / "openlane-chiptop-full.log", plan=False)
                run_dir = area.find_latest_run_dir(design_dir, tag)
                status = "done"

        report_data = area.parse_synthesis_reports(run_dir, args.top)
        metrics = preroute.read_metrics(run_dir)
        missing = preroute.missing_required_tokens(run_dir, required) if not args.skip_openlane and not args.plan else []
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
            "generated_src": str(src_dir),
            "openlane_config": str(config_path),
            "run_dir": str(run_dir or ""),
            "synth_area_um2": report_data["area"] or metric(metrics, "design__instance__area"),
            "cell_count": report_data["cells"] or metric(metrics, "design__instance__count"),
            "unmapped_cell_count": metric(metrics, "design__instance_unmapped__count"),
            "check_error_count": metric(metrics, "synthesis__check_error__count"),
            "final_area_um2": metric(metrics, "design__instance__area"),
            "die_area_um2": metric(metrics, "design__die__area"),
            "core_area_um2": metric(metrics, "design__core__area"),
            "wns": metric(metrics, "timing__setup__ws", "design__setup__ws"),
            "tns": metric(metrics, "timing__setup__tns", "design__setup__tns"),
            "routing_violations": metric(metrics, "route__drc_errors", "routing__drc_errors"),
            "magic_drc_violations": metric(metrics, "magic__drc_error__count"),
            "antenna_violations": metric(metrics, "design__antenna_violation__count"),
            "lvs_errors": metric(metrics, "lvs__error__count"),
            "missing_required": ";".join(missing),
            "metrics_json": metrics_path,
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
