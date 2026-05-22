#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate RTL and run OpenLane2 synthesis-only area estimates for Girdap.

This script is intentionally a thin orchestration layer:
  1. Build/reuse Chipyard generated SystemVerilog for each Girdap config.
  2. Pick the accelerator top module from the generated RTL.
  3. Emit an OpenLane2 config whose flow is only Yosys.Synthesis.
  4. Run OpenLane2 and collect a CSV summary from synthesis reports.

It does not run placement, routing, DRC, LVS, or stream-out. The resulting
numbers are useful for early relative area comparison, not final signoff area.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHIPYARD_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPENLANE2_ROOT = Path("/home/ubuntu/firesim_build_disk/openlane2")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out" / "girdap-area"


@dataclass(frozen=True)
class GirdapComponent:
    name: str
    top_patterns: tuple[str, ...]


@dataclass(frozen=True)
class GirdapDesign:
    flavor: str
    chipyard_config: str
    description: str
    components: tuple[GirdapComponent, ...]


GIRDAP_DESIGNS: tuple[GirdapDesign, ...] = (
    GirdapDesign(
        flavor="iri",
        chipyard_config="Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512Config",
        description="8x8 BF16 matmul + 8x8 BF16 online attention, HW packer, exp LUT, maxK=512",
        components=(
            GirdapComponent(
                name="attention",
                top_patterns=(
                    r"^FpgaSafeOnlineAttention8x8.*",
                    r".*OnlineAttention8x8.*",
                ),
            ),
            GirdapComponent(
                name="matmul",
                top_patterns=(
                    r"^SystolicArrayFpgaSafe8x8.*",
                    r".*FpgaSafe8x8.*",
                    r".*Matmul.*8x8.*",
                ),
            ),
        ),
    ),
    GirdapDesign(
        flavor="vasat",
        chipyard_config="FusedOnlineAttention8x8BF16FpgaSafePackerExpLutConfig",
        description="8x8 BF16 online attention, HW packer, exp LUT",
        components=(
            GirdapComponent(
                name="attention",
                top_patterns=(
                    r"^FpgaSafeOnlineAttention8x8.*",
                    r".*OnlineAttention8x8.*",
                ),
            ),
        ),
    ),
    GirdapDesign(
        flavor="kucuk",
        chipyard_config="MatmulAccel8x8BF16FpgaSafeConfig",
        description="8x8 BF16 FPGA-safe systolic matmul",
        components=(
            GirdapComponent(
                name="matmul",
                top_patterns=(
                    r"^SystolicArrayFpgaSafe8x8.*",
                    r".*FpgaSafe8x8.*",
                    r".*Matmul.*8x8.*",
                ),
            ),
        ),
    ),
    GirdapDesign(
        flavor="minik",
        chipyard_config="SoftmaxAccel128Config",
        description="128-bit bus BF16 online softmax RoCC",
        components=(
            GirdapComponent(
                name="softmax",
                top_patterns=(
                    r"^OnlineSoftmax.*",
                    r"^Softmax.*",
                ),
            ),
        ),
    ),
)


MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b", re.MULTILINE)
INSTANCE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\([^;]*?\)\s*)?[A-Za-z_][A-Za-z0-9_$]*\s*\(",
    re.MULTILINE | re.DOTALL,
)
AREA_RE = re.compile(r"Chip area for (?:top )?module ['`]?(?:\\)?([^'`\\\s:]+)['`]?:\s*([0-9]+(?:\.[0-9]+)?)")
CELL_RE = re.compile(r"^\s*Number of cells:\s*([0-9]+)\s*$", re.MULTILINE)
STAT_LINE_RE = re.compile(r"^\s*(Number of [A-Za-z ]+):\s*([0-9]+)\s*$", re.MULTILINE)
SEQ_AREA_RE = re.compile(r"of which used for sequential elements:\s*([0-9]+(?:\.[0-9]+)?)\s*\(([0-9]+(?:\.[0-9]+)?)%\)")
CHECK_PROBLEMS_RE = re.compile(r"Found and reported\s+([0-9]+)\s+problems?")
VERILOG_KEYWORDS = {
    "always", "assign", "begin", "case", "else", "endmodule", "for", "function",
    "generate", "if", "initial", "input", "logic", "module", "output", "reg",
    "wire",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthesis-only OpenLane2 area estimates for Girdap configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        default="all",
        help="Comma-separated Girdap flavors/config names to run, or 'all'.",
    )
    parser.add_argument("--chipyard-root", type=Path, default=CHIPYARD_ROOT)
    parser.add_argument("--openlane2-root", type=Path, default=DEFAULT_OPENLANE2_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 16)))
    parser.add_argument("--pdk", default=os.environ.get("PDK", "sky130A"))
    parser.add_argument("--scl", default=os.environ.get("STD_CELL_LIBRARY", "sky130_fd_sc_hd"))
    parser.add_argument("--clock-port", default="clock")
    parser.add_argument("--clock-period", type=float, default=20.0)
    parser.add_argument("--core-util", type=int, default=20)
    parser.add_argument("--target-density", type=float, default=0.35)
    parser.add_argument(
        "--top-mode",
        choices=("accelerator", "chiptop", "testharness", "all"),
        default="accelerator",
        help="Top to synthesize. Accelerator is best for relative Girdap area; all runs accelerator components plus ChipTop.",
    )
    parser.add_argument(
        "--top",
        action="append",
        default=[],
        metavar="FLAVOR=MODULE",
        help="Override top module. Use FLAVOR=MODULE or FLAVOR.COMPONENT=MODULE. Can be repeated.",
    )
    parser.add_argument(
        "--verilog-cmd-template",
        default="make -C {chipyard_root}/sims/verilator CONFIG={config} verilog",
        help="Command template used to generate SystemVerilog.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse existing sims/verilator/generated-src output.",
    )
    parser.add_argument(
        "--skip-openlane",
        action="store_true",
        help="Only generate/find RTL and write OpenLane2 configs.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print actions and write configs where possible, but do not execute commands.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove this script's output directory before running.",
    )
    parser.add_argument(
        "--copy-verilog",
        action="store_true",
        help="Copy generated Verilog into out-dir instead of referencing generated-src in place.",
    )
    parser.add_argument(
        "--no-prune-verilog",
        action="store_true",
        help="Pass every generated RTL file to OpenLane instead of the selected top's dependency closure.",
    )
    parser.add_argument(
        "--no-synlig",
        action="store_true",
        help="Use plain Yosys read_verilog instead of Synlig/read_systemverilog.",
    )
    parser.add_argument(
        "--synlig-defer",
        action="store_true",
        help="Use Synlig deferred linking. Disabled by default because stale Surelog state can leak across runs.",
    )
    parser.add_argument(
        "--reuse-runs",
        action="store_true",
        help="Allow OpenLane to resume an existing run tag instead of deleting this script's prior run directory.",
    )
    return parser.parse_args()


def selected_designs(selection: str) -> list[GirdapDesign]:
    by_flavor = {d.flavor: d for d in GIRDAP_DESIGNS}
    by_config = {d.chipyard_config: d for d in GIRDAP_DESIGNS}
    if selection == "all":
        return list(GIRDAP_DESIGNS)

    result: list[GirdapDesign] = []
    for raw_name in selection.split(","):
        name = raw_name.strip()
        if not name:
            continue
        design = by_flavor.get(name) or by_config.get(name)
        if design is None:
            valid = ", ".join(d.flavor for d in GIRDAP_DESIGNS)
            raise SystemExit(f"unknown config '{name}'. Valid flavors: {valid}")
        result.append(design)
    return result


def parse_top_overrides(items: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--top expects FLAVOR=MODULE, got '{item}'")
        flavor, module = item.split("=", 1)
        overrides[flavor.strip()] = module.strip()
    return overrides


def run_command(cmd: str, cwd: Path, log_path: Path, plan: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cmd] {cmd}")
    print(f"[log] {log_path}")
    if plan:
        return
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with rc={proc.returncode}: {cmd}\nsee {log_path}")


def generated_src_dir(chipyard_root: Path, config: str) -> Path:
    base = chipyard_root / "sims" / "verilator" / "generated-src"
    exact = base / f"chipyard.harness.TestHarness.{config}"
    if exact.exists():
        return exact
    matches = sorted(p for p in base.glob(f"*{config}*") if p.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no generated-src directory found for {config} under {base}")
    choices = "\n  ".join(str(p) for p in matches[:20])
    raise RuntimeError(f"multiple generated-src directories match {config}; use --skip-generate after pruning:\n  {choices}")


def verilog_files(src_dir: Path) -> list[Path]:
    files = sorted(
        p
        for p in src_dir.rglob("*")
        if p.suffix in {".sv", ".v"} and p.is_file()
    )
    if not files:
        raise FileNotFoundError(f"no .sv/.v files found under {src_dir}")
    return files


def copy_or_reference_verilog(files: list[Path], design_dir: Path, copy: bool) -> list[Path]:
    if not copy:
        return files

    rtl_dir = design_dir / "src"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: dict[str, int] = {}
    for src in files:
        name = src.name
        if name in seen:
            seen[name] += 1
            dst = rtl_dir / f"{src.stem}_{seen[name]}{src.suffix}"
        else:
            seen[name] = 0
            dst = rtl_dir / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def modules_in_files(files: Iterable[Path]) -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in MODULE_RE.finditer(text):
            modules.setdefault(match.group(1), path)
    return modules


def module_text(path: Path, module_name: str) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        rf"^\s*module\s+{re.escape(module_name)}\b.*?^\s*endmodule\b",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(0) if match else ""


def prune_verilog_to_top(files: list[Path], top: str) -> list[Path]:
    modules = modules_in_files(files)
    if top not in modules:
        raise RuntimeError(f"top module {top} was not found in generated RTL")

    needed_modules: set[str] = set()
    needed_files: set[Path] = set()
    worklist = [top]

    while worklist:
        module = worklist.pop()
        if module in needed_modules:
            continue
        path = modules.get(module)
        if path is None:
            continue

        needed_modules.add(module)
        needed_files.add(path)
        body = module_text(path, module)
        for inst_match in INSTANCE_RE.finditer(body):
            child = inst_match.group(1)
            if child in VERILOG_KEYWORDS:
                continue
            if child in modules and child not in needed_modules:
                worklist.append(child)

    return [path for path in files if path in needed_files]


def choose_top(
    design: GirdapDesign,
    component: GirdapComponent,
    files: list[Path],
    top_mode: str,
    overrides: dict[str, str],
) -> str:
    component_key = f"{design.flavor}.{component.name}"
    if component_key in overrides:
        return overrides[component_key]
    if design.flavor in overrides and len(design.components) == 1:
        return overrides[design.flavor]
    if component.name == "chiptop" or top_mode == "chiptop":
        return "ChipTop"
    if component.name == "testharness" or top_mode == "testharness":
        return "TestHarness"

    modules = modules_in_files(files)
    if not modules:
        raise RuntimeError(f"could not discover modules for {design.flavor}")

    for pattern in component.top_patterns:
        regex = re.compile(pattern)
        matches = sorted(name for name in modules if regex.match(name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer the highest-level non-Impl wrapper if Chisel generated both.
            preferred = [m for m in matches if not m.endswith("_1") and "Impl" not in m]
            if len(preferred) == 1:
                return preferred[0]
            raise RuntimeError(
                f"ambiguous top for {design.flavor}.{component.name} using pattern '{pattern}': "
                f"{', '.join(matches[:30])}. Use --top {design.flavor}.{component.name}=MODULE."
            )

    candidates = sorted(
        name for name in modules
        if any(token in name.lower() for token in ("girdap", "attention", "softmax", "systolic", "matmul"))
    )
    preview = "\n  ".join(candidates[:80])
    raise RuntimeError(
        f"could not auto-pick accelerator top for {design.flavor}.{component.name}. "
        f"Use --top {design.flavor}.{component.name}=MODULE. Candidates:\n  {preview}"
    )


def write_openlane_config(
    path: Path,
    design_name: str,
    files: list[Path],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_or_abs = [str(p) for p in files]
    config = {
        "meta": {
            "version": 2,
            "flow": ["Yosys.Synthesis"],
        },
        "DESIGN_NAME": design_name,
        "VERILOG_FILES": rel_or_abs,
        "CLOCK_PORT": args.clock_port,
        "CLOCK_PERIOD": args.clock_period,
        "FP_CORE_UTIL": args.core_util,
        "PL_TARGET_DENSITY": args.target_density,
        # Generated Chisel/CIRCT RTL can contain SystemVerilog assignment patterns
        # such as '{...}; plain Yosys read_verilog rejects those.
        "USE_SYNLIG": not args.no_synlig,
        "SYNLIG_DEFER": args.synlig_defer,
    }
    if args.pdk:
        config["PDK"] = args.pdk
    if args.scl:
        config["STD_CELL_LIBRARY"] = args.scl
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def openlane_command(args: argparse.Namespace, config_path: Path, design_dir: Path, tag: str) -> str:
    return (
        f"openlane -j {args.jobs} --design-dir {shell_quote(design_dir)} "
        f"--run-tag {shell_quote(tag)} {shell_quote(config_path)}"
    )


def shell_quote(path_or_text: object) -> str:
    text = str(path_or_text)
    if re.fullmatch(r"[A-Za-z0-9_./:=,+@%-]+", text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def find_latest_run_dir(design_dir: Path, tag: str) -> Path | None:
    exact = design_dir / "runs" / tag
    if exact.exists():
        return exact
    runs_dir = design_dir / "runs"
    if not runs_dir.exists():
        return None
    runs = sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def parse_synthesis_reports(run_dir: Path | None, top: str) -> dict[str, str]:
    if run_dir is None or not run_dir.exists():
        return empty_report_data()

    candidates = sorted(
        p
        for p in run_dir.rglob("*")
        if p.is_file() and p.suffix in {".rpt", ".log", ".txt"}
    )
    area = ""
    cells = ""
    report = ""
    sequential_area = ""
    sequential_area_pct = ""
    stat_data: dict[str, str] = {}
    check_problems = ""
    latch_count = ""
    fallback_area = ""
    fallback_cells = ""
    fallback_report = ""
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if path.name == "chk.rpt":
            problem_matches = CHECK_PROBLEMS_RE.findall(text)
            if problem_matches:
                check_problems = problem_matches[-1]
        if path.name == "latch.rpt":
            latch_count = str(len(re.findall(r"\blatch\b", text, flags=re.IGNORECASE)))
        if "Number of wires:" in text and (path.name in {"stat.rpt", "post_dff.rpt"} or not stat_data):
            parsed_stats = {
                key.lower().replace("number of ", "").replace(" ", "_"): value
                for key, value in STAT_LINE_RE.findall(text)
            }
            if parsed_stats:
                stat_data = parsed_stats
        cell_matches = CELL_RE.findall(text)
        last_cell_count = cell_matches[-1] if cell_matches else ""
        for match in AREA_RE.finditer(text):
            module, value = match.groups()
            seq_match = SEQ_AREA_RE.search(text[match.end(): match.end() + 200])
            if not fallback_area:
                fallback_area = value
                fallback_cells = last_cell_count
                fallback_report = str(path)
            if module == top:
                area = value
                cells = last_cell_count
                report = str(path)
                if seq_match:
                    sequential_area, sequential_area_pct = seq_match.groups()
                break
        if area:
            break
        if not cells and last_cell_count:
            cells = last_cell_count
            report = str(path)
    if not area and fallback_area:
        area = fallback_area
        cells = fallback_cells or cells
        report = fallback_report
    data = empty_report_data()
    data.update(
        {
            "area": area,
            "cells": cells,
            "sequential_area": sequential_area,
            "sequential_area_pct": sequential_area_pct,
            "report": report,
            "check_problems": check_problems,
            "latch_count": latch_count,
        }
    )
    data.update(
        {
            "wire_count": stat_data.get("wires", ""),
            "wire_bits": stat_data.get("wire_bits", ""),
            "public_wire_count": stat_data.get("public_wires", ""),
            "public_wire_bits": stat_data.get("public_wire_bits", ""),
            "port_count": stat_data.get("ports", ""),
            "port_bits": stat_data.get("port_bits", ""),
            "memory_count": stat_data.get("memories", ""),
            "memory_bits": stat_data.get("memory_bits", ""),
            "process_count": stat_data.get("processes", ""),
        }
    )
    return data


def empty_report_data() -> dict[str, str]:
    return {
        "area": "",
        "cells": "",
        "sequential_area": "",
        "sequential_area_pct": "",
        "wire_count": "",
        "wire_bits": "",
        "public_wire_count": "",
        "public_wire_bits": "",
        "port_count": "",
        "port_bits": "",
        "memory_count": "",
        "memory_bits": "",
        "process_count": "",
        "check_problems": "",
        "latch_count": "",
        "report": "",
    }


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "flavor",
        "component",
        "chipyard_config",
        "description",
        "top_module",
        "generated_src",
        "openlane_config",
        "run_dir",
        "synth_area",
        "synth_area_um2",
        "cell_count",
        "sequential_area_um2",
        "sequential_area_pct",
        "wire_count",
        "wire_bits",
        "public_wire_count",
        "public_wire_bits",
        "port_count",
        "port_bits",
        "memory_count",
        "memory_bits",
        "process_count",
        "check_problems",
        "latch_count",
        "report",
        "status",
    ]
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
    designs = selected_designs(args.configs)
    top_overrides = parse_top_overrides(args.top)

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
            run_command(cmd, cwd=chipyard_root, log_path=logs_dir / "generate-verilog.log", plan=args.plan)

        src_dir = generated_src_dir(chipyard_root, design.chipyard_config)
        files = verilog_files(src_dir)
        if args.top_mode == "accelerator":
            run_components = design.components
        elif args.top_mode == "all":
            run_components = design.components + (GirdapComponent(name="chiptop", top_patterns=()),)
        else:
            run_components = (GirdapComponent(name=args.top_mode, top_patterns=()),)
        component_rows: list[dict[str, str]] = []
        accelerator_rows: list[dict[str, str]] = []
        for component in run_components:
            component_dir = design_dir / component.name
            component_logs_dir = component_dir / "logs"
            local_files = copy_or_reference_verilog(files, component_dir, args.copy_verilog)
            top = choose_top(design, component, local_files, args.top_mode, top_overrides)
            synth_files = local_files if args.no_prune_verilog else prune_verilog_to_top(local_files, top)
            config_path = component_dir / "config.json"
            write_openlane_config(config_path, top, synth_files, args)

            run_dir = None
            status = "config-only" if args.skip_openlane or args.plan else "unknown"
            if not args.skip_openlane:
                tag = f"synth-{design.flavor}-{component.name}"
                cmd = openlane_command(args, config_path, component_dir, tag)
                if args.plan:
                    print(f"[cmd] {cmd}")
                else:
                    existing_run = component_dir / "runs" / tag
                    if existing_run.exists() and not args.reuse_runs:
                        shutil.rmtree(existing_run)
                    run_command(
                        cmd,
                        cwd=component_dir,
                        log_path=component_logs_dir / "openlane-yosys-synthesis.log",
                        plan=False,
                    )
                    run_dir = find_latest_run_dir(component_dir, tag)
                    status = "done"

            report_data = parse_synthesis_reports(run_dir, top)
            row = {
                "flavor": design.flavor,
                "component": component.name,
                "chipyard_config": design.chipyard_config,
                "description": design.description,
                "top_module": top,
                "generated_src": str(src_dir),
                "openlane_config": str(config_path),
                "run_dir": str(run_dir or ""),
                "synth_area": report_data["area"],
                "synth_area_um2": report_data["area"],
                "cell_count": report_data["cells"],
                "sequential_area_um2": report_data["sequential_area"],
                "sequential_area_pct": report_data["sequential_area_pct"],
                "wire_count": report_data["wire_count"],
                "wire_bits": report_data["wire_bits"],
                "public_wire_count": report_data["public_wire_count"],
                "public_wire_bits": report_data["public_wire_bits"],
                "port_count": report_data["port_count"],
                "port_bits": report_data["port_bits"],
                "memory_count": report_data["memory_count"],
                "memory_bits": report_data["memory_bits"],
                "process_count": report_data["process_count"],
                "check_problems": report_data["check_problems"],
                "latch_count": report_data["latch_count"],
                "report": report_data["report"],
                "status": status,
            }
            component_rows.append(row)
            if component.name in {c.name for c in design.components}:
                accelerator_rows.append(row)
            rows.append(row)
            print(f"[top:{component.name}] {top}")
            print(f"[rtl:{component.name}] {len(synth_files)} / {len(local_files)} files")
            print(f"[config:{component.name}] {config_path}")

        if args.top_mode in {"accelerator", "all"} and len(accelerator_rows) > 1:
            areas = [float(row["synth_area"]) for row in accelerator_rows if row.get("synth_area")]
            cells = [int(row["cell_count"]) for row in accelerator_rows if row.get("cell_count")]
            rows.append(
                {
                    "flavor": design.flavor,
                    "component": "aggregate",
                    "chipyard_config": design.chipyard_config,
                    "description": design.description,
                    "top_module": "",
                    "generated_src": str(src_dir),
                    "openlane_config": "",
                    "run_dir": "",
                    "synth_area": f"{sum(areas):.6f}" if len(areas) == len(accelerator_rows) else "",
                    "synth_area_um2": f"{sum(areas):.6f}" if len(areas) == len(accelerator_rows) else "",
                    "cell_count": str(sum(cells)) if len(cells) == len(accelerator_rows) else "",
                    "sequential_area_um2": (
                        f"{sum(float(row['sequential_area_um2']) for row in accelerator_rows):.6f}"
                        if all(row.get("sequential_area_um2") for row in accelerator_rows)
                        else ""
                    ),
                    "report": "",
                    "status": "aggregate",
                }
            )

    summary = out_dir / "summary.csv"
    write_summary(summary, rows)
    print(f"\n[summary] {summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
