#!/usr/bin/env python3
"""Generate SVG benchmark analysis from software/uartlog RESULT_JSON records."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UARTLOG = ROOT / "software" / "uartlog"
OUT = ROOT / ".analysis"
FIG = OUT / "figures"
DATA = OUT / "data"

CONFIG_DIRS = {
    "2x2": UARTLOG / "atik_2x2",
    "4x4": UARTLOG / "atik_4x4",
    "8x8": UARTLOG / "atik_8x8_v2",
}

CONFIG_COLORS = {
    "RocketCore": "#8a8f98",
    "2x2": "#c9b6ff",
    "4x4": "#8f63dd",
    "8x8": "#4b238f",
}

WORKLOAD_ORDER = [
    "matmul-benchmark",
    "attention-benchmark",
    "vit",
    "tiny-bert",
    "gpt2-prefill",
]

WORKLOAD_TITLES = {
    "matmul-benchmark": "Matmul Benchmark",
    "attention-benchmark": "Attention Benchmark",
    "vit": "ViT Workload",
    "tiny-bert": "TinyBERT Workload",
    "gpt2-prefill": "GPT-2 Prefill Workload",
}

WORKLOAD_FILES = {
    "2x2": {
        "matmul-benchmark": "matmul.txt",
        "attention-benchmark": "attn.txt",
        "vit": "vit.txt",
        "tiny-bert": "bert.txt",
        "gpt2-prefill": "gpt2prefill.txt",
    },
    "4x4": {
        "matmul-benchmark": "matmul.txt",
        "attention-benchmark": "attn.txt",
        "vit": "vit.txt",
        "tiny-bert": "bert.txt",
        "gpt2-prefill": "gpt2.txt",
    },
    "8x8": {
        "matmul-benchmark": "matmul.txt",
        "attention-benchmark": "attn.txt",
        "vit": "vit.txt",
        "tiny-bert": "tinybert.txt",
        "gpt2-prefill": "gpt2prefill.txt",
    },
}


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def short_case_name(name: str) -> str:
    if name.startswith("matmul_"):
        m = re.search(r"m(\d+)_n(\d+)_k(\d+)", name)
        if m:
            return f"M{m.group(1)} N{m.group(2)} K{m.group(3)}"
    if name.startswith("attention_") or name.startswith("causal_attention_"):
        causal = name.startswith("causal_")
        m = re.search(r"q(\d+)_kv(\d+)_d(\d+)_v(\d+)", name)
        if m:
            prefix = "Causal " if causal else ""
            return f"{prefix}Q{m.group(1)} KV{m.group(2)} D{m.group(3)} V{m.group(4)}"
    if name.startswith("vit_"):
        m = re.search(r"l(\d+).*d(\d+)_h(\d+).*cls(\d+)", name)
        if m:
            return f"L{m.group(1)} D{m.group(2)} H{m.group(3)} C{m.group(4)}"
    if name.startswith("gpt2prefill_"):
        m = re.search(r"l(\d+)_s(\d+)_d(\d+)_h(\d+).*vocab(\d+)", name)
        if m:
            return f"L{m.group(1)} S{m.group(2)} D{m.group(3)} H{m.group(4)} V{m.group(5)}"
    if "tiny" in name or name.startswith("tinybert"):
        m = re.search(r"l(\d+).*s(\d+).*d(\d+).*h(\d+)", name)
        if m:
            return f"L{m.group(1)} S{m.group(2)} D{m.group(3)} H{m.group(4)}"
    return name.replace("_", " ")


def short_shape(shape: str, fallback: str) -> str:
    if not shape:
        return short_case_name(fallback)
    vals = dict(re.findall(r"([A-Za-z_]+)=([0-9]+)", shape))
    if {"M", "N", "K"}.issubset(vals):
        return f"M{vals['M']} N{vals['N']} K{vals['K']}"
    if {"q", "kv", "d", "value"}.issubset(vals):
        prefix = "Causal " if vals.get("causal") == "1" else ""
        return f"{prefix}Q{vals['q']} KV{vals['kv']} D{vals['d']} V{vals['value']}"
    if {"seq", "d_model", "heads", "classes"}.issubset(vals):
        return f"S{vals['seq']} D{vals['d_model']} H{vals['heads']} C{vals['classes']}"
    if {"seq", "d_model", "heads", "vocab"}.issubset(vals):
        return f"S{vals['seq']} D{vals['d_model']} H{vals['heads']} V{vals['vocab']}"
    return short_case_name(fallback)


def parse_result_json(path: Path, config: str) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    current_shape = ""
    for line in path.read_text(errors="replace").splitlines():
        if line.strip().startswith("shape :"):
            current_shape = line.split("shape :", 1)[1].strip()
            continue
        marker = "RESULT_JSON "
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        item["config"] = config
        item["source"] = str(path.relative_to(ROOT))
        item["speedup"] = item["cpu_cycles"] / item["hw_cycles"] if item.get("hw_cycles") else 0.0
        item["shape"] = current_shape
        item["case_key"] = current_shape or f"case_{item.get('case', '')}:{item.get('name', '')}"
        item["case_label"] = short_shape(current_shape, item.get("name", f"case_{item.get('case', '')}"))
        rows.append(item)
    return rows


def collect() -> list[dict]:
    rows: list[dict] = []
    for config, directory in CONFIG_DIRS.items():
        for workload, filename in WORKLOAD_FILES[config].items():
            for row in parse_result_json(directory / filename, config):
                row["expected_workload"] = workload
                rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fields = [
        "config",
        "workload",
        "case",
        "name",
        "case_label",
        "shape",
        "case_key",
        "status",
        "cpu_cycles",
        "hw_cycles",
        "speedup",
        "speedup_x100",
        "mismatches",
        "source",
    ]
    with (DATA / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def buffered_max(value: float, buffer: float = 0.12) -> float:
    if value <= 1:
        return 1.15
    return value * (1.0 + buffer)


def fmt_cycles(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def svg_line_plot(workload: str, rows: list[dict]) -> str:
    by_config = defaultdict(dict)
    labels = {}
    order = {}
    for row in rows:
        case_key = row["case_key"]
        by_config[row["config"]][case_key] = row
        labels[case_key] = row["case_label"]
        order.setdefault(case_key, int(row["case"]))

    cases = sorted(labels, key=lambda k: (order.get(k, 0), labels[k], k))
    width = max(1000, 150 + 95 * len(cases))
    height = 620
    ml, mr, mt, mb = 90, 40, 75, 155
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    max_speedup = max([1.0] + [r["speedup"] for r in rows])
    ymax = buffered_max(max_speedup)

    def x_for(idx: int) -> float:
        if len(cases) == 1:
            return ml + plot_w / 2
        return ml + idx * (plot_w / (len(cases) - 1))

    def y_for(v: float) -> float:
        return mt + plot_h - (v / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="26" font-weight="700">{esc(WORKLOAD_TITLES.get(workload, workload))}: Per-Case Speedup</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
    ]

    ticks = 5
    for i in range(ticks + 1):
        val = ymax * i / ticks
        y = y_for(val)
        parts.append(f'<line x1="{ml-6}" y1="{y:.2f}" x2="{ml+plot_w}" y2="{y:.2f}" stroke="#e7e7e7" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{y+5:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="14">{val:.0f}x</text>')

    # CPU baseline.
    yb = y_for(1.0)
    parts.append(f'<line x1="{ml}" y1="{yb:.2f}" x2="{ml+plot_w}" y2="{yb:.2f}" stroke="{CONFIG_COLORS["RocketCore"]}" stroke-width="2.5" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{ml+plot_w-4}" y="{yb-8:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="{CONFIG_COLORS["RocketCore"]}">RocketCore baseline 1x</text>')

    for config in ("2x2", "4x4", "8x8"):
        pts = []
        for idx, case_key in enumerate(cases):
            row = by_config.get(config, {}).get(case_key)
            if row is None:
                continue
            pts.append((x_for(idx), y_for(row["speedup"]), row))
        if not pts:
            continue
        path = " ".join(f'{x:.2f},{y:.2f}' for x, y, _ in pts)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{CONFIG_COLORS[config]}" stroke-width="3.5"/>')
        for x, y, row in pts:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{CONFIG_COLORS[config]}" stroke="#ffffff" stroke-width="1.5"/>')

    for idx, case_key in enumerate(cases):
        x = x_for(idx)
        parts.append(f'<line x1="{x:.2f}" y1="{mt+plot_h}" x2="{x:.2f}" y2="{mt+plot_h+6}" stroke="#222" stroke-width="1.5"/>')
        label = esc(labels[case_key])
        rotate = -38 if len(cases) > 6 else -25
        parts.append(f'<text x="{x:.2f}" y="{mt+plot_h+24}" transform="rotate({rotate} {x:.2f} {mt+plot_h+24})" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="13">{label}</text>')

    parts.append(f'<text x="{ml+plot_w/2}" y="{height-22}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Problem Size</text>')
    parts.append(f'<text x="24" y="{mt+plot_h/2}" transform="rotate(-90 24 {mt+plot_h/2})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Speedup vs RocketCore</text>')

    lx = width - 265
    ly = 68
    parts.append(f'<rect x="{lx-18}" y="{ly-24}" width="235" height="112" fill="#ffffff" stroke="#dddddd"/>')
    legend = [("RocketCore", "RocketCore 1x"), ("2x2", "Atik 2x2"), ("4x4", "Atik 4x4"), ("8x8", "Atik 8x8")]
    for i, (key, text) in enumerate(legend):
        y = ly + i * 25
        color = CONFIG_COLORS[key]
        if key == "RocketCore":
            parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{color}" stroke-width="3" stroke-dasharray="7 5"/>')
        else:
            parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{lx+45}" y="{y+5}" font-family="Helvetica,Arial,sans-serif" font-size="14">{esc(text)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_aggregate_bar(agg: dict[str, dict[str, float]]) -> str:
    workloads = [w for w in WORKLOAD_ORDER if w in agg]
    width = 1220
    height = 650
    ml, mr, mt, mb = 90, 40, 80, 125
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    max_speedup = max([1.0] + [agg[w][cfg] for w in workloads for cfg in ("2x2", "4x4", "8x8") if cfg in agg[w]])
    ymax = buffered_max(max_speedup)

    def y_for(v: float) -> float:
        return mt + plot_h - (v / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="40" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="27" font-weight="700">Aggregate Workload Speedup</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
    ]

    for i in range(6):
        val = ymax * i / 5
        y = y_for(val)
        parts.append(f'<line x1="{ml-6}" y1="{y:.2f}" x2="{ml+plot_w}" y2="{y:.2f}" stroke="#e7e7e7" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{y+5:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="14">{val:.0f}x</text>')

    yb = y_for(1.0)
    parts.append(f'<line x1="{ml}" y1="{yb:.2f}" x2="{ml+plot_w}" y2="{yb:.2f}" stroke="{CONFIG_COLORS["RocketCore"]}" stroke-width="2.5" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{ml+plot_w-4}" y="{yb-8:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="{CONFIG_COLORS["RocketCore"]}">RocketCore baseline 1x</text>')

    group_w = plot_w / max(1, len(workloads))
    bar_w = min(48, group_w / 5)
    for wi, workload in enumerate(workloads):
        cx = ml + group_w * wi + group_w / 2
        for bi, cfg in enumerate(("2x2", "4x4", "8x8")):
            speed = agg[workload].get(cfg)
            if speed is None:
                continue
            x = cx + (bi - 1) * (bar_w + 10) - bar_w / 2
            y = y_for(speed)
            h = mt + plot_h - y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{CONFIG_COLORS[cfg]}" stroke="#222" stroke-width="1"/>')
            parts.append(f'<text x="{x+bar_w/2:.2f}" y="{y-8:.2f}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" font-weight="700">{speed:.1f}x</text>')
        label = esc(WORKLOAD_TITLES.get(workload, workload).replace(" Benchmark", ""))
        parts.append(f'<text x="{cx:.2f}" y="{mt+plot_h+32}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="15">{label}</text>')

    parts.append(f'<text x="{ml+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Workload</text>')
    parts.append(f'<text x="24" y="{mt+plot_h/2}" transform="rotate(-90 24 {mt+plot_h/2})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Aggregated Speedup vs RocketCore</text>')

    lx = width - 245
    ly = 80
    parts.append(f'<rect x="{lx-18}" y="{ly-24}" width="215" height="112" fill="#ffffff" stroke="#dddddd"/>')
    legend = [("RocketCore", "RocketCore 1x"), ("2x2", "Atik 2x2"), ("4x4", "Atik 4x4"), ("8x8", "Atik 8x8")]
    for i, (key, text) in enumerate(legend):
        y = ly + i * 25
        color = CONFIG_COLORS[key]
        if key == "RocketCore":
            parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{color}" stroke-width="3" stroke-dasharray="7 5"/>')
        else:
            parts.append(f'<rect x="{lx}" y="{y-8}" width="34" height="14" fill="{color}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{lx+45}" y="{y+5}" font-family="Helvetica,Arial,sans-serif" font-size="14">{esc(text)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_aggregate_cycles_bar(cycles: dict[str, dict[str, float]]) -> str:
    workloads = [w for w in WORKLOAD_ORDER if w in cycles]
    width = 1320
    height = 720
    ml, mr, mt, mb = 105, 45, 82, 140
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    max_cycles = max(
        [1.0] + [cycles[w][cfg] for w in workloads for cfg in ("RocketCore", "2x2", "4x4", "8x8") if cfg in cycles[w]]
    )
    ymax = buffered_max(max_cycles)

    def y_for(v: float) -> float:
        return mt + plot_h - (v / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="40" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="27" font-weight="700">Aggregate Workload Cycles</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
    ]

    for i in range(6):
        val = ymax * i / 5
        y = y_for(val)
        parts.append(f'<line x1="{ml-6}" y1="{y:.2f}" x2="{ml+plot_w}" y2="{y:.2f}" stroke="#e7e7e7" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{y+5:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="14">{fmt_cycles(val)}</text>')

    group_w = plot_w / max(1, len(workloads))
    bar_w = min(42, group_w / 6)
    configs = ("RocketCore", "2x2", "4x4", "8x8")
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for wi, workload in enumerate(workloads):
        cx = ml + group_w * wi + group_w / 2
        tops = {}
        for offset, cfg in zip(offsets, configs):
            val = cycles[workload].get(cfg)
            if val is None:
                continue
            x = cx + offset * (bar_w + 8) - bar_w / 2
            y = y_for(val)
            h = mt + plot_h - y
            color = CONFIG_COLORS[cfg]
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" stroke="#222" stroke-width="1"/>')
            parts.append(f'<text x="{x+bar_w/2:.2f}" y="{y-7:.2f}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="11">{fmt_cycles(val)}</text>')
            tops[cfg] = (x + bar_w / 2, y, val)

        if "RocketCore" in tops:
            rx, ry, cpu_val = tops["RocketCore"]
            for cfg in ("2x2", "4x4", "8x8"):
                if cfg not in tops:
                    continue
                ax, ay, hw_val = tops[cfg]
                speed = cpu_val / hw_val if hw_val else 0.0
                label_x = (rx + ax) / 2
                label_y = min(ry, ay) - 20 - ({"2x2": 0, "4x4": 18, "8x8": 36}[cfg])
                parts.append(
                    f'<line x1="{rx:.2f}" y1="{ry:.2f}" x2="{ax:.2f}" y2="{ay:.2f}" '
                    f'stroke="{CONFIG_COLORS[cfg]}" stroke-width="2" stroke-dasharray="5 4"/>'
                )
                parts.append(
                    f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" '
                    f'font-family="Helvetica,Arial,sans-serif" font-size="12" font-weight="700" '
                    f'fill="{CONFIG_COLORS[cfg]}">{speed:.1f}x</text>'
                )

        label = esc(WORKLOAD_TITLES.get(workload, workload).replace(" Benchmark", ""))
        parts.append(f'<text x="{cx:.2f}" y="{mt+plot_h+35}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="15">{label}</text>')

    parts.append(f'<text x="{ml+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Workload</text>')
    parts.append(f'<text x="28" y="{mt+plot_h/2}" transform="rotate(-90 28 {mt+plot_h/2})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Aggregated Clock Cycles</text>')

    lx = width - 245
    ly = 84
    parts.append(f'<rect x="{lx-18}" y="{ly-24}" width="215" height="112" fill="#ffffff" stroke="#dddddd"/>')
    legend = [("RocketCore", "RocketCore"), ("2x2", "Atik 2x2"), ("4x4", "Atik 4x4"), ("8x8", "Atik 8x8")]
    for i, (key, text) in enumerate(legend):
        y = ly + i * 25
        parts.append(f'<rect x="{lx}" y="{y-8}" width="34" height="14" fill="{CONFIG_COLORS[key]}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{lx+45}" y="{y+5}" font-family="Helvetica,Arial,sans-serif" font-size="14">{esc(text)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_peak_speedup_bar(peak: dict[str, dict[str, float]]) -> str:
    workloads = [w for w in WORKLOAD_ORDER if w in peak]
    width = 1220
    height = 650
    ml, mr, mt, mb = 90, 40, 80, 125
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    max_speedup = max([1.0] + [peak[w][cfg] for w in workloads for cfg in ("2x2", "4x4", "8x8") if cfg in peak[w]])
    ymax = buffered_max(max_speedup)

    def y_for(v: float) -> float:
        return mt + plot_h - (v / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="40" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="27" font-weight="700">Peak Per-Case Speedup</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
    ]

    for i in range(6):
        val = ymax * i / 5
        y = y_for(val)
        parts.append(f'<line x1="{ml-6}" y1="{y:.2f}" x2="{ml+plot_w}" y2="{y:.2f}" stroke="#e7e7e7" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{y+5:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="14">{val:.0f}x</text>')

    yb = y_for(1.0)
    parts.append(f'<line x1="{ml}" y1="{yb:.2f}" x2="{ml+plot_w}" y2="{yb:.2f}" stroke="{CONFIG_COLORS["RocketCore"]}" stroke-width="2.5" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{ml+plot_w-4}" y="{yb-8:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="{CONFIG_COLORS["RocketCore"]}">RocketCore baseline 1x</text>')

    group_w = plot_w / max(1, len(workloads))
    bar_w = min(48, group_w / 5)
    for wi, workload in enumerate(workloads):
        cx = ml + group_w * wi + group_w / 2
        for bi, cfg in enumerate(("2x2", "4x4", "8x8")):
            speed = peak[workload].get(cfg)
            if speed is None:
                continue
            x = cx + (bi - 1) * (bar_w + 10) - bar_w / 2
            y = y_for(speed)
            h = mt + plot_h - y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{CONFIG_COLORS[cfg]}" stroke="#222" stroke-width="1"/>')
            parts.append(f'<text x="{x+bar_w/2:.2f}" y="{y-8:.2f}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" font-weight="700">{speed:.1f}x</text>')
        label = esc(WORKLOAD_TITLES.get(workload, workload).replace(" Benchmark", ""))
        parts.append(f'<text x="{cx:.2f}" y="{mt+plot_h+32}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="15">{label}</text>')

    parts.append(f'<text x="{ml+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Workload</text>')
    parts.append(f'<text x="24" y="{mt+plot_h/2}" transform="rotate(-90 24 {mt+plot_h/2})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Best Observed Speedup vs RocketCore</text>')

    lx = width - 245
    ly = 80
    parts.append(f'<rect x="{lx-18}" y="{ly-24}" width="215" height="112" fill="#ffffff" stroke="#dddddd"/>')
    legend = [("RocketCore", "RocketCore 1x"), ("2x2", "Atik 2x2"), ("4x4", "Atik 4x4"), ("8x8", "Atik 8x8")]
    for i, (key, text) in enumerate(legend):
        y = ly + i * 25
        color = CONFIG_COLORS[key]
        if key == "RocketCore":
            parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{color}" stroke-width="3" stroke-dasharray="7 5"/>')
        else:
            parts.append(f'<rect x="{lx}" y="{y-8}" width="34" height="14" fill="{color}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{lx+45}" y="{y+5}" font-family="Helvetica,Arial,sans-serif" font-size="14">{esc(text)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_peak_cycles_bar(peak_rows: dict[str, dict[str, dict]]) -> str:
    workloads = [w for w in WORKLOAD_ORDER if w in peak_rows]
    width = 1680
    height = 760
    ml, mr, mt, mb = 105, 45, 82, 165
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    max_cycles = max(
        [1.0]
        + [
            float(row[field])
            for w in workloads
            for row in peak_rows[w].values()
            for field in ("cpu_cycles", "hw_cycles")
        ]
    )
    min_cycles = min(
        [
            float(row[field])
            for w in workloads
            for row in peak_rows[w].values()
            for field in ("cpu_cycles", "hw_cycles")
            if float(row[field]) > 0
        ]
    )
    ymin = 10 ** math.floor(math.log10(min_cycles))
    ymax = 10 ** math.ceil(math.log10(buffered_max(max_cycles)))
    log_min = math.log10(ymin)
    log_span = math.log10(ymax) - log_min

    def y_for(v: float) -> float:
        return mt + plot_h - ((math.log10(max(v, ymin)) - log_min) / log_span) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="40" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="27" font-weight="700">Peak-Case Clock Cycles</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#222" stroke-width="2"/>',
    ]

    tick = ymin
    while tick <= ymax:
        y = y_for(tick)
        parts.append(f'<line x1="{ml-6}" y1="{y:.2f}" x2="{ml+plot_w}" y2="{y:.2f}" stroke="#e7e7e7" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{y+5:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="14">{fmt_cycles(tick)}</text>')
        tick *= 10

    group_w = plot_w / max(1, len(workloads))
    configs = ("2x2", "4x4", "8x8")
    for wi, workload in enumerate(workloads):
        gx = ml + group_w * wi
        present_cfgs = [cfg for cfg in configs if cfg in peak_rows[workload]]
        pair_w = group_w / max(1, len(present_cfgs))
        bar_w = min(24, pair_w / 4.5)
        for ci, cfg in enumerate(present_cfgs):
            row = peak_rows[workload].get(cfg)
            if row is None:
                continue
            pair_cx = gx + pair_w * ci + pair_w / 2
            cpu_val = float(row["cpu_cycles"])
            hw_val = float(row["hw_cycles"])
            speed = float(row["speedup"])
            cpu_x = pair_cx - bar_w - 4
            hw_x = pair_cx + 4
            cpu_y = y_for(cpu_val)
            hw_y = y_for(hw_val)
            base_y = y_for(ymin)
            cpu_h = base_y - cpu_y
            hw_h = base_y - hw_y

            parts.append(f'<rect x="{cpu_x:.2f}" y="{cpu_y:.2f}" width="{bar_w:.2f}" height="{cpu_h:.2f}" fill="{CONFIG_COLORS["RocketCore"]}" stroke="#222" stroke-width="1"/>')
            parts.append(f'<rect x="{hw_x:.2f}" y="{hw_y:.2f}" width="{bar_w:.2f}" height="{hw_h:.2f}" fill="{CONFIG_COLORS[cfg]}" stroke="#222" stroke-width="1"/>')
            parts.append(f'<text x="{cpu_x+bar_w/2:.2f}" y="{cpu_y-7:.2f}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="10">{fmt_cycles(cpu_val)}</text>')
            parts.append(f'<text x="{hw_x+bar_w/2:.2f}" y="{hw_y-7:.2f}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="10">{fmt_cycles(hw_val)}</text>')
            parts.append(
                f'<line x1="{cpu_x+bar_w/2:.2f}" y1="{cpu_y:.2f}" x2="{hw_x+bar_w/2:.2f}" y2="{hw_y:.2f}" '
                f'stroke="{CONFIG_COLORS[cfg]}" stroke-width="2" stroke-dasharray="5 4"/>'
            )
            parts.append(
                f'<text x="{pair_cx:.2f}" y="{min(cpu_y, hw_y)-20:.2f}" text-anchor="middle" '
                f'font-family="Helvetica,Arial,sans-serif" font-size="12" font-weight="700" '
                f'fill="{CONFIG_COLORS[cfg]}">{speed:.1f}x</text>'
            )
            parts.append(f'<text x="{pair_cx:.2f}" y="{mt+plot_h+32}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13">{cfg}</text>')

        label = esc(WORKLOAD_TITLES.get(workload, workload).replace(" Benchmark", ""))
        parts.append(f'<text x="{gx+group_w/2:.2f}" y="{mt+plot_h+62}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="15">{label}</text>')

    parts.append(f'<text x="{ml+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Workload</text>')
    parts.append(f'<text x="28" y="{mt+plot_h/2}" transform="rotate(-90 28 {mt+plot_h/2})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="600">Clock Cycles for Best-Speedup Case</text>')

    lx = width - 245
    ly = 84
    parts.append(f'<rect x="{lx-18}" y="{ly-24}" width="215" height="112" fill="#ffffff" stroke="#dddddd"/>')
    legend = [("RocketCore", "RocketCore"), ("2x2", "Atik 2x2"), ("4x4", "Atik 4x4"), ("8x8", "Atik 8x8")]
    for i, (key, text) in enumerate(legend):
        y = ly + i * 25
        parts.append(f'<rect x="{lx}" y="{y-8}" width="34" height="14" fill="{CONFIG_COLORS[key]}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{lx+45}" y="{y+5}" font-family="Helvetica,Arial,sans-serif" font-size="14">{esc(text)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def write_report(rows: list[dict], aggregate: dict[str, dict[str, float]]) -> None:
    lines = [
        "# UART Benchmark Analysis",
        "",
        "Generated from `software/uartlog/*/RESULT_JSON` records.",
        "",
        "## Inputs",
        "",
        "- `2x2`: `software/uartlog/atik_2x2`",
        "- `4x4`: `software/uartlog/atik_4x4`",
        "- `8x8`: `software/uartlog/atik_8x8_v2`",
        "",
        "The older `atik_8x8_v1` logs are not used because `atik_8x8_v2` contains the fuller matching workload set.",
        "",
        "## Figures",
        "",
        "- `figures/*_speedup.svg`: per-case speedup lines. The x-axis is problem size, and the y-axis is speedup over RocketCore.",
        "- `figures/aggregate_speedup.svg`: aggregate speedup by summing CPU cycles and hardware cycles over common problem shapes in each workload.",
        "- `figures/aggregate_cycles.svg`: aggregate cycle bars with speedup annotations from RocketCore to each accelerator config.",
        "- `figures/peak_speedup.svg`: best single-case speedup observed for each workload/config across all logged cases.",
        "- `figures/peak_cycles.svg`: matched RocketCore and accelerator clock cycles for each config's own best-speedup case.",
        "",
        "RocketCore is shown as a grey 1x baseline. Atik 2x2, 4x4, and 8x8 use progressively darker purple tones.",
        "",
        "## Aggregate Speedups",
        "",
        "| Workload | 2x2 | 4x4 | 8x8 |",
        "|---|---:|---:|---:|",
    ]
    for workload in WORKLOAD_ORDER:
        if workload not in aggregate:
            continue
        vals = aggregate[workload]
        lines.append(
            f"| {WORKLOAD_TITLES.get(workload, workload)} | "
            f"{vals.get('2x2', 0):.2f}x | {vals.get('4x4', 0):.2f}x | {vals.get('8x8', 0):.2f}x |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Speedup is computed as `cpu_cycles / hw_cycles`.")
    lines.append("- Aggregate speedup is computed as `sum(cpu_cycles) / sum(hw_cycles)` over problem shapes present in all three configs.")
    lines.append("- Only records with `status == PASS` are included in plots and aggregate calculations.")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows = [r for r in collect() if r.get("status") == "PASS"]
    write_csv(rows)

    by_workload = defaultdict(list)
    for row in rows:
        by_workload[row["workload"]].append(row)

    aggregate: dict[str, dict[str, float]] = {}
    aggregate_cycles: dict[str, dict[str, float]] = {}
    peak_speedups: dict[str, dict[str, float]] = {}
    peak_rows: dict[str, dict[str, dict]] = {}
    with (DATA / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            ["workload", "config", "included_cases", "cpu_cycles_sum", "hw_cycles_sum", "aggregate_speedup"],
        )
        writer.writeheader()
        for workload in WORKLOAD_ORDER:
            wr = by_workload.get(workload, [])
            if not wr:
                continue
            aggregate[workload] = {}
            aggregate_cycles[workload] = {}
            peak_speedups[workload] = {}
            peak_rows[workload] = {}

            for cfg in ("2x2", "4x4", "8x8"):
                all_cfg_rows = [r for r in wr if r["config"] == cfg]
                if all_cfg_rows:
                    peak_row = max(all_cfg_rows, key=lambda r: float(r["speedup"]))
                    peak_speedups[workload][cfg] = float(peak_row["speedup"])
                    peak_rows[workload][cfg] = peak_row

            case_keys_by_cfg = {
                cfg: {r["case_key"] for r in wr if r["config"] == cfg}
                for cfg in ("2x2", "4x4", "8x8")
            }
            common_case_keys = set.intersection(*(case_keys_by_cfg[cfg] for cfg in ("2x2", "4x4", "8x8")))
            cpu_rows = [r for r in wr if r["config"] == "2x2" and r["case_key"] in common_case_keys]
            aggregate_cycles[workload]["RocketCore"] = sum(int(r["cpu_cycles"]) for r in cpu_rows)
            for cfg in ("2x2", "4x4", "8x8"):
                cr = [r for r in wr if r["config"] == cfg and r["case_key"] in common_case_keys]
                if not cr:
                    continue
                cpu_sum = sum(int(r["cpu_cycles"]) for r in cr)
                hw_sum = sum(int(r["hw_cycles"]) for r in cr)
                speed = cpu_sum / hw_sum if hw_sum else 0.0
                aggregate[workload][cfg] = speed
                aggregate_cycles[workload][cfg] = hw_sum
                writer.writerow(
                    {
                        "workload": workload,
                        "config": cfg,
                        "included_cases": len(cr),
                        "cpu_cycles_sum": cpu_sum,
                        "hw_cycles_sum": hw_sum,
                        "aggregate_speedup": f"{speed:.6f}",
                    }
                )

    for workload in WORKLOAD_ORDER:
        wr = by_workload.get(workload, [])
        if wr:
            (FIG / f"{workload.replace('-', '_')}_speedup.svg").write_text(svg_line_plot(workload, wr))

    (FIG / "aggregate_speedup.svg").write_text(svg_aggregate_bar(aggregate))
    (FIG / "aggregate_cycles.svg").write_text(svg_aggregate_cycles_bar(aggregate_cycles))
    (FIG / "peak_speedup.svg").write_text(svg_peak_speedup_bar(peak_speedups))
    (FIG / "peak_cycles.svg").write_text(svg_peak_cycles_bar(peak_rows))
    write_report(rows, aggregate)

    print(f"parsed {len(rows)} passing RESULT_JSON records")
    print(f"wrote {FIG}")
    print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
