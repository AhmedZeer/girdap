#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GEMM software and hardware cycle counts from a girdap log."
    )
    parser.add_argument(
        "logfile",
        type=Path,
        help="Path to the GEMM log text file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <logfile_stem>_gemm_cycles.png next to the log.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the plot window after saving.",
    )
    return parser.parse_args()


def parse_log(log_path: Path) -> tuple[str | None, list[str], list[int], list[int]]:
    title = None
    labels: list[str] = []
    sw_cycles: list[int] = []
    hw_cycles: list[int] = []

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if title is None and line.startswith("==="):
                title = line.strip("= ").strip()

            if not line.startswith("CSV_DATA,"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 8:
                continue

            try:
                m_dim = int(parts[2])
                n_dim = int(parts[3])
                k_dim = int(parts[4])
                sw = int(parts[5])
                hw = int(parts[6])
            except ValueError:
                continue

            labels.append(f"{m_dim}x{n_dim}x{k_dim}")
            sw_cycles.append(sw)
            hw_cycles.append(hw)

    return title, labels, sw_cycles, hw_cycles


def build_output_path(log_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return log_path.with_name(f"{log_path.stem}_gemm_cycles.png")


def main() -> None:
    args = parse_args()
    title_from_log, labels, sw_cycles, hw_cycles = parse_log(args.logfile)

    if not labels:
        raise SystemExit(
            "No GEMM CSV_DATA rows were found. Expected lines like "
            "'CSV_DATA,case,M,N,K,sw_cycles,hw_e2e_cycles,...'."
        )

    output_path = build_output_path(args.logfile, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = list(range(len(labels)))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_positions,
        sw_cycles,
        marker="o",
        linewidth=2,
        color="#c0392b",
        label="Software cycles",
        zorder=3,
    )
    plt.plot(
        x_positions,
        hw_cycles,
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Hardware e2e cycles",
        zorder=3,
    )

    y_min = min(min(sw_cycles), min(hw_cycles))
    y_max = max(max(sw_cycles), max(hw_cycles))
    y_span = max(y_max - y_min, 1)

    for x_pos, sw, hw in zip(x_positions, sw_cycles, hw_cycles):
        faster = (float(sw) / float(hw)) if hw != 0 else float("inf")
        low_y = min(sw, hw)
        high_y = max(sw, hw)
        mid_y = (sw + hw) / 2.0
        label_y = min(high_y + 0.04 * y_span, y_max + 0.08 * y_span)

        plt.plot(
            [x_pos, x_pos],
            [low_y, high_y],
            linestyle="--",
            linewidth=1.2,
            color="#7f8c8d",
            alpha=0.85,
            zorder=1,
        )
        plt.annotate(
            f"{faster:.1f}x",
            xy=(x_pos, mid_y),
            xytext=(x_pos, label_y),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#34495e",
            arrowprops={
                "arrowstyle": "-",
                "color": "#7f8c8d",
                "lw": 1.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )

    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.xlabel("GEMM size (M x N x K)")
    plt.ylabel("Cycles")
    plt.title(args.title or title_from_log or "GEMM Cycle Comparison")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to '{output_path}'")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
