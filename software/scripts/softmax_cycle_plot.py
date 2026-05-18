#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class SoftmaxPoint:
    total_size: int
    detail_label: str
    sw_cycles: int
    hw_cycles: int
    pass1_cycles: int
    pass2_cycles: int
    mismatches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot softmax software and hardware cycle counts from a girdap log."
    )
    parser.add_argument(
        "logfile",
        type=Path,
        help="Path to the softmax log text file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <logfile_stem>_softmax_cycles.png next to the log.",
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
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot both axes on a logarithmic scale to reduce clutter for wide size ranges.",
    )
    return parser.parse_args()


def parse_log(log_path: Path) -> tuple[str | None, list[SoftmaxPoint]]:
    title = None
    points: list[SoftmaxPoint] = []

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

            try:
                if len(parts) == 8:
                    size = int(parts[1])
                    sw_cycles = int(parts[2])
                    hw_cycles = int(parts[3])
                    pass1_cycles = int(parts[4])
                    pass2_cycles = int(parts[5])
                    mismatches = int(parts[7])
                    points.append(
                        SoftmaxPoint(
                            total_size=size,
                            detail_label=f"N={size}",
                            sw_cycles=sw_cycles,
                            hw_cycles=hw_cycles,
                            pass1_cycles=pass1_cycles,
                            pass2_cycles=pass2_cycles,
                            mismatches=mismatches,
                        )
                    )
                    continue

                if len(parts) == 10:
                    rows = int(parts[2])
                    cols = int(parts[3])
                    sw_cycles = int(parts[4])
                    hw_cycles = int(parts[5])
                    pass1_cycles = int(parts[6])
                    pass2_cycles = int(parts[7])
                    mismatches = int(parts[9])
                    points.append(
                        SoftmaxPoint(
                            total_size=rows * cols,
                            detail_label=f"{rows}x{cols}",
                            sw_cycles=sw_cycles,
                            hw_cycles=hw_cycles,
                            pass1_cycles=pass1_cycles,
                            pass2_cycles=pass2_cycles,
                            mismatches=mismatches,
                        )
                    )
                    continue
            except ValueError:
                continue

    points.sort(key=lambda point: point.total_size)
    return title, points


def build_output_path(log_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    # return log_path.with_name(f"{log_path.stem}_softmax_cycles.png")
    return log_path.with_name(f"{log_path.stem}_cycles.png")


def warn_on_mismatches(points: list[SoftmaxPoint]) -> None:
    bad_points = [point for point in points if point.mismatches != 0]
    if not bad_points:
        return

    labels = ", ".join(
        f"{point.total_size} ({point.detail_label})" for point in bad_points
    )
    print(
        "Warning: non-zero mismatches were reported for "
        f"{len(bad_points)} case(s): {labels}"
    )


def main() -> None:
    args = parse_args()
    title_from_log, points = parse_log(args.logfile)

    if not points:
        raise SystemExit(
            "No softmax CSV_DATA rows were found. Expected lines like "
            "'CSV_DATA,<size>,<sw>,<hw_e2e>,<pass1>,<pass2>,<speedup>,<mismatches>' "
            "or "
            "'CSV_DATA,<case>,<rows>,<cols>,<sw>,<hw_e2e>,<pass1>,<pass2>,<speedup>,<mismatches>'."
        )

    output_path = build_output_path(args.logfile, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [point.total_size for point in points]
    sw_cycles = [point.sw_cycles for point in points]
    hw_cycles = [point.hw_cycles for point in points]
    pass1_cycles = [point.pass1_cycles for point in points]
    pass2_cycles = [point.pass2_cycles for point in points]

    plt.figure(figsize=(11, 6))
    plt.plot(
        x_values,
        sw_cycles,
        marker="o",
        linewidth=2,
        color="#c0392b",
        label="Software cycles",
        zorder=4,
    )
    plt.plot(
        x_values,
        hw_cycles,
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Hardware e2e cycles",
        zorder=4,
    )
    # plt.plot(
    #     x_values,
    #     pass1_cycles,
    #     marker="^",
    #     linewidth=1.6,
    #     linestyle="--",
    #     color="#16a085",
    #     label="Hardware pass1 cycles",
    #     zorder=3,
    # )
    # plt.plot(
    #     x_values,
    #     pass2_cycles,
    #     marker="v",
    #     linewidth=1.6,
    #     linestyle="--",
    #     color="#8e44ad",
    #     label="Hardware pass2 cycles",
    #     zorder=3,
    # )

    y_min = min(min(sw_cycles), min(hw_cycles), min(pass1_cycles), min(pass2_cycles))
    y_max = max(max(sw_cycles), max(hw_cycles), max(pass1_cycles), max(pass2_cycles))
    y_span = max(y_max - y_min, 1)

    if args.log_scale:
      plt.xscale("log")
      plt.yscale("log")

    for x_pos, point in zip(x_values, points):
        faster = (float(point.sw_cycles) / float(point.hw_cycles)) if point.hw_cycles != 0 else float("inf")
        low_y = min(point.sw_cycles, point.hw_cycles)
        high_y = max(point.sw_cycles, point.hw_cycles)
        if args.log_scale:
            mid_y = (float(point.sw_cycles) * float(point.hw_cycles)) ** 0.5
            label_y = min(high_y * 1.18, y_max * 1.10)
        else:
            mid_y = (point.sw_cycles + point.hw_cycles) / 2.0
            label_y = min(high_y + 0.05 * y_span, y_max + 0.10 * y_span)

        plt.plot(
            [x_pos, x_pos],
            [low_y, high_y],
            linestyle="--",
            linewidth=1.1,
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
            zorder=5,
        )
        if not args.log_scale:
            plt.annotate(
                point.detail_label,
                xy=(x_pos, low_y),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color="#555555",
                rotation=0,
                zorder=5,
            )

    if not args.log_scale:
        plt.xticks(x_values, [str(value) for value in x_values])
    plt.xlabel("Total data size (elements)")
    plt.ylabel("Cycles")
    plt.title(args.title or title_from_log or "Softmax Cycle Comparison")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5, which="both" if args.log_scale else "major")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to '{output_path}'")

    warn_on_mismatches(points)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
