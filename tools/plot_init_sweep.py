#!/usr/bin/env python3
"""
plot_init_sweep.py

Heatmaps of the RMS error over the initial guess sweep made with make_init_sweep.py:
initial position offset (rows) x initial orientation offset (columns), one panel per
body, every cell the mean over all bags and seeds of that (pos, ang) pair. All panels
share one color scale so the bodies can be compared.

    python tools/plot_init_sweep.py [results/init_sweep/results.csv] [--metric pos|vel|att]
                                    [--bodies Drone2 Drone3 Payload] [--output FILE] [--show]

Needs a Python with matplotlib, e.g. /opt/anaconda3/bin/python.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# column suffix in the results CSV, axis label
METRICS = {
    "pos": ("_pos_rms_m", "RMS position error [m]"),
    "vel": ("_vel_rms_mps", "RMS velocity error [m/s]"),
    "att": ("_att_rms_deg", "RMS attitude error [deg]"),
}

# experiment names written by make_init_sweep.py, e.g. "pos0.15_ang5"
LABEL = re.compile(r"^pos(?P<pos>[0-9.]+)_ang(?P<ang>[0-9.]+)$")

# single-hue sequential ramp, light (low error) -> dark (high error)
CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

SURFACE = "#ffffff"
INK = "#1f1f1f"
MUTED = "#6b6b6b"


def load_runs(csv_file):
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    runs, skipped = [], defaultdict(int)
    for row in rows:
        m = LABEL.match(row["experiment"])
        if not m:
            skipped["not a sweep experiment"] += 1
        elif row["status"] != "ok":
            skipped["failed"] += 1
        else:
            runs.append((float(m["pos"]), float(m["ang"]), row))
    return runs, rows[0].keys() if rows else [], skipped


def cell_means(runs, column):
    # (pos, ang) -> mean over every bag and seed of that pair
    values = defaultdict(list)
    for pos, ang, row in runs:
        if row[column] != "":
            values[(pos, ang)].append(float(row[column]))
    positions = sorted({p for p, _ in values})
    angles = sorted({a for _, a in values})
    grid = np.full((len(positions), len(angles)), np.nan)
    counts = np.zeros_like(grid, dtype=int)
    for (pos, ang), v in values.items():
        grid[positions.index(pos), angles.index(ang)] = np.mean(v)
        counts[positions.index(pos), angles.index(ang)] = len(v)
    return positions, angles, grid, counts


def draw_panel(ax, positions, angles, grid, title, vmax):
    # pcolormesh on the cell indices (not the offsets) so every cell has the same size
    mesh = ax.pcolormesh(grid, cmap=CMAP, vmin=0.0, vmax=vmax, edgecolors=SURFACE, linewidth=2)
    for i in range(len(positions)):
        for j in range(len(angles)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j + 0.5, i + 0.5, "-", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            dark_cell = v / vmax > 0.55
            ax.text(j + 0.5, i + 0.5, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color=SURFACE if dark_cell else INK)

    ax.set_xticks(np.arange(len(angles)) + 0.5)
    ax.set_xticklabels([f"{a:g}" for a in angles])
    ax.set_yticks(np.arange(len(positions)) + 0.5)
    ax.set_yticklabels([f"{p:g}" for p in positions])
    ax.set_xlabel("Initial orientation offset [deg]", color=MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_aspect("equal")
    ax.tick_params(length=0, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="results/init_sweep/results.csv")
    parser.add_argument("--metric", choices=METRICS, default="pos")
    parser.add_argument("--bodies", nargs="+", help="e.g. Drone2 Payload (default: every body in the CSV)")
    parser.add_argument("--output", help="image file (default: init_sweep_<metric>_heatmap.png next to the CSV)")
    parser.add_argument("--show", action="store_true", help="also open the figure")
    args = parser.parse_args()

    suffix, unit_label = METRICS[args.metric]
    runs, columns, skipped = load_runs(args.csv)
    if not runs:
        raise SystemExit(f"{args.csv}: no successful sweep runs found")
    for reason, n in skipped.items():
        print(f"skipped {n} row(s): {reason}")

    available = [c[: -len(suffix)] for c in columns if c.endswith(suffix)]
    bodies = args.bodies or available
    unknown = [b for b in bodies if b not in available]
    if unknown:
        raise SystemExit(f"unknown bodies {unknown}, the CSV has {available}")

    panels = [(b, *cell_means(runs, b + suffix)) for b in bodies]
    vmax = max(np.nanmax(grid) for _, _, _, grid, _ in panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1.2, 4.4), squeeze=False,
                             facecolor=SURFACE)
    axes = axes[0]
    for ax, (body, positions, angles, grid, counts) in zip(axes, panels):
        mesh = draw_panel(ax, positions, angles, grid, body, vmax)
        print(f"{body}: {counts.min()}-{counts.max()} runs per cell")
    axes[0].set_ylabel("Initial position offset [m]", color=MUTED)

    cbar = fig.colorbar(mesh, ax=list(axes), shrink=0.8, pad=0.02)
    cbar.set_label(unit_label, color=MUTED)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, colors=MUTED)

    bags = sorted({row["bag"] for _, _, row in runs})
    seeds = sorted({int(row["seed"]) for _, _, row in runs})
    fig.suptitle(f"{unit_label.split(' [')[0]} vs. initial guess offset", x=0.02, ha="left",
                 color=INK, fontsize=13)
    fig.text(0.02, 0.89, f"Mean per cell over bags {', '.join(bags)} and seeds {seeds}",
             ha="left", color=MUTED, fontsize=9)

    output = Path(args.output) if args.output else Path(args.csv).parent / f"init_sweep_{args.metric}_heatmap.png"
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
