#!/usr/bin/env python3
"""Generate the CPU performance figures used by poster.tex.

Values are copied from the Performance table in the repository README.
Figures are drawn at roughly half the printed size and scaled up by LaTeX,
so font sizes here are about half of what appears on the poster.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"
DATASETS = ["chr1", "chr6", "E. coli", "HPRC v1.1", "HPRC v2.0", "HPRC v2.1"]
NAN = np.nan

# Fixed draw order, top to bottom within each dataset group. GFAz first so the
# eye lands on it before the baselines.
METHODS = ["GFAz", "GBZ", "sqz+bgzip", "sqz", "Zstd", "gzip"]

COLORS = {
    # Validated with the dataviz palette checker (all-pairs, light surface):
    # lightness band, chroma floor, CVD separation and normal-vision floor all
    # pass. sqz and sqz+bgzip share the amber hue as one tool family, split by
    # lightness. Do not hand-tweak without re-running the validator.
    "gzip": "#56B4E9",
    "Zstd": "#0072B2",
    "sqz": "#E69F00",
    "sqz+bgzip": "#8A6A00",
    "GBZ": "#009E73",
    "GFAz": "#9D2235",
}
INK = "#212a36"
MUTED = "#5b6673"
GRID = "#e2e6ea"
BAND = "#f5f6f8"

RATIO = {
    "gzip": [5.59, 5.04, 4.69, 4.02, 4.19, 4.19],
    "Zstd": [7.54, 6.99, 5.67, 5.32, 6.49, 6.43],
    "sqz": [3.09, 5.51, 1.26, NAN, NAN, NAN],
    "sqz+bgzip": [18.0, 20.8, 7.46, NAN, NAN, NAN],
    "GBZ": [9.52, 19.2, 5.58, 14.0, 66.8, 64.2],
    "GFAz": [35.4, 35.4, 18.4, 22.4, 83.8, 82.8],
}

COMPRESSION = {
    "gzip": [46.2, 41.0, 33.3, 36.4, 49.1, 48.9],
    "Zstd": [2178, 1712, 1356, 1657, 1514, 1540],
    "sqz": [3.95, 3.56, 4.57, NAN, NAN, NAN],
    "sqz+bgzip": [3.97, 3.56, 4.53, NAN, NAN, NAN],
    "GBZ": [12.1, 10.7, 20.2, 84.5, 130, 136],
    "GFAz": [1320, 1355, 226, 291, 555, 538],
}

DECOMPRESSION = {
    "gzip": [359, 348, 310, 319, 342, 343],
    "Zstd": [1618, 1515, 1258, 1234, 1240, 1241],
    "sqz": [21.6, 20.1, 34.0, NAN, NAN, NAN],
    "sqz+bgzip": [21.4, 20.4, 32.2, NAN, NAN, NAN],
    "GBZ": [284, 281, 197, 650, 648, 652],
    "GFAz": [2307, 2943, 834, 2292, 5426, 5325],
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def row_bands(ax, base):
    """Alternating background bands so each dataset group reads as one unit."""
    for i, y in enumerate(base):
        if i % 2:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)


def legend_strip(fig, ax, ncol=6, y=0.5):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=COLORS[m], markeredgecolor="white",
                   markeredgewidth=0.8, label=m)
        for m in reversed(METHODS)
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=False, fontsize=10, handletextpad=0.35,
        columnspacing=1.3, borderaxespad=0,
    )


def bar_ratio(output):
    """Compression ratio: grouped horizontal bars, linear axis."""
    fig, ax = plt.subplots(figsize=(9.0, 3.17))
    base = np.arange(len(DATASETS))[::-1]
    row_bands(ax, base)

    n = len(METHODS)
    h = 0.78 / n
    for k, method in enumerate(METHODS):
        vals = np.asarray(RATIO[method], dtype=float)
        # METHODS[0] sits at the top of each group.
        y = base + 0.39 - h / 2 - k * h
        ax.barh(y, np.nan_to_num(vals), height=h * 0.86,
                color=COLORS[method], zorder=3,
                edgecolor="white", linewidth=0.5)

    # Label GFAz on every row, plus the strongest baseline for contrast.
    gfaz = np.asarray(RATIO["GFAz"], dtype=float)
    for i, (yb, v) in enumerate(zip(base, gfaz)):
        ax.text(v + 1.2, yb + 0.39 - h / 2, f"{v:g}×", va="center",
                ha="left", fontsize=11, fontweight="bold",
                color=COLORS["GFAz"], zorder=5)
        rivals = {m: RATIO[m][i] for m in METHODS[1:]
                  if np.isfinite(RATIO[m][i])}
        bm = max(rivals, key=rivals.get)
        k = METHODS.index(bm)
        ax.text(rivals[bm] + 1.2, yb + 0.39 - h / 2 - k * h,
                f"{rivals[bm]:g}×", va="center", ha="left",
                fontsize=9, color=MUTED, zorder=5)

    ax.set_yticks(base, DATASETS, fontweight="bold")
    ax.set_xlabel("compression ratio  (uncompressed / compressed)")
    ax.set_xlim(0, 96)
    ax.set_ylim(-0.55, len(DATASETS) - 0.45)
    ax.grid(axis="x", color=GRID, linewidth=1.0)
    ax.set_axisbelow(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)

    legend_strip(fig, ax, ncol=6, y=0.999)
    fig.subplots_adjust(left=0.118, right=0.995, bottom=0.190, top=0.845)
    fig.savefig(OUT / output)
    plt.close(fig)


def dot_panel(ax, data, xlabel, title):
    base = np.arange(len(DATASETS))[::-1]
    row_bands(ax, base)

    n = len(METHODS)
    step = 0.62 / (n - 1)
    for k, method in enumerate(METHODS):
        vals = np.asarray(data[method], dtype=float)
        mask = np.isfinite(vals)
        y = base + 0.31 - k * step
        big = method == "GFAz"
        ax.scatter(vals[mask], y[mask], s=135 if big else 62,
                   color=COLORS[method], edgecolor="white",
                   linewidth=1.0, zorder=6 if big else 4)

    gfaz = np.asarray(data["GFAz"], dtype=float)
    for i, (yb, v) in enumerate(zip(base, gfaz)):
        ax.annotate(f"{v:g}", (v, yb + 0.31), xytext=(9, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=10.5, fontweight="bold", color=COLORS["GFAz"],
                    zorder=7)

    ax.set_yticks(base, DATASETS, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_xscale("log")
    ax.set_xlim(2, 2.4e4)
    ax.set_ylim(-0.55, len(DATASETS) - 0.45)
    ax.grid(axis="x", color=GRID, linewidth=1.0)
    ax.set_axisbelow(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)


def throughput(output):
    """Compression and decompression throughput share one legend."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.09), sharey=True)
    dot_panel(axes[0], COMPRESSION, "MiB/s  ·  log scale", "Compression")
    dot_panel(axes[1], DECOMPRESSION, "MiB/s  ·  log scale",
              "Decompression")
    fig.subplots_adjust(left=0.118, right=0.995, bottom=0.200, top=0.865,
                        wspace=0.07)
    fig.savefig(OUT / output)
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    bar_ratio("cpu_ratio.pdf")
    throughput("cpu_throughput.pdf")


if __name__ == "__main__":
    main()
