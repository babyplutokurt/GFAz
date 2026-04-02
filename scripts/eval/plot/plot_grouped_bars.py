#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Paper-ish matplotlib style
# -------------------------
def set_paper_style():
    plt.rcParams.update({
        "font.size": 8,
        "font.family": "serif",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "figure.dpi": 300,
    })


def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Compressor", "Ratio", "Compression_Speed_MBps", "Decompression_Speed_MBps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.copy()
    df["Dataset"] = Path(path).stem
    return df


def load_all(csv_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files match: {csv_glob}")
    dfs = [read_one_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def default_order(compressors):
    """
    Give a stable, nice order (yours last, CPU then GPU).
    Anything not listed keeps alphabetical order after the known ones.
    """
    preferred = [
        "Gzip",
        "Zstd",
        "GBZ",
        "sqz",
        "sqz+bgzip",
        "sqz+Zstd",
        "gfaz(CPU)",
        "gfaz(GPU)",
    ]
    preferred_set = set(preferred)
    rest = sorted([c for c in compressors if c not in preferred_set])
    # keep only those that actually exist
    preferred_present = [c for c in preferred if c in compressors]
    return preferred_present + rest


def plot_grouped_bars(
    df_long: pd.DataFrame,
    value_col: str,
    ylabel: str,
    outpath: str,
    compressor_order=None,
    dataset_order=None,
    logy=False,
):
    # Pivot: rows=datasets, cols=compressors, values=value
    pivot = df_long.pivot_table(index="Dataset", columns="Compressor", values=value_col, aggfunc="first")

    if dataset_order is None:
        dataset_order = list(pivot.index)
    pivot = pivot.reindex(dataset_order)

    if compressor_order is None:
        compressor_order = default_order(list(pivot.columns))
    pivot = pivot.reindex(columns=compressor_order)

    datasets = list(pivot.index)
    compressors = list(pivot.columns)

    # Figure size tuned for 2-column papers: ~7 inches wide
    fig, ax = plt.subplots(figsize=(7.0, 2.4))

    n_groups = len(datasets)
    n_bars = len(compressors)

    x = np.arange(n_groups)
    total_width = 0.86
    bar_w = total_width / max(n_bars, 1)

    # Colors: let matplotlib pick defaults; we only control hatching for GPU/CPU distinction
    # CPU/GPU: hatch GPU bars for print-friendly distinction
    def hatch_for(name: str) -> str | None:
        if "(GPU)" in name:
            return "///"
        return None

    # Plot bars
    for i, comp in enumerate(compressors):
        vals = pivot[comp].to_numpy(dtype=float)

        # handle missing values: NaN -> no bar
        mask = np.isfinite(vals)
        xpos = x - total_width/2 + (i + 0.5) * bar_w

        bars = ax.bar(
            xpos[mask],
            vals[mask],
            width=bar_w,
            label=comp,
            hatch=hatch_for(comp),
            edgecolor="black",
            linewidth=0.3,
        )

        # Optional: mark missing entries with "OOM/NA" text
        # If you want "OOM" specifically, you could encode missing as empty in CSV
        for j, ok in enumerate(mask):
            if not ok:
                ax.text(
                    xpos[j],
                    0,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    clip_on=True
                )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")

    # light y-grid for readability
    ax.yaxis.grid(True, linewidth=0.3)
    ax.set_axisbelow(True)

    if logy:
        ax.set_yscale("log")

    # Legend: put on top, multi-column
    ncol = min(4, max(1, int(np.ceil(len(compressors) / 2))))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=ncol, handlelength=1.2, columnspacing=1.0)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot compressor evaluation from multiple per-dataset CSVs.")
    parser.add_argument("--csv_glob", type=str, default="csv/*.csv",
                        help="Glob for CSV files (default: csv/*.csv)")
    parser.add_argument("--outdir", type=str, default="figs",
                        help="Output directory (default: figs)")
    parser.add_argument("--log_speed", action="store_true",
                        help="Use log scale for speed plots (sometimes helpful if values span a lot)")
    args = parser.parse_args()

    set_paper_style()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_all(args.csv_glob)

    # Basic cleanup: strip compressor names
    df["Compressor"] = df["Compressor"].astype(str).str.strip()
    df["Dataset"] = df["Dataset"].astype(str)

    # Ratio
    plot_grouped_bars(
        df, "Ratio", "Compression ratio (×)",
        os.path.join(args.outdir, "fig_ratio.pdf"),
        logy=False,
    )

    # Compression speed
    plot_grouped_bars(
        df, "Compression_Speed_MBps", "Compression throughput (MB/s)",
        os.path.join(args.outdir, "fig_comp_speed.pdf"),
        logy=args.log_speed,
    )

    # Decompression speed
    plot_grouped_bars(
        df, "Decompression_Speed_MBps", "Decompression throughput (MB/s)",
        os.path.join(args.outdir, "fig_decomp_speed.pdf"),
        logy=args.log_speed,
    )

    print(f"Saved figures to: {args.outdir}/ (PDF)")


if __name__ == "__main__":
    main()
