#!/usr/bin/env python3
"""
Plot Pareto-style figures: Throughput vs Compression Ratio.
Based on the 6 CSV files, create a combined 2-panel plot:
[Compression Speed vs Ratio] | [Decompression Speed vs Ratio]

Normalized to Zstd (Zstd = 1.0).
Each compressor has a distinct color. Each dataset uses a distinct marker.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------
# Configuration
# -------------------------
COLOR_MAP = {
    "Gzip":       "#fde0dd",  # Gray-ish
    "Zstd":       "#bdbdbd",  # Blue-ish
    "GBZ":        "#fc9272",  # Orange-ish
    "sqz":        "#9ecae1",  # Purple-ish
    "sqz+bgzip":  "#3182bd",  # Brown-ish
    "sqz+Zstd":   "#1c9099",  # Red-ish
    "gfaz(CPU)":  "#bcbddc",  # Pink-ish
    "gfaz(GPU)":  "#addd8e",  # Green-ish
}

MARKER_MAP = {
    "chr1": "o",       # Circle
    "chr6": "s",       # Square
    "Ecoli": "^",      # Triangle Up
    "HPRCv11": "D",    # Diamond
    "HPRCv20": "P",    # Plus (filled)
    "HPRCv21": "X",    # X (filled)
}

# -------------------------
# Style
# -------------------------
def set_paper_style():
    plt.rcParams.update({
        "font.size": 8,
        "font.family": "serif",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "figure.dpi": 300,
    })

# -------------------------
# Data Loading
# -------------------------
def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Compressor", "Ratio", "Compression_Speed_MBps", "Decompression_Speed_MBps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    
    df = df.copy()
    df["Compressor"] = df["Compressor"].astype(str).str.strip()
    df["Dataset"] = Path(path).stem
    return df

def load_all(csv_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files match: {csv_glob}")
    dfs = [read_one_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    
    # ensure numeric
    for col in ["Ratio", "Compression_Speed_MBps", "Decompression_Speed_MBps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def default_order(compressors):
    preferred = [
        "Gzip",
        "Zstd",
        "sqz",
        "sqz+bgzip",
        "sqz+Zstd",
        "GBZ",
        "gfaz(CPU)",
        "gfaz(GPU)",
    ]
    preferred_set = set(preferred)
    rest = sorted([c for c in compressors if c not in preferred_set])
    preferred_present = [c for c in preferred if c in compressors]
    return preferred_present + rest

# -------------------------
# Normalization
# -------------------------
def normalize_to_zstd(df: pd.DataFrame, zstd_name="Zstd") -> pd.DataFrame:
    if "Dataset" not in df.columns:
        raise RuntimeError("Internal error: 'Dataset' column missing after load_all().")

    # baseline per dataset
    base = df[df["Compressor"] == zstd_name][
        ["Dataset", "Ratio", "Compression_Speed_MBps", "Decompression_Speed_MBps"]
    ].copy()

    # if duplicates exist, keep first
    base = base.drop_duplicates(subset=["Dataset"], keep="first").rename(columns={
        "Ratio": "Ratio_zstd",
        "Compression_Speed_MBps": "Compression_Speed_zstd",
        "Decompression_Speed_MBps": "Decompression_Speed_zstd",
    })

    out = df.merge(base, on="Dataset", how="left")

    # drop datasets without baseline
    missing_ds = out[out["Ratio_zstd"].isna()]["Dataset"].unique().tolist()
    if missing_ds:
        print(f"[WARN] Dropping datasets without '{zstd_name}' baseline row: {missing_ds}")
        out = out[~out["Dataset"].isin(missing_ds)].copy()

    # avoid divide-by-zero
    out.loc[out["Ratio_zstd"] == 0, "Ratio_zstd"] = np.nan
    out.loc[out["Compression_Speed_zstd"] == 0, "Compression_Speed_zstd"] = np.nan
    out.loc[out["Decompression_Speed_zstd"] == 0, "Decompression_Speed_zstd"] = np.nan

    out["Ratio_norm"] = out["Ratio"] / out["Ratio_zstd"]
    out["Compression_Speed_norm"] = out["Compression_Speed_MBps"] / out["Compression_Speed_zstd"]
    out["Decompression_Speed_norm"] = out["Decompression_Speed_MBps"] / out["Decompression_Speed_zstd"]

    return out

# -------------------------
# Plotting
# -------------------------
def draw_scatter(
    ax,
    df: pd.DataFrame,
    x_col: str,
    xlabel: str,
    compressor_order: list,
    logx=True
):
    datasets = df["Dataset"].unique()
    
    for comp in compressor_order:
        comp_color = COLOR_MAP.get(comp, "#333333")
        
        # Filter for this compressor
        sub = df[df["Compressor"] == comp]
        
        for idx, row in sub.iterrows():
            ds = row["Dataset"]
            marker = MARKER_MAP.get(ds, "o")
            
            x_val = row[x_col]
            y_val = row["Ratio_norm"]
            
            if pd.isna(x_val) or pd.isna(y_val) or x_val <= 0 or y_val <= 0:
                continue
            
            ax.scatter(
                x_val, y_val,
                c=comp_color,
                marker=marker,
                edgecolors="black",
                linewidth=0.5,
                s=60,
                alpha=0.9,
                label=comp 
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Compression Ratio / Zstd")
    
    if logx:
        ax.set_xscale("log", base=2)
    
    # Use log scale for Y (Ratio) as well, so the y=1/x curve appears as a straight line
    ax.set_yscale("log", base=2)
    
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

    # Baseline indicators (Zstd = 1.0)
    # Baseline indicators (Zstd = 1.0)
    ax.axhline(1.0, color="#666666", linestyle="-.", linewidth=0.8, zorder=1)
    ax.axvline(1.0, color="#666666", linestyle="-.", linewidth=0.8, zorder=1)
    
    # ISO-efficiency curve: Ratio * Speed = 1 => y = 1/x
    # Trade-off: Ratio increases 2x, Speed drops 2x
    xlim = ax.get_xlim()
    # Generate points across the visible range
    # Ensure xlim is positive (since log scale)
    x_min = max(xlim[0], 1e-6) 
    x_max = xlim[1]
    
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 100, base=10)
    y_line = 1.0 / x_line
    
    ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0, label='_nolegend_', zorder=1)
    
    # Restore limits in case plot command expanded them
    ax.set_xlim(xlim)


def plot_combined_pareto(df, outpath, compressor_order):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), constrained_layout=False)
    
    # 1. Comp Speed vs Ratio
    draw_scatter(
        axes[0],
        df,
        "Compression_Speed_norm",
        "Compression Throughput / Zstd",
        compressor_order,
        logx=True
    )
    
    # 2. Decomp Speed vs Ratio
    draw_scatter(
        axes[1],
        df,
        "Decompression_Speed_norm",
        "Decompression Throughput / Zstd",
        compressor_order,
        logx=True
    )
    
    # ---------------------------
    # Unified Legend
    # ---------------------------
    datasets = df["Dataset"].unique()

    # 1. Compressor Legend Items
    comp_handles = []
    for comp in compressor_order:
        color = COLOR_MAP.get(comp, "#333333")
        h = Line2D([0], [0], marker='o', color='w', label=comp,
                   markerfacecolor=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        comp_handles.append(h)
    
    # 2. Dataset Legend Items
    desired_ds_order = ["chr1", "chr6", "Ecoli", "HPRCv11", "HPRCv20", "HPRCv21"]
    ds_legend_items = [d for d in desired_ds_order if d in datasets]
    ds_legend_items += [d for d in unique_datasets(df) if d not in ds_legend_items]
    
    ds_handles = []
    for ds in ds_legend_items:
        mk = MARKER_MAP.get(ds, "o")
        h = Line2D([0], [0], marker=mk, color='w', label=ds,
                   markerfacecolor='white', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        h.set_markerfacecolor('#777777')
        ds_handles.append(h)
        
    # Combine all handles and labels
    all_handles = comp_handles + ds_handles
    
    # Place legend outside right
    # (x, y) coordinates for bbox_to_anchor are relative to figure size
    fig.legend(
        handles=all_handles,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),
        title="Legend",
        fontsize=7,
        title_fontsize=8
    )
    
    # Adjust layout to make room for legend on the right
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    fig.savefig(outpath)
    plt.close(fig)


def unique_datasets(df):
    return sorted(df["Dataset"].unique())


def main():
    parser = argparse.ArgumentParser(description="Plot Compression/Decompression Performance vs Ratio.")
    parser.add_argument("--csv_glob", type=str, default="csv/*.csv",
                        help="Glob for per-dataset CSV files (default: csv/*.csv)")
    parser.add_argument("--outdir", type=str, default="figs_pareto",
                        help="Output directory (default: figs_pareto)")
    parser.add_argument("--zstd_name", type=str, default="Zstd",
                        help="Exact compressor name used for baseline row (default: Zstd)")
    args = parser.parse_args()

    set_paper_style()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_all(args.csv_glob)
    
    # Normalize!
    df = normalize_to_zstd(df, zstd_name=args.zstd_name)
    
    # Determine compressor order
    unique_comps = df["Compressor"].unique()
    comp_order = default_order(unique_comps)
    
    outpath = os.path.join(args.outdir, "fig_pareto_combined.pdf")
    plot_combined_pareto(df, outpath, comp_order)

    print(f"Saved combined pareto figure to: {outpath}")

if __name__ == "__main__":
    main()
