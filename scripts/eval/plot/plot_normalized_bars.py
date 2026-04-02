#!/usr/bin/env python3
"""
Plot compressor evaluation figures (normalized to Zstd per dataset).

Input: multiple per-dataset CSV files, each with columns:
  Compressor,Ratio,Compression_Speed_MBps,Decompression_Speed_MBps

Example:
  python scripts/eval/plot/plot_normalized_bars.py --csv_glob "csv/*.csv" --outdir figs_norm

Outputs (PDF):
  figs_norm/fig_ratio_norm.pdf
  figs_norm/fig_comp_speed_norm.pdf
  figs_norm/fig_decomp_speed_norm.pdf
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOR_MAP = {
    "Gzip":       "#fde0dd",  # Gray
    "Zstd":       "#bdbdbd",  # Blue
    "GBZ":        "#fc9272",  # Orange
    "sqz":        "#9ecae1",  # Purple
    "sqz+bgzip":  "#3182bd",  # Brown
    "sqz+Zstd":   "#1c9099",  # Red
    "gfaz(CPU)":  "#bcbddc",  # Pink
    "gfaz(GPU)":  "#addd8e",  # Green
}



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
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "figure.dpi": 300,
    })


# -------------------------
# Data loading
# -------------------------
def read_one_csv(path: str) -> pd.DataFrame:
    # utf-8-sig handles Excel-exported CSVs with BOM
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"Compressor", "Ratio", "Compression_Speed_MBps", "Decompression_Speed_MBps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["Compressor"] = df["Compressor"].astype(str).str.strip()
    df["Dataset"] = Path(path).stem  # <-- critical
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


# -------------------------
# Normalization (robust, merge-based)
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


def draw_panel(
    ax,
    df_long: pd.DataFrame,
    value_col: str,
    ylabel: str,
    dataset_order: list,
    compressor_order: list,
    logy=False,
    baseline_line=True,
):
    pivot = df_long.pivot_table(index="Dataset", columns="Compressor", values=value_col, aggfunc="first")
    
    # Reindex
    pivot = pivot.reindex(dataset_order)
    pivot = pivot.reindex(columns=compressor_order)

    datasets = list(pivot.index)
    compressors = list(pivot.columns)

    n_groups = len(datasets)
    n_bars = len(compressors)
    x = np.arange(n_groups)

    total_width = 0.86
    bar_w = total_width / max(n_bars, 1)

    def hatch_for(name: str):
        # turned off hatching
        return None

    for i, comp in enumerate(compressors):
        vals = pivot[comp].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        xpos = x - total_width / 2 + (i + 0.5) * bar_w

        color = COLOR_MAP.get(comp, "#CCCCCC")

        ax.bar(
            xpos[mask],
            vals[mask],
            width=bar_w,
            label=comp,
            color=color,
            hatch=hatch_for(comp),
            edgecolor="black",
            linewidth=0.3,
        )

        # Mark missing entries
        for j, ok in enumerate(mask):
            if not ok:
                ax.text(
                    xpos[j], 0.02, "NA",
                    ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(),
                    fontsize=6, rotation=90,
                )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    
    # Grid and lines
    ax.yaxis.grid(True, linewidth=0.3)
    ax.set_axisbelow(True)

    for i in range(len(datasets) - 1):
        ax.axvline(
            x=i + 0.5,
            linewidth=1.0,
            color="black",
            linestyle="--",
            alpha=0.8,
            zorder=0,
        )

    if baseline_line:
        ax.axhline(1.0, linewidth=1.0, color="red", linestyle="--", alpha=0.7)

    if logy:
        ax.set_yscale("log", base=2)
        
    return datasets  # return labels for use in the main driver


def plot_combined_vertical(df, outpath, zstd_name, dataset_order, log_speed):
    # Determine compressor order once
    unique_comps = df["Compressor"].unique()
    compressor_order = default_order(unique_comps)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.0), sharex=True)
    
    # 1. Ratio
    draw_panel(
        axes[0], df, "Ratio_norm", 
        f"Ratio / {zstd_name}", 
        dataset_order, compressor_order, logy=False
    )
    
    # 2. Comp Speed
    draw_panel(
        axes[1], df, "Compression_Speed_norm", 
        f"Comp. / {zstd_name}", 
        dataset_order, compressor_order, logy=log_speed
    )
    
    # 3. Decomp Speed
    labels = draw_panel(
        axes[2], df, "Decompression_Speed_norm", 
        f"Decomp. / {zstd_name}", 
        dataset_order, compressor_order, logy=log_speed
    )
    
    # Set x-labels only on the bottom plot
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    
    # Global Legend (using handles from the first plot)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(compressor_order),
        handlelength=0.7,
        handletextpad=0.3,
        columnspacing=0.5,
        frameon=False
    )

    # Adjust layout to make room for legend at top
    # tight_layout rect parameter: (left, bottom, right, top)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot normalized compressor evaluation (normalized to Zstd per dataset).")
    parser.add_argument("--csv_glob", type=str, default="csv/*.csv",
                        help="Glob for per-dataset CSV files (default: csv/*.csv)")
    parser.add_argument("--outdir", type=str, default="figs_norm",
                        help="Output directory (default: figs_norm)")
    parser.add_argument("--zstd_name", type=str, default="Zstd",
                        help="Exact compressor name used for baseline row (default: Zstd)")
    parser.add_argument("--log_speed", action="store_true",
                        help="Use log scale for speed plots (helpful if methods differ by orders of magnitude)")
    args = parser.parse_args()

    set_paper_style()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_all(args.csv_glob)
    # quick sanity print (comment out later)
    # print("Loaded columns:", df.columns.tolist())
    # print(df.head(2))

    df = normalize_to_zstd(df, zstd_name=args.zstd_name)

    dataset_order = ["chr1", "chr6", "Ecoli", "HPRCv11", "HPRCv20", "HPRCv21"]

    outpath = os.path.join(args.outdir, "fig_eval_norm_combined.pdf")
    plot_combined_vertical(
        df,
        outpath=outpath,
        zstd_name=args.zstd_name,
        dataset_order=dataset_order,
        log_speed=args.log_speed
    )

    print(f"Saved combined normalized figure to: {outpath}")


if __name__ == "__main__":
    main()
