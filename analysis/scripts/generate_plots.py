"""Generate distribution plots from the FULL-dataset histograms produced by
compute_statistics.py (not a sample) - one grid per feature group,
plus target and correlation plots. Reads pre-computed histogram counts
(../data/histogram.npy + bin_edges.npy), not raw rows - matches
how compute_statistics.py itself stays memory-bounded.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from data import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

ALL_COLUMNS = FEATURE_COLUMNS + list(TARGET_COLUMNS)

GROUPS = {
    "i0_bid_price": [f"i0_p{i}" for i in range(0, 11)],
    "i0_ask_price": [f"i0_p{i}" for i in range(11, 22)],
    "i0_bid_volume": [f"i0_v{i}" for i in range(0, 11)],
    "i0_ask_volume": [f"i0_v{i}" for i in range(11, 22)],
    "i0_trade_price": [f"i0_dp{i}" for i in range(4)],
    "i0_trade_volume": [f"i0_dv{i}" for i in range(4)],
    "i1_bid_price": [f"i1_p{i}" for i in range(0, 11)],
    "i1_ask_price": [f"i1_p{i}" for i in range(11, 22)],
    "i1_bid_volume": [f"i1_v{i}" for i in range(0, 11)],
    "i1_ask_volume": [f"i1_v{i}" for i in range(11, 22)],
    "i1_trade_price": [f"i1_dp{i}" for i in range(4)],
    "i1_trade_volume": [f"i1_dv{i}" for i in range(4)],
    "additional": [f"a{i}" for i in range(8)],
}


def bar_hist(ax, hist_counts: np.ndarray, edges: np.ndarray, color: str, log_y: bool = False) -> None:
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax.bar(centers, hist_counts, width=width, color=color, edgecolor="none")
    if log_y:
        ax.set_yscale("log")


def plot_group(histogram: np.ndarray, bin_edges: np.ndarray, col_index: dict, group_name: str, columns: list[str]) -> None:
    n = len(columns)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, columns):
        i = col_index[col]
        bar_hist(ax, histogram[i], bin_edges[i], "#4C72B0")
        ax.set_title(col, fontsize=10)
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Feature group: {group_name} (full dataset, {histogram[col_index[columns[0]]].sum():,} rows)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / f"group_{group_name}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_targets(histogram: np.ndarray, bin_edges: np.ndarray, col_index: dict) -> None:
    t0_i, t1_i = col_index["t0"], col_index["t1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_hist(axes[0], histogram[t0_i], bin_edges[t0_i], "#DD8452", log_y=True)
    axes[0].set_title("t0 distribution (log scale - full dataset)")
    bar_hist(axes[1], histogram[t1_i], bin_edges[t1_i], "#55A868", log_y=True)
    axes[1].set_title("t1 distribution (log scale - full dataset)")
    fig.tight_layout()
    out_path = FIG_DIR / "targets.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_targets_raw(histogram: np.ndarray, bin_edges: np.ndarray, col_index: dict) -> None:
    """Same data as plot_targets(), but on a linear (unscaled) y-axis. Included
    on request for a direct, no-transformation view - note this makes the
    ~22%-of-rows exact-zero spike (see analysis/README.md finding 1) dwarf
    everything else visually, which is precisely why the log-scale version
    is the more informative one for seeing the tail shape. Both are kept:
    this one shows true relative frequency at a glance, the log one shows
    the tail structure the raw view hides."""
    t0_i, t1_i = col_index["t0"], col_index["t1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_hist(axes[0], histogram[t0_i], bin_edges[t0_i], "#DD8452", log_y=False)
    axes[0].set_title("t0 distribution (raw/linear scale - full dataset)")
    bar_hist(axes[1], histogram[t1_i], bin_edges[t1_i], "#55A868", log_y=False)
    axes[1].set_title("t1 distribution (raw/linear scale - full dataset)")
    fig.tight_layout()
    out_path = FIG_DIR / "targets_raw.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_correlations() -> None:
    with open(RESULTS_DIR / "feature_target_correlations.csv") as f:
        rows = list(csv.DictReader(f))
    corr_t0 = np.array([float(r["corr_t0"]) for r in rows])
    corr_t1 = np.array([float(r["corr_t1"]) for r in rows])
    names = [r["feature"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    x = np.arange(len(names))
    axes[0].bar(x, corr_t0, color="#DD8452")
    axes[0].set_ylabel("corr with t0")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[1].bar(x, corr_t1, color="#55A868")
    axes[1].set_ylabel("corr with t1")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(x[::4])
    axes[1].set_xticklabels([names[i] for i in x[::4]], rotation=90, fontsize=7)
    fig.suptitle("Per-feature Pearson correlation with each target (exact, full train.parquet)")
    fig.tight_layout()
    out_path = FIG_DIR / "feature_target_correlations.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    histogram = np.load(DATA_DIR / "histogram.npy")
    bin_edges = np.load(DATA_DIR / "bin_edges.npy")
    col_index = {name: i for i, name in enumerate(ALL_COLUMNS)}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for group_name, columns in GROUPS.items():
        plot_group(histogram, bin_edges, col_index, group_name, columns)
    plot_targets(histogram, bin_edges, col_index)
    plot_targets_raw(histogram, bin_edges, col_index)
    plot_correlations()


if __name__ == "__main__":
    main()
