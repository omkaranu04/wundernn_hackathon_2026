"""Log-scale companion to generate_plots.py: same feature-group grid
layout, same pre-computed full-dataset histograms
(../data/histogram.npy + bin_edges.npy), but each column's x-axis is
`log(m + x)` instead of raw `x`.

Per-feature domain-safe offset: some columns are negative or straddle
zero (see ../results/feature_statistics.md - frac_negative/frac_positive
are both nontrivial for nearly every column), so a shared or zero offset
would make `log` undefined for part of the range. Instead each column i
gets its own `m_i = -bin_edges[i, 0] + eps_i`, where `bin_edges[i, 0]` is
that column's own padded histogram lower bound (from compute_statistics.py
- already ~1.5x the observed min/max span) and `eps_i` is a small positive
margin scaled to the column's own range. This guarantees `m_i + x > 0`
for every bin in that column's histogram, chosen independently per column
("depending on the range of the feature", per request) rather than one
global constant.

Because `log(m + x)` is strictly monotonic wherever it's defined, this
only needs to remap each column's bin EDGES through the transform - the
counts already in each bin from the full-dataset scan are still exactly
right, they just live at new (non-uniform-width) x positions. No re-scan
of train.parquet is needed.
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

EPS_FRACTION = 1e-3  # margin as a fraction of the column's own (padded) range


def compute_offsets(bin_edges: np.ndarray) -> np.ndarray:
    """m_i = -bin_edges[i, 0] + eps_i, so log(m_i + x) is defined for every x
    in [bin_edges[i, 0], bin_edges[i, -1]] with a small positive margin at
    the low end (avoids log(~0) blowing up right at the edge)."""
    lo = bin_edges[:, 0]
    span = bin_edges[:, -1] - bin_edges[:, 0]
    eps = np.maximum(span * EPS_FRACTION, 1e-9)
    return -lo + eps


def bar_hist_log(ax, hist_counts: np.ndarray, log_edges: np.ndarray, color: str, log_y: bool = False) -> None:
    """Variable-width bars: log(m+x) transforms uniform raw bins into
    non-uniform log-space bins, so each bar needs its own width/position."""
    widths = np.diff(log_edges)
    ax.bar(log_edges[:-1], hist_counts, width=widths, align="edge", color=color, edgecolor="none")
    if log_y:
        ax.set_yscale("log")


def plot_group(histogram: np.ndarray, bin_edges: np.ndarray, offsets: np.ndarray, col_index: dict,
               group_name: str, columns: list[str]) -> None:
    n = len(columns)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, columns):
        i = col_index[col]
        log_edges = np.log(offsets[i] + bin_edges[i])
        bar_hist_log(ax, histogram[i], log_edges, "#4C72B0")
        ax.set_title(f"{col}  (m={offsets[i]:.3g})", fontsize=9)
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Feature group: {group_name} - log(m+x) scale, per-column m (full dataset, "
                 f"{histogram[col_index[columns[0]]].sum():,} rows)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / f"group_{group_name}_log.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_targets_log(histogram: np.ndarray, bin_edges: np.ndarray, offsets: np.ndarray, col_index: dict) -> None:
    t0_i, t1_i = col_index["t0"], col_index["t1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, i, name, color in [(axes[0], t0_i, "t0", "#DD8452"), (axes[1], t1_i, "t1", "#55A868")]:
        log_edges = np.log(offsets[i] + bin_edges[i])
        bar_hist_log(ax, histogram[i], log_edges, color, log_y=True)
        ax.set_title(f"{name} distribution - log(m+x) scale (m={offsets[i]:.3g}, count-y log scale)")
    fig.tight_layout()
    out_path = FIG_DIR / "targets_log_domain.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_offsets_csv(offsets: np.ndarray) -> None:
    out_path = RESULTS_DIR / "log_plot_offsets.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "m_offset", "note"])
        for name, m in zip(ALL_COLUMNS, offsets):
            writer.writerow([name, f"{m:.6g}", "log(m+x) chosen so m+x>0 across this column's full histogram range"])
    print(f"wrote {out_path}")


def main():
    histogram = np.load(DATA_DIR / "histogram.npy")
    bin_edges = np.load(DATA_DIR / "bin_edges.npy")
    col_index = {name: i for i, name in enumerate(ALL_COLUMNS)}
    offsets = compute_offsets(bin_edges)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for group_name, columns in GROUPS.items():
        plot_group(histogram, bin_edges, offsets, col_index, group_name, columns)
    plot_targets_log(histogram, bin_edges, offsets, col_index)
    write_offsets_csv(offsets)


if __name__ == "__main__":
    main()
