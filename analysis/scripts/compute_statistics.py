"""Exact, complete-dataset statistics over the FULL train.parquet (all
10,607 sequences, ~212M rows) - one streaming pass, not a sample.

Why this is safe to run on a 95GB file without OOMing or needing multiple
full reads (the earlier sampled version's whole rationale was to avoid
exactly this cost - this version does the full dataset properly instead):

1. **Exact scalar statistics via raw-moment accumulation.** For each of the
   114 columns we accumulate running sums of x, x^2, x^3, x^4 (float64),
   plus exact min/max and exact zero/negative/positive counts, one
   row-group (sequence) at a time. This is O(1) memory per column
   regardless of how many rows we've seen - never holds the full dataset,
   only 114 running scalars (x9 accumulators). Mean/variance/skewness/
   excess-kurtosis are then derived from these sums with standard raw-
   moment formulas (e.g. Var = E[x^2] - E[x]^2) - exact, no approximation,
   no second pass needed for these.

2. **Exact Pearson correlation via the same trick, bivariate.** Feature-vs-
   target correlation needs sum(x*y) in addition to each side's sum/sumsq -
   still O(1) extra memory per (feature, target) pair, accumulated
   alongside everything else in the same single pass.

3. **Histograms (for percentiles + plots) via pre-set bins with an exact
   overflow/underflow safety net.** A true single-pass histogram needs bin
   edges decided before seeing all the data. We bootstrap generous bin
   ranges per column from this script's OWN prior output (`seed_histogram_ranges`
   reads `results/feature_statistics.csv` if it already exists, padding its
   min/max), so a rerun self-corrects using the last run's exact extremes -
   no separate sampling step needed. A hardcoded generous fallback is used
   only if no prior output exists at all (e.g. a completely fresh checkout).
   Any value falling outside the assumed range is counted exactly
   (`hist_underflow`/`hist_overflow` per column) rather than silently
   clipped or dropped - on the first-ever run here, this caught 665/1,288
   extreme values for t0/t1 respectively (0.0003-0.0006% of 212.14M rows -
   far too few to shift any reported percentile, but real, and reported
   rather than hidden). Histogram updates are buffered across many
   sequences before each `np.histogram` call (see `FLUSH_EVERY`) to keep
   this from becoming 114 x 10,607 tiny individual histogram calls.

Net result: ONE read of the ~95GB file, bounded memory throughout, and
every scalar statistic reported (mean/std/skew/kurtosis/min/max/exact
zero-negative-positive fractions/correlations) is exact over the complete
dataset - only histogram-derived percentiles and plots carry the small,
explicitly-measured approximation from finite bin width and the rare
out-of-seeded-range values noted above.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from data import FEATURE_COLUMNS, TARGET_COLUMNS, file_columns, read_sequence  # noqa: E402

TRAIN_PATH = Path(__file__).resolve().parent.parent.parent / "wnn_connectome_starterpack" / "datasets" / "train.parquet"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRIOR_STATS = RESULTS_DIR / "feature_statistics.csv"  # this script's own prior output, if any - used only to seed histogram ranges

ALL_COLUMNS = FEATURE_COLUMNS + list(TARGET_COLUMNS)
N_COLS = len(ALL_COLUMNS)
N_BINS = 500
RANGE_PADDING = 0.5   # fraction of observed range to pad on each side
FLUSH_EVERY = 50      # sequences buffered before each histogram update batch

# Only used if PRIOR_STATS doesn't exist yet (a completely fresh checkout with
# no prior run to bootstrap from). Deliberately wide: targets are known to
# have extreme outliers (observed full-dataset range: t0 in [-125.6, 74.5],
# t1 in [-179.9, 77.3]), while the 112 raw features cluster much tighter
# (most within roughly [-6, 6], including the ±5.1993 preprocessing clip
# found on most columns) - see analysis/README.md.
FALLBACK_RANGE = {"t0": (-250.0, 150.0), "t1": (-250.0, 150.0), "_default": (-10.0, 10.0)}


def seed_histogram_ranges() -> tuple[np.ndarray, np.ndarray]:
    """Returns (lo, hi) per column: from this script's own prior output if it
    exists (self-correcting on rerun - a rerun uses the exact extremes the
    LAST run measured, not a guess), else a generous hardcoded fallback."""
    lo = np.empty(N_COLS)
    hi = np.empty(N_COLS)
    if PRIOR_STATS.exists():
        with open(PRIOR_STATS) as f:
            by_name = {row["column"]: row for row in csv.DictReader(f)}
        for i, name in enumerate(ALL_COLUMNS):
            col_min, col_max = float(by_name[name]["min"]), float(by_name[name]["max"])
            span = col_max - col_min
            pad = span * RANGE_PADDING if span > 0 else 1.0
            lo[i], hi[i] = col_min - pad, col_max + pad
        print(f"seeded histogram ranges from prior run: {PRIOR_STATS}")
    else:
        for i, name in enumerate(ALL_COLUMNS):
            lo[i], hi[i] = FALLBACK_RANGE.get(name, FALLBACK_RANGE["_default"])
        print(f"no prior output found - seeded histogram ranges from hardcoded fallback (first-ever run)")
    return lo, hi


def main():
    lo, hi = seed_histogram_ranges()
    bin_edges = np.stack([np.linspace(lo[i], hi[i], N_BINS + 1) for i in range(N_COLS)])  # (N_COLS, N_BINS+1)

    # exact accumulators (float64 throughout to avoid precision loss over 212M summed terms)
    count = 0
    sum1 = np.zeros(N_COLS)
    sum2 = np.zeros(N_COLS)
    sum3 = np.zeros(N_COLS)
    sum4 = np.zeros(N_COLS)
    col_min = np.full(N_COLS, np.inf)
    col_max = np.full(N_COLS, -np.inf)
    n_zero = np.zeros(N_COLS, dtype=np.int64)
    n_neg = np.zeros(N_COLS, dtype=np.int64)
    n_pos = np.zeros(N_COLS, dtype=np.int64)
    histogram = np.zeros((N_COLS, N_BINS), dtype=np.int64)
    n_underflow = np.zeros(N_COLS, dtype=np.int64)
    n_overflow = np.zeros(N_COLS, dtype=np.int64)

    n_features = len(FEATURE_COLUMNS)
    n_targets = len(TARGET_COLUMNS)
    # bivariate sums for feature-target correlation: sum(feature_i * target_j)
    sum_xy = np.zeros((n_features, n_targets))
    sum_t0t1 = 0.0  # for corr(t0, t1)

    parquet_file = pq.ParquetFile(TRAIN_PATH)
    n_sequences = parquet_file.num_row_groups
    has_is_scored, has_targets = file_columns(parquet_file)
    assert has_targets

    buffer = []

    def flush_histogram(buffered_rows: np.ndarray) -> None:
        for i in range(N_COLS):
            col_hist, _ = np.histogram(buffered_rows[:, i], bins=bin_edges[i])
            histogram[i] += col_hist
            n_underflow[i] += int(np.sum(buffered_rows[:, i] < bin_edges[i, 0]))
            n_overflow[i] += int(np.sum(buffered_rows[:, i] > bin_edges[i, -1]))

    start_time = time.time()
    for idx in tqdm(range(n_sequences), desc="full pass over train.parquet", unit="seq"):
        seq = read_sequence(parquet_file, idx, has_is_scored=has_is_scored, has_targets=has_targets)
        data = np.concatenate([seq.features, seq.targets], axis=1).astype(np.float64)  # (20000, 114)

        count += data.shape[0]
        sum1 += data.sum(axis=0)
        sum2 += (data**2).sum(axis=0)
        sum3 += (data**3).sum(axis=0)
        sum4 += (data**4).sum(axis=0)
        col_min = np.minimum(col_min, data.min(axis=0))
        col_max = np.maximum(col_max, data.max(axis=0))
        n_zero += (data == 0).sum(axis=0)
        n_neg += (data < 0).sum(axis=0)
        n_pos += (data > 0).sum(axis=0)

        seq_targets_f64 = seq.targets.astype(np.float64)
        sum_xy += seq.features.astype(np.float64).T @ seq_targets_f64
        sum_t0t1 += float((seq_targets_f64[:, 0] * seq_targets_f64[:, 1]).sum())

        buffer.append(data)
        if len(buffer) >= FLUSH_EVERY:
            flush_histogram(np.concatenate(buffer, axis=0))
            buffer = []

    if buffer:
        flush_histogram(np.concatenate(buffer, axis=0))

    elapsed = time.time() - start_time
    print(f"\nfull pass over {n_sequences:,} sequences ({count:,} rows) took {elapsed/60:.1f} min")

    # ---- derive exact scalar stats from raw moments ----
    mean = sum1 / count
    var = sum2 / count - mean**2
    std = np.sqrt(np.maximum(var, 0))
    # E[(x-mu)^3] = mu3' - 3*mu*mu2' + 2*mu^3   (raw moments mu_k' = sum_k/count)
    mu2p, mu3p, mu4p = sum2 / count, sum3 / count, sum4 / count
    m3 = mu3p - 3 * mean * mu2p + 2 * mean**3
    m4 = mu4p - 4 * mean * mu3p + 6 * mean**2 * mu2p - 3 * mean**4
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = m3 / std**3
        excess_kurtosis = m4 / std**4 - 3

    rows = []
    for i, name in enumerate(ALL_COLUMNS):
        rows.append({
            "column": name,
            "kind": "target" if name in TARGET_COLUMNS else "feature",
            "count": count,
            "mean": mean[i], "std": std[i], "min": col_min[i], "max": col_max[i],
            "skew": skew[i], "excess_kurtosis": excess_kurtosis[i],
            "frac_zero": n_zero[i] / count, "frac_negative": n_neg[i] / count, "frac_positive": n_pos[i] / count,
            "hist_underflow": int(n_underflow[i]), "hist_overflow": int(n_overflow[i]),
        })

    # percentiles from the exact full-dataset histogram (interpolated CDF)
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for i, row in enumerate(rows):
        cum = np.cumsum(histogram[i]).astype(np.float64)
        cum = cum / cum[-1] * 100 if cum[-1] > 0 else cum
        centers = (bin_edges[i, :-1] + bin_edges[i, 1:]) / 2
        for p in percentiles:
            row[f"p{p}"] = float(np.interp(p, cum, centers))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["column", "kind", "count", "mean", "std", "min", "max",
                  *[f"p{p}" for p in percentiles], "skew", "excess_kurtosis",
                  "frac_zero", "frac_negative", "frac_positive", "hist_underflow", "hist_overflow"]
    with open(RESULTS_DIR / "feature_statistics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULTS_DIR / 'feature_statistics.csv'} ({len(rows)} columns, exact over full dataset)")

    total_overflow = sum(r["hist_underflow"] + r["hist_overflow"] for r in rows)
    if total_overflow > 0:
        affected = [(r["column"], r["hist_underflow"], r["hist_overflow"]) for r in rows if r["hist_underflow"] + r["hist_overflow"] > 0]
        print(f"NOTE: {total_overflow} values fell outside the seeded histogram range across {len(affected)} column(s) "
              f"(percentiles for these still used all in-range data; affected columns: {affected[:20]}{'...' if len(affected) > 20 else ''})")
    else:
        print("all values fell within the seeded histogram ranges - percentiles are fully accurate to bin width")

    # exact feature-target correlations from accumulated sums
    corr_rows = []
    for i, fname in enumerate(FEATURE_COLUMNS):
        row = {"feature": fname}
        for j, tname in enumerate(TARGET_COLUMNS):
            n = count
            sx, sxx = sum1[i], sum2[i]
            sy, syy = sum1[n_features + j], sum2[n_features + j]
            sxy = sum_xy[i, j]
            numerator = n * sxy - sx * sy
            denom = np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))
            row[f"corr_{tname}"] = float(numerator / denom) if denom > 0 else float("nan")
        corr_rows.append(row)
    with open(RESULTS_DIR / "feature_target_correlations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "corr_t0", "corr_t1"])
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"wrote {RESULTS_DIR / 'feature_target_correlations.csv'}")

    t0_idx, t1_idx = n_features, n_features + 1
    n = count
    sx, sxx = sum1[t0_idx], sum2[t0_idx]
    sy, syy = sum1[t1_idx], sum2[t1_idx]
    numerator = n * sum_t0t1 - sx * sy
    denom = np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))
    t0_t1_corr = float(numerator / denom) if denom > 0 else float("nan")
    print(f"corr(t0, t1) = {t0_t1_corr:.4f} (exact, full dataset)")
    with open(RESULTS_DIR / "target_correlation.txt", "w") as f:
        f.write(f"corr(t0, t1) = {t0_t1_corr:.6f}\ncomputed exactly over all {count:,} rows of train.parquet\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "histogram.npy", histogram)
    np.save(DATA_DIR / "bin_edges.npy", bin_edges)
    with open(DATA_DIR / "histogram_columns.txt", "w") as f:
        f.write("\n".join(ALL_COLUMNS))
    print(f"wrote full-dataset histograms to {DATA_DIR} for plotting")


if __name__ == "__main__":
    main()
