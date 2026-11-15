"""Can a sequence's tick unit be read from its INPUTS early enough to route on it?  (experiment 032, 2026-11-03)

A rival's notes: four tick units, fixed within a sequence (0.1254, 0.2413, 0.4685, 0.6569); the largest carries
47.7% of the metric weight but is predicted far worse (corr 0.44 vs 0.70-0.72); per-unit routing gave them +0.0007.
Routing needs the unit at inference time, i.e. from inputs only. This probe
  1. labels every sequence's unit from its targets (the smallest common step of its nonzero |t0|, |t1|),
  2. builds per-sequence statistics of the first N input rows (mean, std, mean |diff| and the share of zero diffs of
     each of the 112 features) for N in --prefixes, and
  3. fits a classifier on train sequences and reports held-out accuracy on valid sequences per N, plus the
     metric-weight share of each unit.

    python analysis/scripts/tick_unit_probe.py --n-train 2000 > analysis/results/tick_unit_probe.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))
from data import file_columns, read_sequence  # noqa: E402

UNITS = np.array([0.1254, 0.2413, 0.4685, 0.6569])


def unit_of(y):
    """Index of the tick unit whose integer multiples best explain the sequence's nonzero |targets|."""
    nz = np.abs(y[y != 0])
    if len(nz) == 0:
        return -1
    nz = nz[:5000]
    err = [np.mean(np.abs(nz / u - np.round(nz / u)) < 0.02) for u in UNITS]
    # prefer the LARGEST unit that explains (almost) everything: multiples of 0.6569 are not multiples of 0.1254
    ok = [k for k in range(4) if err[k] > 0.97]
    return max(ok) if ok else int(np.argmax(err))


def feats(X, n):
    x = X[:n].astype(np.float64); d = np.diff(x, axis=0)
    return np.concatenate([x.mean(0), x.std(0), np.abs(d).mean(0), (d == 0).mean(0)])


def load(path, idx, prefixes):
    pf = pq.ParquetFile(path); hs, ht = file_columns(pf)
    lab, F, wmass = [], {n: [] for n in prefixes}, []
    for c, i in enumerate(idx):
        s = read_sequence(pf, int(i), has_is_scored=hs, has_targets=ht)
        y = s.targets[s.need_prediction]
        lab.append(unit_of(y.ravel()))
        wmass.append(np.abs(np.clip(y, -2, 2)).sum())
        for n in prefixes:
            F[n].append(feats(s.features, n))
        if (c + 1) % 250 == 0:
            print(f"# {path.name}: {c + 1}/{len(idx)}", file=sys.stderr, flush=True)
    return np.array(lab), {n: np.array(v) for n, v in F.items()}, np.array(wmass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--prefixes", type=int, nargs="+", default=[100, 300, 1000, 3000])
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    ds = ROOT / "wnn_connectome_starterpack/datasets"
    ntr = pq.ParquetFile(ds / "train.parquet").num_row_groups; nva = pq.ParquetFile(ds / "valid.parquet").num_row_groups
    ytr, Ftr, _ = load(ds / "train.parquet", np.linspace(0, ntr - 1, a.n_train).astype(int), a.prefixes)
    yva, Fva, wva = load(ds / "valid.parquet", np.arange(nva), a.prefixes)
    print(f"# tick-unit probe: {len(ytr)} train sequences, {len(yva)} valid sequences (all)\n")
    for name, y in (("train", ytr), ("valid", yva)):
        print(f"{name} unit shares: " + ", ".join(f"{UNITS[k]}: {np.mean(y == k):.1%}" for k in range(4)) + f", unlabelled {np.mean(y < 0):.1%}")
    print("valid metric-weight share (sum |clip y|): " + ", ".join(f"{UNITS[k]}: {wva[yva == k].sum() / wva.sum():.1%}" for k in range(4)))
    print("\n| first N rows | held-out accuracy (valid) | majority baseline |")
    print("|---|---|---|")
    m = (ytr >= 0); mv = (yva >= 0)
    base = np.mean(yva[mv] == np.bincount(ytr[m]).argmax())
    for n in a.prefixes:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1).fit(Ftr[n][m], ytr[m])
        acc = np.mean(clf.predict(Fva[n][mv]) == yva[mv])
        print(f"| {n} | {acc:.3f} | {base:.3f} |")


if __name__ == "__main__":
    main()
