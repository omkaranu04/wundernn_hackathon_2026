"""Do the targets live on a tick lattice?  (check before building a tick-aware output head)

A rival's public notes claim t0/t1 take values on a lattice of two tick sizes
(~0.657 and ~0.469), with 22.6% exact zeros. This counts distinct values, exact
zeros and the most frequent values of each target on need_prediction rows of N
train sequences, and tests whether the values are (near-)integer multiples of a
small set of step sizes.

    python analysis/scripts/target_lattice_probe.py --n-seq 200
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))
from data import file_columns, read_sequence  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n-seq", type=int, default=200); a = ap.parse_args()
pf = pq.ParquetFile(ROOT / "wnn_connectome_starterpack/datasets/train.parquet"); hs, ht = file_columns(pf)
ys = []
for i in np.linspace(0, pf.num_row_groups - 1, a.n_seq).astype(int):
    s = read_sequence(pf, int(i), has_is_scored=hs, has_targets=ht)
    ys.append(s.targets[s.need_prediction])
Y = np.concatenate(ys).astype(np.float64)
print(f"# {len(Y):,} rows from {a.n_seq} train sequences\n")
for t in (0, 1):
    y = Y[:, t]
    vals, cnt = np.unique(np.round(y, 6), return_counts=True)
    w = np.abs(np.clip(y, -2, 2))
    print(f"## t{t}: {len(vals):,} distinct values (rounded 1e-6); exact zeros {np.mean(y == 0):.1%}; std {y.std():.3f}")
    top = np.argsort(-cnt)[:12]
    print("most frequent:", ", ".join(f"{vals[j]:+.6f} ({cnt[j] / len(y):.1%})" for j in top))
    nz = np.abs(y[y != 0])
    print(f"smallest |nonzero| values: {', '.join(f'{v:.6f}' for v in np.unique(np.round(nz, 6))[:8])}")
    for step in (0.656868, 0.468536):
        r = nz / step
        print(f"  step {step}: share of nonzero |y| within 1% of an integer multiple: {np.mean(np.abs(r - np.round(r)) < 0.01):.1%}")
    k = np.round(np.abs(y) / 0.656868)
    for m in range(0, 5):
        sel = k == m
        print(f"  |y| ~ {m} x 0.6569: rows {sel.mean():.1%}, share of metric weight sum|clip y| {w[sel].sum() / w.sum():.1%}")
    print()
print(f"corr(t0, t1) = {np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]:+.4f}; share of rows with t0 == t1 == 0: {np.mean((Y[:, 0] == 0) & (Y[:, 1] == 0)):.1%}")
