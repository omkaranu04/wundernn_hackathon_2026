"""Which FUTURE quantity are the targets?  (step 1 of the "auxiliary future targets" idea)

The bidirectional teacher of experiment 029 reached valid WP 0.997, so t0/t1 are
almost a deterministic function of future inputs. This probe looks for that
function in the simplest form: for every feature j and horizon k, the pooled
Pearson correlation (over need_prediction rows of N sequences) between the
target at time t and

    fwd(j, k) = x_j[t+k] - x_j[t]       (future change)

with the matching PAST change x_j[t] - x_j[t-k] as a reference for how much of
it is already visible. It also fits, per horizon k, a least-squares map from all
112 future changes to each target and reports its R^2 - if one horizon (or a
small range) explains most of the variance, that is the target's horizon, and
neighbouring horizons become natural auxiliary targets.

    python analysis/scripts/target_horizon_probe.py --n-seq 300 > analysis/results/target_horizon_probe.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))
from data import FEATURE_COLUMNS, file_columns, read_sequence  # noqa: E402

HORIZONS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400]


class Corr:
    """Pooled Pearson moments for many (x columns) x (2 targets)."""

    def __init__(self, nx):
        self.n = 0; self.sx = np.zeros(nx); self.sxx = np.zeros(nx); self.sy = np.zeros(2); self.syy = np.zeros(2); self.sxy = np.zeros((nx, 2))

    def add(self, x, y):
        x = x.astype(np.float64); y = y.astype(np.float64)
        self.n += len(x); self.sx += x.sum(0); self.sxx += (x * x).sum(0); self.sy += y.sum(0); self.syy += (y * y).sum(0); self.sxy += x.T @ y

    def corr(self):
        n = self.n; mx, my = self.sx / n, self.sy / n
        cov = self.sxy / n - np.outer(mx, my)
        vx = np.maximum(self.sxx / n - mx ** 2, 1e-12); vy = np.maximum(self.syy / n - my ** 2, 1e-12)
        return cov / np.sqrt(np.outer(vx, vy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "wnn_connectome_starterpack/datasets/train.parquet"))
    ap.add_argument("--n-seq", type=int, default=300)
    a = ap.parse_args()
    pf = pq.ParquetFile(a.train); hs, ht = file_columns(pf)
    idx = np.linspace(0, pf.num_row_groups - 1, a.n_seq).astype(int)
    fwd = {k: Corr(112) for k in HORIZONS}; past = {k: Corr(112) for k in HORIZONS}
    level = Corr(112)
    xtx = {k: np.zeros((113, 113)) for k in HORIZONS}; xty = {k: np.zeros((113, 2)) for k in HORIZONS}; yty = np.zeros(2); ysum = np.zeros(2); ny = {k: 0 for k in HORIZONS}
    yty_k = {k: np.zeros(2) for k in HORIZONS}; ysum_k = {k: np.zeros(2) for k in HORIZONS}
    for c, i in enumerate(idx):
        s = read_sequence(pf, int(i), has_is_scored=hs, has_targets=ht)
        X, Y, need = s.features.astype(np.float64), s.targets.astype(np.float64), s.need_prediction
        T = len(X)
        level.add(X[need], Y[need])
        for k in HORIZONS:
            t = np.arange(T - k); m = need[: T - k] & (t >= k)
            tt = t[m]
            d = X[tt + k] - X[tt]; y = Y[tt]
            fwd[k].add(d, y); past[k].add(X[tt] - X[tt - k], y)
            A = np.hstack([d, np.ones((len(d), 1))])
            xtx[k] += A.T @ A; xty[k] += A.T @ y; yty_k[k] += (y * y).sum(0); ysum_k[k] += y.sum(0); ny[k] += len(y)
        if (c + 1) % 50 == 0:
            print(f"# {c + 1}/{len(idx)} sequences", file=sys.stderr, flush=True)

    print(f"# target_horizon_probe: {len(idx)} train sequences, need_prediction rows\n")
    print("## Least-squares R^2 of each target from ALL 112 future changes x[t+k]-x[t]\n")
    print("| k | R^2 t0 | R^2 t1 | best single feature t0 (corr) | best single feature t1 (corr) |")
    print("|---|---|---|---|---|")
    for k in HORIZONS:
        w = np.linalg.solve(xtx[k] + 1e-6 * np.eye(113), xty[k])
        # residual sum of squares via normal equations: y'y - w'X'y
        rss = yty_k[k] - (w * xty[k]).sum(0)
        tss = yty_k[k] - ysum_k[k] ** 2 / ny[k]
        r2 = 1 - rss / tss
        cf = fwd[k].corr()
        b0, b1 = np.argmax(np.abs(cf[:, 0])), np.argmax(np.abs(cf[:, 1]))
        print(f"| {k} | {r2[0]:.3f} | {r2[1]:.3f} | {FEATURE_COLUMNS[b0]} ({cf[b0, 0]:+.3f}) | {FEATURE_COLUMNS[b1]} ({cf[b1, 1]:+.3f}) |")
    print("\n## Top-8 features by |corr(target, future change)| at the best horizon, with the past-change and level reference\n")
    for tgt in (0, 1):
        best_k = max(HORIZONS, key=lambda k: np.abs(fwd[k].corr()[:, tgt]).max())
        cf, cp, cl = fwd[best_k].corr()[:, tgt], past[best_k].corr()[:, tgt], level.corr()[:, tgt]
        print(f"### t{tgt}: best horizon k = {best_k}\n")
        print("| feature | corr future change | corr past change | corr level |")
        print("|---|---|---|---|")
        for j in np.argsort(-np.abs(cf))[:8]:
            print(f"| {FEATURE_COLUMNS[j]} | {cf[j]:+.3f} | {cp[j]:+.3f} | {cl[j]:+.3f} |")
        print()
    print("## Max |corr| over features of the FUTURE change, per horizon (t0 / t1)\n")
    print(" ".join(f"k={k}:{np.abs(fwd[k].corr()[:, 0]).max():.3f}/{np.abs(fwd[k].corr()[:, 1]).max():.3f}" for k in HORIZONS))


if __name__ == "__main__":
    main()
