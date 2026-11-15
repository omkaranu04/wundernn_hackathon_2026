"""Per-tick-unit output scale/offset on HONEST predictions (experiment 032, 2026-11-03).

Uses analysis/results/per_unit_preds.npz (per_unit_wp.py: S1_a100, a train-only student, on valid scored rows).
The loss is mostly scale-free per sequence, so predictions may be mis-scaled BETWEEN tick units, which matters for a
pooled correlation. Fits, per unit and target, a scale (and optionally an offset) maximising global WP on one half of
the valid sequences (even index) and scores the other half (odd), and vice versa - an honest 2-fold estimate.

    python analysis/scripts/per_unit_calibration.py > analysis/results/per_unit_calibration.txt
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/scripts"))
from per_unit_wp import wp  # noqa: E402
from tick_unit_probe import UNITS  # noqa: E402

z = np.load(ROOT / "analysis/results/per_unit_preds.npz")
Y, P, U, S = z["Y"].astype(np.float64), z["P"].astype(np.float64), z["U"], z["S"]


def apply(P, U, sc, off):
    Q = P.copy()
    for k in range(4):
        Q[U == k] = Q[U == k] * sc[k] + off[k]
    return Q


def fit(mask, use_offset):
    """Coordinate search over per-unit, per-target scale (and offset) maximising global WP on `mask` rows."""
    sc = np.ones((4, 2)); off = np.zeros((4, 2))
    grid_s = np.exp(np.linspace(-0.7, 0.7, 29)); grid_o = np.linspace(-0.15, 0.15, 31)
    y, p, u = Y[mask], P[mask], U[mask]
    best = wp(y, apply(p, u, sc, off), np.ones(len(y), bool))
    for _ in range(3):
        for k in range(1, 4):  # unit 0 fixed at scale 1 (correlation is scale invariant overall)
            for t in range(2):
                for g in grid_s:
                    s2 = sc.copy(); s2[k, t] = g
                    v = wp(y, apply(p, u, s2, off), np.ones(len(y), bool))
                    if v > best + 1e-7:
                        best, sc = v, s2
        if use_offset:
            for k in range(4):
                for t in range(2):
                    for g in grid_o:
                        o2 = off.copy(); o2[k, t] = g
                        v = wp(y, apply(p, u, sc, o2), np.ones(len(y), bool))
                        if v > best + 1e-7:
                            best, off = v, o2
    return sc, off, best


print(f"# per-unit calibration of S1_a100 (train-only) on valid scored rows ({len(Y):,})\n")
print(f"uncalibrated global WP {wp(Y, P, np.ones(len(Y), bool)):.6f}\n")
for use_offset in (False, True):
    gains = []
    for fold in (0, 1):
        tr = (S % 2) == fold; te = ~tr
        sc, off, fit_wp = fit(tr, use_offset)
        base = wp(Y[te], P[te], np.ones(te.sum(), bool)); cal = wp(Y[te], apply(P[te], U[te], sc, off), np.ones(te.sum(), bool))
        gains.append(cal - base)
        print(f"{'scale+offset' if use_offset else 'scale only'} fold {fold}: held-out WP {base:.6f} -> {cal:.6f} ({cal - base:+.6f}); "
              f"scales t0 {np.round(sc[:, 0], 3).tolist()} t1 {np.round(sc[:, 1], 3).tolist()}"
              + (f" offsets t0 {np.round(off[:, 0], 3).tolist()} t1 {np.round(off[:, 1], 3).tolist()}" if use_offset else ""))
    print(f"  mean held-out gain: {np.mean(gains):+.6f}\n")
