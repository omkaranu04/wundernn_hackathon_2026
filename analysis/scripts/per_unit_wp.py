"""Per-tick-unit weighted Pearson of an HONEST model on valid (experiment 032, 2026-11-03).

tick_unit_probe.py showed the unit is readable from the first 100 input rows (99.9%). This asks whether a model
trained without valid (S1_a100, train-only teachers) is much worse on some units - i.e. whether routing / per-unit
specialists have room. Reports, per unit: share of metric weight, WP inside the unit, and the global WP if that
unit's predictions were perfect (an upper bound on what fixing that unit could add).

    python analysis/scripts/per_unit_wp.py --model experiments/031_seed_ensemble_distillation/results/checkpoints_S1_a100:best_ema.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared")); sys.path.insert(0, str(ROOT / "experiments/031_seed_ensemble_distillation"))
sys.path.insert(0, str(ROOT / "analysis/scripts"))
from data import file_columns, read_sequence  # noqa: E402
from teacher_preds import load_model  # noqa: E402
from tick_unit_probe import UNITS, unit_of  # noqa: E402


def wp(y, p, w_mask):
    """Global weighted Pearson averaged over the two targets, clip +-2, weights |clip y| on masked rows."""
    out = []
    for t in range(2):
        yy = np.clip(y[w_mask, t], -2, 2); pp = np.clip(p[w_mask, t], -2, 2); w = np.abs(yy)
        my, mp = (w * yy).sum() / w.sum(), (w * pp).sum() / w.sum()
        out.append((w * (yy - my) * (pp - mp)).sum() / np.sqrt((w * (yy - my) ** 2).sum() * (w * (pp - mp) ** 2).sum()))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device); m = load_model(a.model, dev)
    pf = pq.ParquetFile(ROOT / "wnn_connectome_starterpack/datasets/valid.parquet"); hs, ht = file_columns(pf)
    Y, P, U = [], [], []
    with torch.no_grad():
        for i in range(pf.num_row_groups):
            s = read_sequence(pf, i, has_is_scored=hs, has_targets=ht)
            pred = m(torch.from_numpy(s.features)[None].to(dev))[0][0].float().cpu().numpy()
            sel = s.need_prediction & s.is_scored
            Y.append(s.targets[sel]); P.append(pred[sel]); U.append(np.full(sel.sum(), unit_of(s.targets[s.need_prediction].ravel())))
            if (i + 1) % 300 == 0:
                print(f"# {i + 1}/{pf.num_row_groups}", file=sys.stderr, flush=True)
    S = np.concatenate([np.full(len(u), i) for i, u in enumerate(U)])  # sequence index per row
    Y, P, U = np.concatenate(Y).astype(np.float64), np.concatenate(P).astype(np.float64), np.concatenate(U)
    np.savez_compressed(ROOT / "analysis/results/per_unit_preds.npz", Y=Y.astype(np.float32), P=P.astype(np.float32), U=U.astype(np.int8), S=S.astype(np.int32))
    allm = np.ones(len(Y), bool); g = wp(Y, P, allm)
    w = np.abs(np.clip(Y, -2, 2)).sum(1)
    print(f"# per-unit WP of {a.model} on valid scored rows ({len(Y):,} rows)\n\nglobal WP {g:.6f}\n")
    print("| unit | weight share | WP inside unit | global WP if this unit were perfect |")
    print("|---|---|---|---|")
    for k in range(4):
        mk = U == k
        Pk = P.copy(); Pk[mk] = Y[mk]
        print(f"| {UNITS[k]} | {w[mk].sum() / w.sum():.1%} | {wp(Y, P, mk):.4f} | {wp(Y, Pk, allm):.4f} |")


if __name__ == "__main__":
    main()
