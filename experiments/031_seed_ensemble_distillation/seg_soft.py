"""037 (2026-11-06, adversarial-ideation idea 2): SEGMENT-END labels from the causal teacher pool.

Fact (checked on valid: 39 exceptions in 744k scored rows): targets change only on rows where at least one i0 price
column (i0_p0..p21) changes. So y(t) = y(te) for te = min(last row before the next i0 price change, t + K), and the
causal pool's prediction READ AT te is a look-ahead label that never sees the price move settling the target.
Honest check (train-only T10 teachers, valid fold 0), 0.6569 unit: row t 0.454, seg12 0.512, seg25 0.548 (LA12 0.551).

Bias fix (discriminator): te is not a stopping time, so on the 0.6569 unit the label's |y|-weighted mean drifts away
from y's (-0.760 -> -0.785 at K=12). Subtract each unit's |y|-weighted mean drift (seg - TV12 at t), per target,
fitted on TRAIN, from the segment-end label of that unit (train and valid alike).

Label written (discriminator's safer 3-way mix, keeps the causal anchor):
    0.5 TV12 + 0.25 seg_K(corrected) + 0.25 LA12  =  0.25 TV12 + 0.25 seg_K(corrected) + 0.5 TV12_LA12

    python seg_soft.py K OUT_TAG     ->  soft/{train,valid}_<OUT_TAG>.npy  (+ soft/seg_drift_<OUT_TAG>.json)
"""
import json
import sys

import numpy as np
import pyarrow.parquet as pq

D = "../../wnn_connectome_starterpack/datasets"
K = int(sys.argv[1]); TAG = sys.argv[2]
PCOLS = [f"i0_p{j}" for j in range(22)]


def seg_index(P, K):
    """te per row: min(next i0-price-change row - 1, t + K, last row)."""
    n = len(P)
    chg = np.zeros(n, bool); chg[1:] = (P[1:] != P[:-1]).any(1)
    idx = np.flatnonzero(chg)
    t = np.arange(n)
    pos = np.searchsorted(idx, t, side="right")
    nxt = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)], n)
    return np.minimum(np.minimum(nxt - 1, t + K), n - 1)


def read(pf, i, cols):
    t = pf.read_row_group(i, columns=cols)
    return {c: t.column(c).to_numpy(zero_copy_only=False) for c in cols}


# pass 1 (train): per-unit |y|-weighted mean drift of seg vs TV12(t), per target
pf = pq.ParquetFile(f"{D}/train.parquet")
TV = np.load("soft/train_TV12.npy", mmap_mode="r"); U = np.load("soft/units_train.npy")
S = np.zeros((4, 2)); W = np.zeros((4, 2)); MY = np.zeros((4, 2)); MT = np.zeros((4, 2))
for i in range(pf.num_row_groups):
    c = read(pf, i, PCOLS + ["need_prediction", "t0", "t1"])
    P = np.stack([c[k] for k in PCOLS], 1)
    te = seg_index(P, K); tv = TV[i].astype(np.float32)
    m = c["need_prediction"].astype(bool)
    y = np.stack([c["t0"], c["t1"]], 1)[m]; w = np.abs(np.clip(y, -2, 2))
    u = U[i]
    S[u] += (w * (tv[te][m] - tv[m])).sum(0); W[u] += w.sum(0); MY[u] += (w * y).sum(0); MT[u] += (w * tv[m]).sum(0)
    if (i + 1) % 2000 == 0:
        print(f"pass 1: {i + 1}/{pf.num_row_groups}", flush=True)
drift = S / W
info = {"K": K, "drift_seg_minus_tv12_per_unit": drift.round(5).tolist(), "wmean_y": (MY / W).round(4).tolist(),
        "wmean_tv12": (MT / W).round(4).tolist(), "units": [0.1254, 0.2413, 0.4685, 0.6569]}
print(json.dumps(info), flush=True)
json.dump(info, open(f"soft/seg_drift_{TAG}.json", "w"), indent=1)

# pass 2: write the corrected 3-way label for train and valid
for split in ("train", "valid"):
    pf = pq.ParquetFile(f"{D}/{split}.parquet")
    TV = np.load(f"soft/{split}_TV12.npy", mmap_mode="r"); TL = np.load(f"soft/{split}_TV12_LA12.npy", mmap_mode="r")
    U = np.load(f"soft/units_{split}.npy")
    O = np.lib.format.open_memmap(f"soft/{split}_{TAG}.npy", mode="w+", dtype=np.float16, shape=TV.shape)
    for i in range(pf.num_row_groups):
        P = np.stack([read(pf, i, PCOLS)[k] for k in PCOLS], 1)
        te = seg_index(P, K); tv = TV[i].astype(np.float32)
        seg = tv[te] - drift[U[i]]
        O[i] = (0.25 * tv + 0.25 * seg + 0.5 * TL[i].astype(np.float32)).astype(np.float16)
    O.flush()
    print(f"wrote soft/{split}_{TAG}.npy {TV.shape}", flush=True)
