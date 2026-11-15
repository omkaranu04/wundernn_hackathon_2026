"""035 (2026-11-06): tick unit of every sequence (from its targets) + class balance of the 0.6569 unit.
Writes soft/units_<split>.npy (int8 per sequence: 0..3 = 0.1254 / 0.2413 / 0.4685 / 0.6569, -1 = no nonzero target).
    python tick_units.py ../../wnn_connectome_starterpack/datasets
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

UNITS = np.array([0.1254, 0.2413, 0.4685, 0.6569])


def unit_of(y):  # same rule as analysis/scripts/tick_unit_probe.py
    nz = np.abs(y[y != 0])
    if len(nz) == 0:
        return -1
    nz = nz[:5000]
    err = [np.mean(np.abs(nz / u - np.round(nz / u)) < 0.02) for u in UNITS]
    ok = [k for k in range(4) if err[k] > 0.97]
    return max(ok) if ok else int(np.argmax(err))


D = Path(sys.argv[1])
for split in ("train", "valid"):
    pf = pq.ParquetFile(D / f"{split}.parquet")
    cols = ["need_prediction", "t0", "t1"] + (["is_scored"] if split == "valid" else [])
    units = np.zeros(pf.num_row_groups, np.int8)
    rows = {k: [0, 0] for k in ("all", "scored")}; wsum = {k: [0.0, 0.0] for k in ("all", "scored")}
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=cols)
        npred = t.column("need_prediction").to_numpy(zero_copy_only=False).astype(bool)
        y = np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1)[npred]
        units[i] = unit_of(y.ravel())
        if units[i] == 3:
            masks = {"all": np.ones(len(y), bool)}
            if split == "valid":
                masks["scored"] = t.column("is_scored").to_numpy(zero_copy_only=False).astype(bool)[npred]
            for k, m in masks.items():
                yy = y[m]; tick = np.abs(yy) > 0.1
                rows[k][0] += tick.sum(); rows[k][1] += tick.size
                w = np.abs(np.clip(yy, -2, 2)); wsum[k][0] += w[tick].sum(); wsum[k][1] += w.sum()
    np.save(f"soft/units_{split}.npy", units)
    print(f"{split}: {pf.num_row_groups} sequences, unit shares " + ", ".join(f"{UNITS[k]}: {np.mean(units == k):.1%}" for k in range(4)) + f", none {np.mean(units < 0):.1%}")
    for k in rows:
        if rows[k][1]:
            print(f"  0.6569 unit, {k} rows: tick (|y|>0.1) share {rows[k][0] / rows[k][1]:.1%} of rows (flat {1 - rows[k][0] / rows[k][1]:.1%})")
