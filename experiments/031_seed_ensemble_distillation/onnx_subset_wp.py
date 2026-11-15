"""Step-by-step ONNX scoring of several graphs on a SUBSET of valid sequences.

For checking what a graph change (int8 quantisation, merged ensemble, ...)
does to accuracy without the 75-minute full scorer: every model is run row by
row with its recurrent state, exactly like solution.py, on every `--every`-th
valid sequence; prints each model's corrected global WP on those sequences,
the max/mean absolute difference of its predictions to the FIRST model, and
its speed.

    python onnx_subset_wp.py --every 15 fp32.onnx int8.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from data import file_columns, read_sequence  # noqa: E402

CLIP = 2.0


def session(path):
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


class Moments:
    def __init__(self):
        self.s = np.zeros((6, 2))

    def add(self, y, p):
        y = np.clip(y, -CLIP, CLIP).astype(np.float64); p = np.clip(p, -CLIP, CLIP).astype(np.float64); w = np.abs(y)
        self.s += np.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0), (w * y * y).sum(0), (w * p * p).sum(0), (w * y * p).sum(0)])

    def wp(self):
        w, wy, wp, wyy, wpp, wyp = self.s
        ym, pm = wy / w, wp / w
        c = (wyp / w - ym * pm) / np.sqrt((wyy / w - ym ** 2) * (wpp / w - pm ** 2))
        return float(c.mean()), float(c[0]), float(c[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--valid", default=str(Path(__file__).resolve().parent.parent.parent / "wnn_connectome_starterpack/datasets/valid.parquet"))
    ap.add_argument("--every", type=int, default=15)
    args = ap.parse_args()

    pf = pq.ParquetFile(args.valid)
    has_scored, has_targets = file_columns(pf)
    sess = [session(m) for m in args.models]
    mom = [Moments() for _ in sess]
    secs = [0.0] * len(sess); dmax = [0.0] * len(sess); dsum = [0.0] * len(sess); rows = 0
    idxs = list(range(0, pf.num_row_groups, args.every))
    for n, idx in enumerate(idxs):
        seq = read_sequence(pf, idx, has_is_scored=has_scored, has_targets=has_targets)
        mask = seq.need_prediction & seq.is_scored
        ref = None
        for k, s in enumerate(sess):
            ins = s.get_inputs()
            state = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], dtype=np.float32) for i in ins if i.name != "features"}
            outs = [o.name for o in s.get_outputs()]
            pred = np.zeros((len(seq.features), 2), dtype=np.float32)
            t = time.perf_counter()
            for r in range(len(seq.features)):
                res = dict(zip(outs, s.run(outs, {"features": seq.features[r], **state})))
                pred[r] = res["prediction"].reshape(-1)[:2]
                state = {name: res["next_" + name] for name in state}
            secs[k] += time.perf_counter() - t
            mom[k].add(seq.targets[mask], pred[mask])
            if ref is None:
                ref = pred
            else:
                d = np.abs(pred - ref)[seq.need_prediction]
                dmax[k] = max(dmax[k], float(d.max())); dsum[k] += float(d.sum())
        rows += int(seq.need_prediction.sum()) * 2
        if n % 10 == 0:
            print(f"{n + 1}/{len(idxs)} sequences", flush=True)
    steps = len(idxs) * 20000
    print(f"\n{len(idxs)} valid sequences (every {args.every}th)")
    for k, m in enumerate(args.models):
        wp, t0, t1 = mom[k].wp()
        print(f"{m[-70:]:70s} WP {wp:.6f} (t0 {t0:.6f}, t1 {t1:.6f})  {secs[k] / steps * 1e6:.1f} us/step"
              + ("" if k == 0 else f"  |diff to first| max {dmax[k]:.4f} mean {dsum[k] / rows:.5f}"))


if __name__ == "__main__":
    main()
