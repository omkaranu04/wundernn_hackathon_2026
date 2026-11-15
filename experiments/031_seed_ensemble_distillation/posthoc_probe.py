"""Post-hoc output probes on valid for ONE trained model (no retraining).

Each probe is a cheap transform of the model's own prediction stream that
could be baked into the ONNX graph with a few nodes if it helped:

  smooth a    causal EMA over time, p'_t = a*p_t + (1-a)*p'_{t-1}
              (targets are smooth: lag-1 autocorrelation ~0.96)
  sharpen b   p'_t = p_t + b*(p_t - p_{t-1})          (if predictions lag)
  power g     sign(p)*|p|^g                            (shape of the output)
  scale c     c*p                                      (only matters through the +-2 clip)
  cross l     p0' = p0 - l*p1, p1' = p1 - l*p0         (t0/t1 are anti-correlated)

Prints the corrected global WP of every variant. One-parameter scans on 3.5M
scored rows, so in-sample selection bias is negligible, but confirm any
winner on held-out halves before using it.

    python posthoc_probe.py --model CKPT_DIR:WEIGHTS --device cpu
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.signal import lfilter
from torch.utils.data import DataLoader

from dataset import ConnectomeSequenceDataset, collate
from eval_wp import GlobalWPAccumulator
from teacher_preds import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--valid", default="../../wnn_connectome_starterpack/datasets/valid.parquet")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, device)
    loader = DataLoader(ConnectomeSequenceDataset(args.valid), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    variants = {"raw": lambda p: p}
    for a in (0.9, 0.75, 0.5, 0.3):
        variants[f"smooth a={a}"] = lambda p, a=a: lfilter([a], [1.0, -(1.0 - a)], p, axis=1)
    for b in (0.5, 1.0):
        variants[f"sharpen b={b}"] = lambda p, b=b: p + b * np.diff(p, axis=1, prepend=p[:, :1])
    for g in (0.7, 0.85, 1.15, 1.3, 1.6):
        variants[f"power g={g}"] = lambda p, g=g: np.sign(p) * np.abs(p) ** g
    for c in (0.5, 1.5, 2.0, 3.0):
        variants[f"scale c={c}"] = lambda p, c=c: c * p
    for l in (-0.2, -0.1, 0.1, 0.2):
        variants[f"cross l={l}"] = lambda p, l=l: p - l * p[..., ::-1]
    acc = {k: GlobalWPAccumulator(torch.device("cpu")) for k in variants}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            res = model(batch["features"].to(device))
            p = (res[0] if isinstance(res, tuple) else res).float().cpu().numpy().astype(np.float64)
            mask = batch["need_prediction"] & batch["is_scored"]
            for k, f in variants.items():
                acc[k].add(batch["targets"], torch.from_numpy(np.ascontiguousarray(f(p))).float(), mask)
            if i % 100 == 0:
                print(f"batch {i}/{len(loader)}", flush=True)
    base = acc["raw"].result()["wp"]
    print(f"\n{'variant':16s}  WP        vs raw")
    for k in variants:
        r = acc[k].result()
        print(f"{k:16s}  {r['wp']:.6f}  {r['wp'] - base:+.6f}   (t0 {r['t0']:.6f}, t1 {r['t1']:.6f})")


if __name__ == "__main__":
    main()
