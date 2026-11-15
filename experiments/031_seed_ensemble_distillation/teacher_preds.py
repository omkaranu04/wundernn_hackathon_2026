"""Teacher-ensemble predictions for experiment 031.

Runs several trained lean models over every sequence of a parquet file and

  --out PATH   writes their MEAN prediction per row, float16 (n_seq, 20000, 2)
               in row-group order (the `--soft-targets` file of train.py);
  --eval       (file must have `is_scored`, i.e. valid.parquet) prints the
               corrected global WP of every single model and of the mean of the
               first k models, k = 1..N - the ensemble's honest ceiling as long
               as no model was trained on that file.

Models are given as CHECKPOINT_DIR:WEIGHTS_FILE (config.json is read from the
directory, so different widths / auxiliary heads can be mixed). Predictions are
averaged RAW (before any clipping); the metric clips afterwards.

    python teacher_preds.py --parquet ../../wnn_connectome_starterpack/datasets/valid.parquet --eval \\
        --models ckpt_a:ema.pt ckpt_b:best_ema.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ConnectomeSequenceDataset, collate
from eval_wp import GlobalWPAccumulator
from model import LeanCosGroupedEncoderLSTMModel, build_model


def load_model(spec: str, device) -> LeanCosGroupedEncoderLSTMModel:
    ckpt_dir, weights = spec.rsplit(":", 1)
    with open(Path(ckpt_dir) / "config.json") as f:
        cfg = json.load(f)
    if cfg.get("bidirectional", False):
        raise SystemExit(f"{spec}: bidirectional (future-seeing) model cannot be a teacher here")
    model = build_model(
        cfg.get("arch", "lean"), hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], price_dim=cfg["price_dim"],
        volume_dim=cfg["volume_dim"], trade_dim=cfg["trade_dim"], head_hidden=cfg["head_hidden"],
        dropout=0.0, aux_scored=cfg.get("aux_scored", False),
    )
    model.load_state_dict(torch.load(Path(ckpt_dir) / weights, map_location="cpu"))
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--models", nargs="+", required=True, help="CHECKPOINT_DIR:WEIGHTS_FILE ...")
    p.add_argument("--out", default=None, help="write the mean prediction here (.npy, float16)")
    p.add_argument("--eval", action="store_true", help="per-model and cumulative-ensemble WP (needs is_scored)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--oof", action="store_true", help="out-of-fold soft targets (see CHECKPOINT:WEIGHTS@FOLD)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--max-batches", type=int, default=0)
    args = p.parse_args()
    if not args.out and not args.eval:
        p.error("nothing to do: give --out and/or --eval")

    device = torch.device(args.device)
    # CHECKPOINT:WEIGHTS@F marks an out-of-fold teacher that never saw sequences with index % n_folds == F;
    # with --oof, a sequence's soft target is the mean of ONLY the teachers whose fold it belongs to.
    folds = [int(s.rsplit("@", 1)[1]) if "@" in s else None for s in args.models]
    models = [load_model(s.rsplit("@", 1)[0], device) for s in args.models]
    if args.oof and any(f is None for f in folds):
        p.error("--oof needs every model as CHECKPOINT:WEIGHTS@FOLD")
    ds = ConnectomeSequenceDataset(args.parquet)
    if args.eval and not ds.has_is_scored:
        p.error("--eval needs a file with is_scored")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    n_seq = len(ds)

    out = None
    single = [GlobalWPAccumulator(device) for _ in models] if args.eval else None
    cumul = [GlobalWPAccumulator(device) for _ in models] if args.eval else None
    t0, row = time.time(), 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            features = batch["features"].to(device)
            if args.eval:
                targets = batch["targets"].to(device)
                scored = (batch["need_prediction"] & batch["is_scored"]).to(device)
            total = None
            for k, model in enumerate(models):
                res = model(features)
                pred = (res[0] if isinstance(res, tuple) else res).float()
                total = pred.clone() if total is None else total + pred
                if args.eval:
                    single[k].add(targets, pred, scored)
                    cumul[k].add(targets, total / (k + 1), scored)
            mean = total / len(models)
            if args.oof:
                idx = torch.arange(row, row + features.shape[0])
                acc = torch.zeros_like(mean); cnt = torch.zeros(features.shape[0], device=device)
                for k, model in enumerate(models):
                    sel = (idx % args.n_folds == folds[k]).to(device)
                    if sel.any():
                        res = model(features[sel]); pr = (res[0] if isinstance(res, tuple) else res).float()
                        acc[sel] += pr; cnt[sel] += 1
                assert (cnt > 0).all(), "a sequence has no out-of-fold teacher"
                mean = acc / cnt.view(-1, 1, 1)
            if args.out:
                if out is None:
                    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(n_seq, mean.shape[1], mean.shape[2]))
                out[row:row + mean.shape[0]] = mean.cpu().numpy().astype(np.float16)
            row += mean.shape[0]
            if i % 100 == 0:
                print(f"batch {i}/{len(loader)}  {row} sequences  {time.time() - t0:.0f}s", flush=True)
    if out is not None:
        out.flush()
        if row != n_seq and not args.max_batches:
            raise SystemExit(f"wrote {row} sequences, expected {n_seq}")
        print(f"wrote mean of {len(models)} models for {row} sequences to {args.out}")
    if args.eval:
        print("\nmodel                                                              single WP    mean of first k")
        for k, spec in enumerate(args.models):
            s, c = single[k].result(), cumul[k].result()
            print(f"{k + 1:2d} {spec[-62:]:62s}  {s['wp']:.6f}    k={k + 1:2d}: {c['wp']:.6f} (t0 {c['t0']:.6f}, t1 {c['t1']:.6f})")


if __name__ == "__main__":
    main()
