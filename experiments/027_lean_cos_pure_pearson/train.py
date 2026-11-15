"""Training script for experiment 019: 004's grouped-encoder LSTM (unchanged
architecture) trained with a configurable loss/optimizer recipe.

Usage (baseline-equivalent recipe, for sanity):
    python train.py --train ... --valid ... --device cuda \\
        --optimizer adam --pearson-scope sequence --mse-weight 0.5 --select-metric loss

Stage 1 arms (loss only, Adam):
    --pearson-scope batch --mse-weight 0.5
    --pearson-scope batch --mse-weight 0.6
Stage 2 (SGD, trial LR picks via short runs, e.g. --epochs 4 --warmup-epochs 1):
    --optimizer sgd --lr 0.1 --momentum 0.9 --clip-norm 1.0 --pearson-scope batch ...

What differs from every earlier experiment's train.py (see README.md):
- `--pearson-scope batch` pools ALL scored rows in the batch into one
  correlation per target (closest affordable proxy for the corrected global
  metric); `sequence` reproduces the old per-sequence-then-average form.
- `--mse-weight w`: loss = w * weighted_MSE + (1-w) * (-weighted_Pearson).
- `--optimizer sgd` (Nesterov momentum by default) with `--clip-norm`.
  Gradient clipping is off by default (0) to keep Adam arms identical to
  earlier experiments; the pre-clip gradient norm is always logged.
- `--select-metric wp` (default): per-epoch corrected global WP on the valid
  set (eval_wp.py) drives best-checkpoint selection, ReduceLROnPlateau and
  early stopping, instead of valid_loss (which isn't comparable across loss
  blends). valid_loss is still logged.

Warmup + ReduceLROnPlateau + early-stopping recipe, batch size 4, 30 epochs,
dropout 0.1 and the data are otherwise unchanged from 004's 30-epoch run.
"""
from __future__ import annotations

import argparse
import os
import sys

_NUM_THREADS = os.environ.get("TRAIN_NUM_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _NUM_THREADS)

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.set_num_threads(int(_NUM_THREADS))

from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import LeanCosGroupedEncoderLSTMModel
from dataset import ConnectomeSequenceDataset, collate
from eval_wp import GlobalWPAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from losses import weighted_mse_loss, weighted_pearson_loss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    p.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True, help="SGD only")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-norm", type=float, default=0.0, help="max grad norm; 0 = no clipping")
    p.add_argument("--mse-weight", type=float, default=0.5, help="loss = w*MSE + (1-w)*(-Pearson)")
    p.add_argument("--pearson-scope", choices=["sequence", "batch"], default="sequence")
    p.add_argument("--select-metric", choices=["wp", "loss"], default="wp")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--warmup-start-factor", type=float, default=0.1)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=2)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--price-dim", type=int, default=24)
    p.add_argument("--volume-dim", type=int, default=24)
    p.add_argument("--trade-dim", type=int, default=12)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0, help="smoke tests only; 0 = all")
    p.add_argument("--max-valid-batches", type=int, default=0, help="smoke tests only; 0 = all")
    return p.parse_args()


def compute_loss(args, predictions, targets, need_prediction) -> torch.Tensor:
    mse = weighted_mse_loss(targets[need_prediction], predictions[need_prediction])
    if args.mse_weight >= 1.0:
        return mse
    if args.pearson_scope == "batch":
        # one pooled correlation per target over every scored row in the batch
        pearson = weighted_pearson_loss(targets[need_prediction].T, predictions[need_prediction].T)
    else:
        per_seq = []
        for b in range(predictions.shape[0]):
            seq_mask = need_prediction[b]
            if seq_mask.sum() < 2:
                continue
            per_seq.append(weighted_pearson_loss(targets[b, seq_mask].T, predictions[b, seq_mask].T))
        pearson = torch.stack(per_seq).mean()
    return args.mse_weight * mse + (1.0 - args.mse_weight) * pearson


def train_one_epoch(args, model, loader, optimizer, device, epoch: int):
    model.train()
    total_loss, total_gnorm, n = 0.0, 0.0, 0
    progress = tqdm(loader, desc=f"epoch {epoch} [train]", unit="batch", leave=False)
    for i, batch in enumerate(progress):
        if args.max_train_batches and i >= args.max_train_batches:
            break
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        need_prediction = batch["need_prediction"].to(device)
        optimizer.zero_grad()
        predictions, _ = model(features)
        loss = compute_loss(args, predictions, targets, need_prediction)
        loss.backward()
        max_norm = args.clip_norm if args.clip_norm > 0 else float("inf")
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()
        total_gnorm += float(gnorm)
        n += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n, 1), total_gnorm / max(n, 1)


@torch.no_grad()
def evaluate(args, model, loader, device, epoch: int):
    """One pass over valid: proxy loss (same recipe loss as training) AND
    corrected global WP, from the same forward passes."""
    model.eval()
    total_loss, n = 0.0, 0
    acc = GlobalWPAccumulator(device)
    psum = torch.zeros(2, dtype=torch.float64, device=device); psq = torch.zeros_like(psum); pn = 0
    progress = tqdm(loader, desc=f"epoch {epoch} [valid]", unit="batch", leave=False)
    for i, batch in enumerate(progress):
        if args.max_valid_batches and i >= args.max_valid_batches:
            break
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        need_prediction = batch["need_prediction"].to(device)
        scored = (batch["need_prediction"] & batch["is_scored"]).to(device)
        predictions, _ = model(features)
        loss = compute_loss(args, predictions, targets, need_prediction)
        total_loss += loss.item()
        n += 1
        acc.add(targets, predictions, scored)
        ps = predictions[scored].double()
        psum += ps.sum(0); psq += (ps * ps).sum(0); pn += ps.shape[0]
        progress.set_postfix(loss=f"{loss.item():.4f}")
    mean = psum / max(pn, 1)
    r = acc.result()
    r["pred_std"] = [float(x) for x in (psq / max(pn, 1) - mean * mean).clamp_min(0).sqrt()]
    return total_loss / max(n, 1), r


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_loader = DataLoader(ConnectomeSequenceDataset(args.train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(ConnectomeSequenceDataset(args.valid), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = LeanCosGroupedEncoderLSTMModel(
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        price_dim=args.price_dim, volume_dim=args.volume_dim, trade_dim=args.trade_dim,
        head_hidden=args.head_hidden, dropout=args.dropout,
    ).to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            nesterov=args.nesterov, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    maximize = args.select_metric == "wp"
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max" if maximize else "min",
        factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({
            "hidden_size": args.hidden_size, "num_layers": args.num_layers,
            "price_dim": args.price_dim, "volume_dim": args.volume_dim, "trade_dim": args.trade_dim,
            "head_hidden": args.head_hidden,
            "recipe": {k: v for k, v in vars(args).items() if k not in ("train", "valid")},
        }, f, indent=2)

    warmup_epochs = max(0, args.warmup_epochs)
    warmup_start_lr = args.lr * args.warmup_start_factor

    best = float("-inf") if maximize else float("inf")
    stale = 0
    epoch_progress = tqdm(range(args.epochs), desc="epochs", unit="epoch")
    for epoch in epoch_progress:
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warmup_start_lr + frac * (args.lr - warmup_start_lr)

        train_loss, gnorm = train_one_epoch(args, model, train_loader, optimizer, device, epoch)
        valid_loss, wp = evaluate(args, model, valid_loader, device, epoch)
        metric = wp["wp"] if maximize else valid_loss

        if epoch >= warmup_epochs:
            scheduler.step(metric)
        lr = optimizer.param_groups[0]["lr"]
        tag = " [warmup]" if epoch < warmup_epochs else ""
        epoch_progress.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}", wp=f"{wp['wp']:.4f}", lr=f"{lr:.2e}")
        # keep "epoch N: train_loss= valid_loss= lr=" prefix: shared/plot_loss_curves.py regex depends on it
        tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} lr={lr:.2e} "
                   f"valid_wp={wp['wp']:.6f} t0={wp['t0']:.6f} t1={wp['t1']:.6f} gnorm={gnorm:.3f} pred_std={wp['pred_std'][0]:.3f}/{wp['pred_std'][1]:.3f}{tag}")

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        improved = metric > best + 1e-6 if maximize else metric < best - 1e-6
        if improved:
            best, stale = metric, 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                tqdm.write(f"early stopping at epoch {epoch}: no improvement in {args.select_metric} for "
                           f"{args.early_stop_patience} epochs (best={best:.6f})")
                break

    print(f"done. best {args.select_metric}={best:.6f}. Next: export_onnx.py then solution.py --validation ... "
          f"for the real WP number (the batched valid_wp above approximates it; see eval_wp.py).")


if __name__ == "__main__":
    main()
