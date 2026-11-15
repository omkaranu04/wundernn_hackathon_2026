"""Training script for the sum-reparameterized grouped-encoder LSTM model
(head predicts (t0, t0+t1) instead of (t0, t1) directly - see model.py).

Usage:
    python train.py \\
        --train  ../../wnn_connectome_starterpack/datasets/train.parquet \\
        --valid  ../../wnn_connectome_starterpack/datasets/valid.parquet \\
        --device cuda

Same training recipe as experiment 004's current (2026-09-17) setup: 30
epochs, linear LR warmup, ReduceLROnPlateau decay, early stopping - the
only thing under test here is the target reparameterization, isolated
against the same architecture/recipe as our best model.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

torch.set_num_threads(int(_NUM_THREADS))

from model import SumReparamGroupedEncoderLSTMModel
from dataset import ConnectomeSequenceDataset, collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from losses import weighted_mse_loss, weighted_pearson_loss  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="path to train.parquet")
    parser.add_argument("--valid", required=True, help="path to valid.parquet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3,
                         help="linear LR warmup from --warmup-start-factor*--lr up to --lr over this many epochs")
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--price-dim", type=int, default=16)
    parser.add_argument("--volume-dim", type=int, default=16)
    parser.add_argument("--trade-dim", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--loss", choices=["mse", "pearson", "mse+pearson"], default="mse+pearson",
        help="mse+pearson averages both; see shared/losses.py. Computed on the "
             "reconstructed raw [t0,t1] predictions, unchanged from every other experiment.",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def compute_loss(loss_name: str, predictions: torch.Tensor, targets: torch.Tensor, need_prediction: torch.Tensor) -> torch.Tensor:
    losses = []
    if "mse" in loss_name:
        losses.append(weighted_mse_loss(targets[need_prediction], predictions[need_prediction]))
    if "pearson" in loss_name:
        per_seq_losses = []
        for b in range(predictions.shape[0]):
            seq_mask = need_prediction[b]
            if seq_mask.sum() < 2:
                continue
            per_seq_losses.append(weighted_pearson_loss(targets[b, seq_mask].T, predictions[b, seq_mask].T))
        if per_seq_losses:
            losses.append(torch.stack(per_seq_losses).mean())
    if not losses:
        raise ValueError(f"no loss terms produced for loss_name={loss_name!r}")
    return sum(losses) / len(losses)


def train_one_epoch(model, loader, optimizer, loss_name, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"epoch {epoch} [train]", unit="batch", leave=False)
    for batch in progress:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        need_prediction = batch["need_prediction"].to(device)
        optimizer.zero_grad()
        predictions, _ = model(features)
        loss = compute_loss(loss_name, predictions, targets, need_prediction)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_name, device, epoch: int) -> float:
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"epoch {epoch} [valid]", unit="batch", leave=False)
    for batch in progress:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        need_prediction = batch["need_prediction"].to(device)
        predictions, _ = model(features)
        loss = compute_loss(loss_name, predictions, targets, need_prediction)
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_ds = ConnectomeSequenceDataset(args.train)
    valid_ds = ConnectomeSequenceDataset(args.valid)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SumReparamGroupedEncoderLSTMModel(
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        price_dim=args.price_dim, volume_dim=args.volume_dim, trade_dim=args.trade_dim,
        head_hidden=args.head_hidden, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({
            "hidden_size": args.hidden_size, "num_layers": args.num_layers,
            "price_dim": args.price_dim, "volume_dim": args.volume_dim, "trade_dim": args.trade_dim,
            "head_hidden": args.head_hidden,
        }, f, indent=2)

    warmup_epochs = max(0, args.warmup_epochs)
    warmup_start_lr = args.lr * args.warmup_start_factor

    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    epoch_progress = tqdm(range(args.epochs), desc="epochs", unit="epoch")
    for epoch in epoch_progress:
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            warmup_lr = warmup_start_lr + frac * (args.lr - warmup_start_lr)
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr

        train_loss = train_one_epoch(model, train_loader, optimizer, args.loss, device, epoch)
        valid_loss = evaluate(model, valid_loader, args.loss, device, epoch)

        if epoch >= warmup_epochs:
            scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        warmup_tag = " [warmup]" if epoch < warmup_epochs else ""
        epoch_progress.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}", lr=f"{current_lr:.2e}")
        tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} lr={current_lr:.2e}{warmup_tag}")

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if valid_loss < best_valid_loss - 1e-6:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                tqdm.write(f"early stopping at epoch {epoch}: no improvement in valid_loss for "
                           f"{args.early_stop_patience} epochs (best={best_valid_loss:.4f})")
                break

    print(f"done. best valid_loss={best_valid_loss:.4f}. "
          f"Next: export_onnx.py then solution.py --validation ... for the real WP number.")


if __name__ == "__main__":
    main()
