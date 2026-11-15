"""Training script for the encoder-MLP + wide single-layer LSTM + sinh-head model.

Usage:
    python train.py \\
        --train  ../../wnn_connectome_starterpack/datasets/train.parquet \\
        --valid  ../../wnn_connectome_starterpack/datasets/valid.parquet

UPDATED 2026-09-16: extended training (epochs default 5 -> 15) with
ReduceLROnPlateau LR decay + early stopping, now that GPU training makes
the extra epochs cheap (see EXPERIMENTS.md). Re-run in place in this same
directory per explicit user request - the original 5-epoch checkpoints
are preserved in `checkpoints_5epoch_backup/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from model import WideLSTMSinhModel
from dataset import ConnectomeSequenceDataset, collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from losses import weighted_mse_loss, weighted_pearson_loss  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="path to train.parquet")
    parser.add_argument("--valid", required=True, help="path to valid.parquet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="multiply LR by this when valid_loss plateaus")
    parser.add_argument("--lr-patience", type=int, default=2, help="epochs with no valid_loss improvement before reducing LR")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=5, help="epochs with no valid_loss improvement before stopping early")
    # Experiment 006 = a genuinely different lever from 002/003/004/005
    # (which all clustered within ~0.002 WP of each other regardless of
    # stacked LSTM depth): a small MLP feature encoder (112->64) feeding a
    # *single*, wide LSTM layer (hidden=224, vs. 2-4 layers at 128 before),
    # plus a `sinh`-activated output head so the network's raw output lives
    # in an arcsinh-matched space for the fat-tailed, signed targets (see
    # model.py's docstring and analysis/results/feature_statistics.md).
    parser.add_argument("--encoder-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=224)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clamp", type=float, default=8.0)
    parser.add_argument(
        "--loss", choices=["mse", "pearson", "mse+pearson"], default="mse+pearson",
        help="mse+pearson averages both; see shared/losses.py",
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

    model = WideLSTMSinhModel(
        encoder_dim=args.encoder_dim, hidden_size=args.hidden_size, num_layers=args.num_layers,
        head_dims=tuple(args.head_dims), dropout=args.dropout, clamp=args.clamp,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({
            "encoder_dim": args.encoder_dim, "hidden_size": args.hidden_size,
            "num_layers": args.num_layers, "head_dims": args.head_dims, "clamp": args.clamp,
        }, f, indent=2)

    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    epoch_progress = tqdm(range(args.epochs), desc="epochs", unit="epoch")
    for epoch in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, optimizer, args.loss, device, epoch)
        valid_loss = evaluate(model, valid_loader, args.loss, device, epoch)
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_progress.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}", lr=f"{current_lr:.2e}")
        tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} lr={current_lr:.2e}")

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
