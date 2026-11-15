"""Training script for the stacked LSTM + MLP head model.

Usage:
    python train.py \\
        --train  ../../wnn_connectome_starterpack/datasets/train.parquet \\
        --valid  ../../wnn_connectome_starterpack/datasets/valid.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import StackedLSTMMLPModel
from dataset import ConnectomeSequenceDataset, collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from losses import weighted_mse_loss, weighted_pearson_loss  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="path to train.parquet")
    parser.add_argument("--valid", required=True, help="path to valid.parquet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    # 2-layer/128-dim LSTM matches the baseline's GRU backbone exactly - the
    # MLP head (vs. experiment 002's plain Linear) is the only experimental
    # variable. Latency-benchmarked before training (see README.md): 0.0874
    # ms/step synthetic, 90.5% of budget even with NO platform speedup
    # assumed - much safer margin than experiment 002's 128.3%.
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
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

    model = StackedLSTMMLPModel(
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        head_dims=tuple(args.head_dims), dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({"hidden_size": args.hidden_size, "num_layers": args.num_layers, "head_dims": args.head_dims}, f, indent=2)

    best_valid_loss = float("inf")
    epoch_progress = tqdm(range(args.epochs), desc="epochs", unit="epoch")
    for epoch in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, optimizer, args.loss, device, epoch)
        valid_loss = evaluate(model, valid_loader, args.loss, device, epoch)
        epoch_progress.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}")
        tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    print(f"done. best valid_loss={best_valid_loss:.4f}. "
          f"Next: export_onnx.py then solution.py --validation ... for the real WP number.")


if __name__ == "__main__":
    main()
