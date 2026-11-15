"""Training script for the LSTM + causal windowed attention model.

Originally trained (2026-09-12) at a scaled-down config (d_model=24,
lstm_layers=1, attention_window=32, ff_multiplier=2) chosen because the
full-size design (d_model=128, lstm_layers=2, attention_window=512,
ff_multiplier=4) measured ~7.8x over the latency budget in PyTorch eager
mode. Those original checkpoints are preserved in
`checkpoints_original_backup/`.

UPDATED 2026-09-19: retrained at that full, originally-intended design
with the same warmup + `ReduceLROnPlateau` decay + early-stopping recipe
used across this project's other 30-epoch runs, per explicit user
instruction to build/train the full-size version despite its known
latency risk. Result: 233.5% of budget uncorrected, confirmed via TWO
platform timeouts (pre- and post- the organizers' runtime-discrepancy fix,
2026-09-19 and 2026-09-23 respectively, within 2s of each other) - a
genuine architectural overage, not a platform-variability artifact. Those
checkpoints are preserved in `checkpoints_dmodel128_fullscale_backup/`.

UPDATED 2026-09-23: scaled back down to d_model=32, lstm_layers=1,
attention_window=32, ff_multiplier=2 (see argparse defaults below) - the
original safe architecture almost unchanged (only d_model bumped 24->32),
now finally given the modern 30-epoch warmup+decay+early-stop recipe the
original 2026-09-12 training never used. Latency should land close to the
original's ~88.5% of budget uncorrected - verified via pre-training
benchmark before committing to the full run, per this project's standard
practice (see EXPERIMENTS.md for the number).

Usage:
    python train.py \\
        --train  ../../wnn_connectome_starterpack/datasets/train.parquet \\
        --valid  ../../wnn_connectome_starterpack/datasets/valid.parquet \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import LSTMAttentionModel
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
                         help="linear LR warmup from --warmup-start-factor*--lr up to --lr over this many epochs, "
                              "before ReduceLROnPlateau decay/early-stopping logic starts tracking")
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="multiply LR by this when valid_loss plateaus")
    parser.add_argument("--lr-patience", type=int, default=2, help="epochs with no valid_loss improvement before reducing LR")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=5, help="epochs with no valid_loss improvement before stopping early")
    # UPDATED 2026-09-23: defaults changed to d_model=32, lstm_layers=1,
    # attention_window=32, ff_multiplier=2 - the original safe/submitted
    # config (d_model=24, everything else identical) bumped only in
    # d_model, 24->32. NOT the d_model=128/lstm_layers=2/window=512
    # "full-scale" config this file defaulted to 2026-09-19 through
    # 2026-09-22 - that config confirmed timed out on the platform TWICE
    # (233.5% of budget uncorrected; checkpoints preserved in
    # `checkpoints_dmodel128_fullscale_backup/`). This config's whole
    # point is to stay close to the original's confirmed-safe latency
    # (~88.5% of budget) while finally giving it the modern 30-epoch
    # warmup+decay+early-stop recipe - verify the actual latency via
    # export_onnx.py + a benchmark on random-init weights before trusting
    # this comment and committing to a full training run.
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--attention-window", type=int, default=32)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--loss", choices=["mse", "pearson", "mse+pearson"], default="mse+pearson",
        help="mse+pearson averages both; consider warming up with mse alone "
             "for the first few epochs since the pearson loss is unstable "
             "when a batch's weighted variance is near zero (see shared/losses.py)",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="fixed seed - reproducibility is required for prize verification")
    return parser.parse_args()


def compute_loss(loss_name: str, predictions: torch.Tensor, targets: torch.Tensor, need_prediction: torch.Tensor) -> torch.Tensor:
    """predictions/targets: (batch, T, 2); need_prediction: (batch, T) bool.

    Only rows with need_prediction=True contribute - matches the competition's
    scoring domain (warm-up rows have no target signal worth optimizing for
    directly, even though the model still runs on them to build up state).
    """
    losses = []

    if "mse" in loss_name:
        pred_sel = predictions[need_prediction]   # (n_selected, 2) - flattening batch/seq is fine for elementwise MSE
        target_sel = targets[need_prediction]
        losses.append(weighted_mse_loss(target_sel, pred_sel))

    if "pearson" in loss_name:
        # Correlation must be computed per-sequence, not over rows pooled
        # across different sequences (a correlation "across sequences" isn't
        # the quantity being scored) - see METRIC.md's per-block definition.
        per_seq_losses = []
        for b in range(predictions.shape[0]):
            seq_mask = need_prediction[b]
            if seq_mask.sum() < 2:
                continue
            # weighted_pearson_loss correlates over the last dim; transpose
            # so that dim is "selected steps of this one sequence" and the
            # leading dim is "target index" (t0, t1) - both get averaged.
            per_seq_losses.append(
                weighted_pearson_loss(targets[b, seq_mask].T, predictions[b, seq_mask].T)
            )
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

    model = LSTMAttentionModel(
        d_model=args.d_model,
        lstm_layers=args.lstm_layers,
        n_heads=args.n_heads,
        attention_window=args.attention_window,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(
            {
                "d_model": args.d_model,
                "lstm_layers": args.lstm_layers,
                "n_heads": args.n_heads,
                "attention_window": args.attention_window,
                "ff_multiplier": args.ff_multiplier,
            },
            f,
            indent=2,
        )

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
          f"Next: score checkpoints/best.pt locally with utils.ScorerStepByStep "
          f"(via solution.py --validation ...) to get the actual WP number - "
          f"the loss values above are a training proxy, not the competition metric.")


if __name__ == "__main__":
    main()
