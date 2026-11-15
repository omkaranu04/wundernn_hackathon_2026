"""STAGE 2a: freeze the trained backbone (checkpoints/best.pt) and extract
its per-step hidden states for a SUBSAMPLE of sequences, to use as the
input features for the two XGBoost regressors trained next
(train_xgboost.py).

Why a subsample, not the full 212M-row train.parquet: storing hidden
states for every row of the full dataset would need ~108GB (212.14M rows
x 128-dim hidden x 4 bytes) - impractical. A few hundred sequences still
gives XGBoost a training set far larger than typical tabular-competition
scale (e.g. the Quant Challenge 2025 Rank-3 LightGBM solution we reviewed
trained on just 80,000 rows total; --train-sequences 300 alone gives
~6M rows) while keeping storage and extraction time reasonable.

Each sampled sequence is still run through its FULL 20,000 steps (via one
batched nn.LSTM call over the whole sequence, mathematically identical to
step-by-step - the same equivalence every experiment's export_onnx.py
verify() step already confirms) so the causal hidden state is correct;
only the ~19,901 need_prediction rows' (hidden_state, t0, t1) triples are
actually saved - warm-up rows contribute to the LSTM's state but aren't
useful training rows for the heads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from data import file_columns, read_sequence  # noqa: E402

from model import BackboneModel

import pyarrow.parquet as pq


def extract(parquet_path: str, model: BackboneModel, device: torch.device, n_sequences: int, seed: int):
    parquet = pq.ParquetFile(parquet_path)
    has_is_scored, has_targets = file_columns(parquet)
    if not has_targets:
        raise ValueError(f"{parquet_path} has no target columns")

    rng = np.random.default_rng(seed)
    n_total = parquet.num_row_groups
    indices = rng.choice(n_total, size=min(n_sequences, n_total), replace=False)
    indices.sort()  # sequential row-group reads are faster than random-order

    hidden_chunks = []
    target_chunks = []

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            seq = read_sequence(parquet, int(idx), has_is_scored=has_is_scored, has_targets=has_targets)
            features = torch.from_numpy(seq.features).unsqueeze(0).to(device)  # (1, 20000, 112)
            lstm_out, _ = model.encode_and_recur(features)  # (1, 20000, hidden_size)
            lstm_out = lstm_out.squeeze(0).cpu().numpy()  # (20000, hidden_size)

            mask = seq.need_prediction
            hidden_chunks.append(lstm_out[mask])
            target_chunks.append(seq.targets[mask])

            if (i + 1) % 50 == 0 or (i + 1) == len(indices):
                print(f"  {i + 1}/{len(indices)} sequences processed")

    X = np.concatenate(hidden_chunks, axis=0).astype(np.float32)
    y = np.concatenate(target_chunks, axis=0).astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="path to train.parquet")
    parser.add_argument("--valid", required=True, help="path to valid.parquet")
    parser.add_argument("--train-sequences", type=int, default=300)
    parser.add_argument("--valid-sequences", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="embeddings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    model = BackboneModel(
        hidden_size=config["hidden_size"], num_layers=config["num_layers"],
        price_dim=config["price_dim"], volume_dim=config["volume_dim"], trade_dim=config["trade_dim"],
        head_hidden=config["head_hidden"], dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device))
    model.eval()
    print(f"loaded frozen backbone from {checkpoint_dir / 'best.pt'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"extracting embeddings from {args.train_sequences} train sequences...")
    X_train, y_train = extract(args.train, model, device, args.train_sequences, args.seed)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    print(f"  X_train {X_train.shape}, y_train {y_train.shape}, "
          f"{X_train.nbytes / 1e9:.2f} GB")

    print(f"extracting embeddings from {args.valid_sequences} valid sequences...")
    X_valid, y_valid = extract(args.valid, model, device, args.valid_sequences, args.seed + 1)
    np.save(out_dir / "X_valid.npy", X_valid)
    np.save(out_dir / "y_valid.npy", y_valid)
    print(f"  X_valid {X_valid.shape}, y_valid {y_valid.shape}, "
          f"{X_valid.nbytes / 1e9:.2f} GB")

    print(f"done. Next: train_xgboost.py --embeddings-dir {out_dir}")


if __name__ == "__main__":
    main()
