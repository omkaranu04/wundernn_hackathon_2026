"""Diagnostic (no training): is there recurrent-state drift over a 20,000-step
sequence that periodic state resets would fix?

Motivated by an idea from another competition's write-up (zeroing the
LSTM state every N steps helped there once the metric became globally
pooled). Two checks on an ALREADY-TRAINED model, using the validated
corrected-global-WP accumulator (019's eval_wp.py):

1. Per-position WP: pooled WP restricted to rows in each 2,000-step bin of
   the sequence, for the normal full-state forward pass. If later bins are
   clearly worse than earlier ones, that's suggestive (not conclusive -
   the data itself can change character within a sequence).
2. Inference-time reset: run the same model with the state zeroed every N
   steps (N in --reset-every), by folding each sequence into chunks of N
   steps with fresh zero state. The model was NOT trained for this, so a
   cold-start penalty at each chunk start is expected; any N that matches
   or beats the full-state baseline is a strong signal. Also reports WP
   by position-within-chunk for the smallest N, to show the cold-start
   cost directly.

Usage:
    python diag.py --model-dir ../004_grouped_encoder_lstm --class-name GroupedEncoderLSTMModel \\
        --valid ../../wnn_connectome_starterpack/datasets/valid.parquet --device cuda --out results_004.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "019_loss_optimizer_recipe"))
from dataset import ConnectomeSequenceDataset, collate  # noqa: E402
from eval_wp import GlobalWPAccumulator  # noqa: E402

BIN = 2000


def load_model(model_dir: Path, class_name: str, device):
    spec = importlib.util.spec_from_file_location("diag_model_mod", model_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.load(open(model_dir / "checkpoints" / "config.json"))
    keys = ("hidden_size", "num_layers", "price_dim", "volume_dim", "trade_dim", "head_hidden")
    model = getattr(mod, class_name)(**{k: cfg[k] for k in keys}, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_dir / "checkpoints" / "best.pt", map_location=device))
    return model.eval()


@torch.no_grad()
def forward_with_resets(model, features: torch.Tensor, every: int) -> torch.Tensor:
    """Zero the recurrent state every `every` steps by folding chunks into the batch dim."""
    B, T, F = features.shape
    if every <= 0 or every >= T:
        return model(features)[0]
    assert T % every == 0, f"T={T} not divisible by {every}"
    chunks = features.reshape(B * (T // every), every, F)
    preds = model(chunks)[0]
    return preds.reshape(B, T, -1)


def fresh_accs(n, device):
    return [GlobalWPAccumulator(device) for _ in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--class-name", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reset-every", type=int, nargs="+", default=[2000, 5000, 10000])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="smoke tests only")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.model_dir), args.class_name, device)
    loader = DataLoader(ConnectomeSequenceDataset(args.valid), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    variants = [0] + list(args.reset_every)  # 0 = full state, no reset
    T = 20000
    n_bins = T // BIN
    overall = {v: GlobalWPAccumulator(device) for v in variants}
    by_bin = {v: fresh_accs(n_bins, device) for v in variants}
    small = min(args.reset_every)
    chunk_edges = [0, 100, 300, 1000, small]
    chunk_edges = sorted(set(e for e in chunk_edges if e <= small))
    by_chunkpos = fresh_accs(len(chunk_edges) - 1, device)
    pos = torch.arange(T, device=device)

    for i, batch in enumerate(tqdm(loader, desc="valid", unit="batch")):
        if args.max_batches and i >= args.max_batches:
            break
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = (batch["need_prediction"] & batch["is_scored"]).to(device)
        for v in variants:
            preds = forward_with_resets(model, features, v)
            overall[v].add(targets, preds, mask)
            for b in range(n_bins):
                bm = mask & ((pos >= b * BIN) & (pos < (b + 1) * BIN)).unsqueeze(0)
                by_bin[v][b].add(targets, preds, bm)
            if v == small:
                within = pos % small
                for k in range(len(chunk_edges) - 1):
                    cm = mask & ((within >= chunk_edges[k]) & (within < chunk_edges[k + 1])).unsqueeze(0)
                    by_chunkpos[k].add(targets, preds, cm)

    res = {
        "overall": {("full_state" if v == 0 else f"reset_every_{v}"): overall[v].result() for v in variants},
        "by_position_bin_2000": {("full_state" if v == 0 else f"reset_every_{v}"): [a.result()["wp"] for a in by_bin[v]] for v in variants},
        f"by_position_within_chunk_reset_{small}": {
            f"[{chunk_edges[k]},{chunk_edges[k+1]})": by_chunkpos[k].result()["wp"] for k in range(len(chunk_edges) - 1)
        },
    }
    Path(args.out).write_text(json.dumps(res, indent=2))

    print("\nOVERALL pooled WP:")
    for k, r in res["overall"].items():
        print(f"  {k:>18}: wp={r['wp']:.6f}  t0={r['t0']:.6f}  t1={r['t1']:.6f}")
    print("\nWP by 2000-step position bin (columns = bins 0..9):")
    for k, row in res["by_position_bin_2000"].items():
        print(f"  {k:>18}: " + " ".join(f"{x:.4f}" for x in row))
    print(f"\nWP by position within chunk, reset_every_{small}:")
    for k, x in res[f"by_position_within_chunk_reset_{small}"].items():
        print(f"  {k:>12}: {x:.4f}")


if __name__ == "__main__":
    main()
