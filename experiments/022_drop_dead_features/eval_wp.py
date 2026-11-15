"""Batched GPU implementation of the CORRECTED ("Global") weighted Pearson
metric, for per-epoch model selection during training.

The organizer scorer (`ScorerStepByStep`) feeds rows one at a time through
ONNX and takes ~80 minutes per model. This computes the same quantity from
batched `model.forward()` passes in minutes, accumulating pooled weighted
moments across every sequence - the exact aggregation of the corrected
metric (PROBLEM_STATEMENT.md section 3): per target, over all rows where
`need_prediction AND is_scored`, clip target and prediction to [-2, 2], weight
w = |clipped target|, one set of pooled sums, then the mean of the two
targets' correlations.

Differences from the real scorer that can't be fully removed: this runs the
PyTorch model (cuDNN LSTM, float32) rather than the exported ONNX graph, so
tiny numeric differences are expected. Validate before trusting it: run
this file as a script against a checkpoint whose real scorer WP is known
(004's 30-epoch checkpoint scored 0.682425, t0=0.677896, t1=0.686955):

    python eval_wp.py --checkpoint ../004_grouped_encoder_lstm/checkpoints/best.pt \\
        --config ../004_grouped_encoder_lstm/checkpoints/config.json \\
        --valid ../../wnn_connectome_starterpack/datasets/valid.parquet --device cuda
"""
from __future__ import annotations

import argparse
import json

import torch

METRIC_CLIP = 2.0


class GlobalWPAccumulator:
    """Pooled weighted moments per target, accumulated in float64."""

    def __init__(self, device):
        z = lambda: torch.zeros(2, dtype=torch.float64, device=device)
        self.w, self.wy, self.wp, self.wyy, self.wpp, self.wyp = z(), z(), z(), z(), z(), z()

    @torch.no_grad()
    def add(self, targets: torch.Tensor, predictions: torch.Tensor, mask: torch.Tensor) -> None:
        """targets/predictions: (B, T, 2); mask: (B, T) bool (need_prediction & is_scored)."""
        y = targets[mask].clamp(-METRIC_CLIP, METRIC_CLIP).double()   # (N, 2)
        p = predictions[mask].clamp(-METRIC_CLIP, METRIC_CLIP).double()
        w = y.abs()
        self.w += w.sum(0)
        self.wy += (w * y).sum(0)
        self.wp += (w * p).sum(0)
        self.wyy += (w * y * y).sum(0)
        self.wpp += (w * p * p).sum(0)
        self.wyp += (w * y * p).sum(0)

    def result(self) -> dict:
        w = self.w.clamp_min(1e-12)
        ym, pm = self.wy / w, self.wp / w
        cov = self.wyp / w - ym * pm
        vy = (self.wyy / w - ym * ym).clamp_min(1e-12)
        vp = (self.wpp / w - pm * pm).clamp_min(1e-12)
        corr = (cov / (vy * vp).sqrt()).clamp(-1.0, 1.0)
        return {"t0": corr[0].item(), "t1": corr[1].item(), "wp": corr.mean().item()}


@torch.no_grad()
def evaluate_wp(model, loader, device, max_batches: int = 0) -> dict:
    model.eval()
    acc = GlobalWPAccumulator(device)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = (batch["need_prediction"] & batch["is_scored"]).to(device)
        predictions, _ = model(features)
        acc.add(targets, predictions, mask)
    return acc.result()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from dataset import ConnectomeSequenceDataset, collate
    from model import GroupedEncoderLSTMModel

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    device = torch.device(args.device)
    model = GroupedEncoderLSTMModel(
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        price_dim=cfg["price_dim"], volume_dim=cfg["volume_dim"],
        trade_dim=cfg["trade_dim"], head_hidden=cfg["head_hidden"], dropout=0.0,
        keep_additional=cfg.get("keep_additional", list(range(8))),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    loader = DataLoader(ConnectomeSequenceDataset(args.valid), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(evaluate_wp(model, tqdm(loader, desc="valid", unit="batch"), device))
