"""2026-11-14: multi-head variant of FullFeatureModel (idea: KlishevDA, 11th place in Wunder's earlier RNN challenge -
4 output heads on one recurrent trunk, averaged, +0.003 R2). K heads 128->64->2, each trained with its OWN loss
(train_mh.py averages the per-head losses), predictions = mean of the heads. The mean of K two-layer heads is exactly
ONE head with K*64 hidden units (first layers stacked, second layers concatenated / K): `to_single()` writes that
standard checkpoint, so packing / the numba engines / teacher_preds need no change."""
import json
from pathlib import Path

import torch
import torch.nn as nn

from model import FullFeatureModel


class FullFeatureMultiHead(FullFeatureModel):
    def __init__(self, n_heads: int = 4, head_hidden: int = 64, dropout: float = 0.1, **kw):
        super().__init__(head_hidden=head_hidden, dropout=dropout, **kw)
        del self.head
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                                                  nn.Linear(head_hidden, 2)) for _ in range(n_heads)])

    def forward(self, features, state=None, return_aux: bool = False):
        out, state = self.lstm(self.encode(features), state)
        per = torch.stack([h(out) for h in self.heads], dim=-2)          # (B, T, K, 2)
        predictions = per if self.training else per.mean(dim=-2)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


def to_single(src_dir: str, weights: str, out_dir: str):
    """Multi-head checkpoint -> standard single-head checkpoint (head_hidden = K * 64) with identical predictions."""
    src = Path(src_dir); cfg = json.load(open(src / "config.json"))
    sd = torch.load(src / weights, map_location="cpu")
    K = cfg["n_heads"]
    new = {k: v for k, v in sd.items() if not k.startswith("heads.")}
    new["head.0.weight"] = torch.cat([sd[f"heads.{k}.0.weight"] for k in range(K)], 0)
    new["head.0.bias"] = torch.cat([sd[f"heads.{k}.0.bias"] for k in range(K)], 0)
    new["head.3.weight"] = torch.cat([sd[f"heads.{k}.3.weight"] for k in range(K)], 1) / K
    new["head.3.bias"] = torch.stack([sd[f"heads.{k}.3.bias"] for k in range(K)], 0).mean(0)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(new, out / weights)
    c2 = dict(cfg); c2["head_hidden"] = cfg["head_hidden"] * K; c2["converted_from"] = f"{src_dir} ({K} heads)"
    json.dump(c2, open(out / "config.json", "w"), indent=2)


if __name__ == "__main__":
    import sys
    to_single(sys.argv[1], sys.argv[2], sys.argv[3])
