"""Non-causal predictor of the hidden `is_scored` flag, per row.

Used ONLY offline to label train.parquet (which has no scoring flag); it is
never part of a submission, so it is free to look at past AND future rows.
Inputs are the 112 raw features, their step-to-step differences, and the
normalized position in the sequence - NEVER the targets t0/t1.
"""
from __future__ import annotations

import torch
from torch import nn

SEQ_LEN = 20000


class MaskPredictor(nn.Module):
    def __init__(self, hidden: int = 96, layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(112 + 112 + 1, hidden), nn.GELU())
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.out = nn.Linear((2 if bidirectional else 1) * hidden, 1)  # bidirectional=False -> strictly causal (past + current row only)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x: (B,T,112); pos: (B,T,1) absolute position / 20000. Returns logits (B,T)."""
        dx = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
        h = self.inp(torch.cat([x, dx, pos], dim=-1))
        h, _ = self.rnn(h)
        return self.out(h).squeeze(-1)


@torch.no_grad()
def predict_sequence(model: MaskPredictor, X: torch.Tensor, win: int = 2000, stride: int = 1000) -> torch.Tensor:
    """X: (T,112) on the model's device -> logits (T,). Overlapping windows;
    each window contributes only its central part (first/last windows also
    cover the sequence edges) so every row has >=500 steps of context on
    both sides, except at the true sequence ends."""
    T = X.shape[0]
    starts = list(range(0, T - win + 1, stride))
    xs = torch.stack([X[s:s + win] for s in starts])
    st = torch.tensor(starts, device=X.device, dtype=torch.float32)
    pos = ((torch.arange(win, device=X.device, dtype=torch.float32)[None] + st[:, None]) / SEQ_LEN).unsqueeze(-1)
    logits = model(xs, pos)
    out = torch.empty(T, device=X.device)
    for i, s in enumerate(starts):
        lo = 0 if i == 0 else s + win // 4
        hi = T if i == len(starts) - 1 else s + 3 * win // 4
        out[lo:hi] = logits[i, lo - s:hi - s]
    return out
