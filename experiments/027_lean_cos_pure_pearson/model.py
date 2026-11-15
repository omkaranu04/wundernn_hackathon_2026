"""Lean cos-expanded grouped-encoder LSTM (experiment 027).

A deliberately slimmed variant of experiment 012:
- INPUTS (68 of the 112 features): instrument-0 price (22) + instrument-0
  volume (22) + ALL trade features dp/dv of both instruments (16) + the 8
  `additional` features. Instrument 1's price and volume (44 features) are
  DROPPED (experiments 021/022: effectively unused).
- EXPANSION: each kept group is expanded to `[x, cos(x)]` (2 channels)
  instead of 012's 5 - experiment 021's channel test found `x` and `cos(x)`
  are the channels 012 actually relies on (asinh/sin/tanh were redundant).
- ENCODERS: grouped MLPs exactly as 012 (Linear + ReLU + Dropout):
  price 44->24, volume 44->24, trade 32->12. The `additional` group is
  expanded to 16 dims and passed straight to the LSTM, as in 012/004 (an
  interpretation choice: the spec said "grouped MLPs" and 012's additional
  group had no MLP; a 16->k MLP for additional is a one-line change).
- RECURRENT CORE: 2-layer LSTM, hidden 128, then 012's head
  Linear(128,64)-ReLU-Dropout-Linear(64,2) predicting (t0, t1) directly.
Per-step vector: 24 + 24 + 12 + 16 = 76 dims (012: 100).
The training loss for this experiment's main arm is pure global weighted
Pearson - see train.py / README.md.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

# (start, end) slices into the 112-dim raw feature vector, per shared/data.py's FEATURE_COLUMNS order
I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)

N_TRANSFORMS = 2  # x, cos


def expand(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*2): [x, cos(x)] concatenated."""
    return torch.cat([raw, torch.cos(raw)], dim=-1)


class GroupEncoder(nn.Module):
    """Single Linear + ReLU - same shape as 004's (see its revision history for why no hidden layer)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeanCosGroupedEncoderLSTMModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        price_dim: int = 24,
        volume_dim: int = 24,
        trade_dim: int = 12,
        head_hidden: int = 64,
        n_targets: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.price_encoder = GroupEncoder(22 * N_TRANSFORMS, price_dim, dropout)   # i0 price only
        self.volume_encoder = GroupEncoder(22 * N_TRANSFORMS, volume_dim, dropout)  # i0 volume only
        self.trade_encoder = GroupEncoder(16 * N_TRANSFORMS, trade_dim, dropout)

        additional_dim = (ADDITIONAL[1] - ADDITIONAL[0]) * N_TRANSFORMS
        lstm_input_size = price_dim + volume_dim + trade_dim + additional_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_targets),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., 112) -> (..., lstm_input_size). Shared by forward() and streaming export."""
        price_raw = features[..., I0_PRICE[0]:I0_PRICE[1]]      # instrument 0 only
        volume_raw = features[..., I0_VOLUME[0]:I0_VOLUME[1]]   # instrument 0 only
        trade_raw = torch.cat([
            features[..., I0_DP[0]:I0_DP[1]], features[..., I0_DV[0]:I0_DV[1]],
            features[..., I1_DP[0]:I1_DP[1]], features[..., I1_DV[0]:I1_DV[1]],
        ], dim=-1)
        additional_raw = features[..., ADDITIONAL[0]:ADDITIONAL[1]]

        return torch.cat([
            self.price_encoder(expand(price_raw)),
            self.volume_encoder(expand(volume_raw)),
            self.trade_encoder(expand(trade_raw)),
            expand(additional_raw),
        ], dim=-1)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112). Returns (predictions, state)."""
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        predictions = self.head(lstm_out)
        return predictions, state
