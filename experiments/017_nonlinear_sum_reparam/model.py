"""Nonlinear-expanded grouped-encoder LSTM (experiment 012's exact
architecture), EXCEPT the head predicts (t0, t0+t1) instead of (t0, t1)
directly - same sum-reparameterization idea as experiment 010, just applied
on top of 012's richer input encoding instead of 004's plain one.

Everything except the last step of forward() is byte-for-byte identical to
012: same 5-channel-per-group expansion ([x, asinh(x), sin(x), cos(x),
tanh(x)]) on all 4 feature-type groups, same encoder dims (price=24,
volume=24, trade=12), same 2-layer/128-hidden LSTM, same 2-layer MLP head
shape. The head's raw 2-dim output (z0, s) is reinterpreted as (t0_hat,
s_hat) where s targets t0+t1, then t1_hat = s_hat - t0_hat is reconstructed
before the prediction is returned - so the model's output, the training
loss, and predict()'s return value are all still in raw [t0, t1] units,
matching every other experiment (`shared/losses.py` needs no changes).

Why the sum, not the difference: see experiment 010's model.py docstring -
targets are negatively correlated (-0.7223), so Var(t0+t1) < Var(t0-t1),
making the sum the genuinely "easier" second target.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)

N_TRANSFORMS = 5  # x, asinh, sin, cos, tanh


def expand(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*5): [x, asinh(x), sin(x), cos(x), tanh(x)] concatenated."""
    return torch.cat([raw, torch.asinh(raw), torch.sin(raw), torch.cos(raw), torch.tanh(raw)], dim=-1)


class GroupEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonlinearSumReparamGroupedEncoderLSTMModel(nn.Module):
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

        self.price_encoder = GroupEncoder(44 * N_TRANSFORMS, price_dim, dropout)
        self.volume_encoder = GroupEncoder(44 * N_TRANSFORMS, volume_dim, dropout)
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
        price_raw = torch.cat([features[..., I0_PRICE[0]:I0_PRICE[1]], features[..., I1_PRICE[0]:I1_PRICE[1]]], dim=-1)
        volume_raw = torch.cat([features[..., I0_VOLUME[0]:I0_VOLUME[1]], features[..., I1_VOLUME[0]:I1_VOLUME[1]]], dim=-1)
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
        """features: (batch, T, 112). Returns (predictions, state), predictions
        already reconstructed into raw [t0, t1] units - see module docstring."""
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        raw = self.head(lstm_out)  # (..., 2) = (t0_hat, s_hat), s_hat targets t0+t1
        t0_hat = raw[..., 0]
        s_hat = raw[..., 1]
        t1_hat = s_hat - t0_hat
        predictions = torch.stack([t0_hat, t1_hat], dim=-1)
        return predictions, state
