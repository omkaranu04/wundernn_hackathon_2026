"""Grouped feature-type encoders -> stacked LSTM -> MLP head, EXCEPT the
head predicts (t0, t0+t1) instead of (t0, t1) directly, exploiting the
targets' strong NEGATIVE correlation (-0.7223).

Architecture is otherwise byte-for-byte identical to experiment 004 (our
best-performing model) - same encoder groupings, same 2-layer/128-hidden
LSTM, same 2-layer MLP head shape. The only change is the last step of
forward(): the head's raw 2-dim output (z0, s) is reinterpreted as
(t0_hat, s_hat) where s targets t0+t1, then t1_hat = s_hat - t0_hat is
reconstructed before the prediction is returned - so the model's output,
the training loss, and predict()'s return value are all still in raw
[t0, t1] units, matching every other experiment. `shared/losses.py`'s
loss functions need no changes.

Why the SUM and not the difference: for two variables with correlation
rho, Var(X-Y) = Var(X)+Var(Y)-2*Cov(X,Y) and Var(X+Y) = Var(X)+Var(Y)+
2*Cov(X,Y). Since corr(t0,t1) is NEGATIVE, Cov(t0,t1) is negative, so the
SUM has LOWER variance (the anti-correlated wiggles partially cancel) and
the DIFFERENCE has HIGHER variance (they reinforce). Using the actual
numbers (std(t0)=1.006, std(t1)=1.011, corr=-0.7223): std(t0+t1)=0.75 vs
std(t0-t1)=1.87 vs std(t0)=1.006/std(t1)=1.011 individually - the sum is
the genuinely "easier", lower-variance second target, not the difference.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)


class GroupEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SumReparamGroupedEncoderLSTMModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        price_dim: int = 16,
        volume_dim: int = 16,
        trade_dim: int = 8,
        head_hidden: int = 64,
        n_targets: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.price_encoder = GroupEncoder(44, price_dim, dropout)
        self.volume_encoder = GroupEncoder(44, volume_dim, dropout)
        self.trade_encoder = GroupEncoder(16, trade_dim, dropout)

        lstm_input_size = price_dim + volume_dim + trade_dim + (ADDITIONAL[1] - ADDITIONAL[0])
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_targets),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        price = torch.cat([features[..., I0_PRICE[0]:I0_PRICE[1]], features[..., I1_PRICE[0]:I1_PRICE[1]]], dim=-1)
        volume = torch.cat([features[..., I0_VOLUME[0]:I0_VOLUME[1]], features[..., I1_VOLUME[0]:I1_VOLUME[1]]], dim=-1)
        trade = torch.cat([
            features[..., I0_DP[0]:I0_DP[1]], features[..., I0_DV[0]:I0_DV[1]],
            features[..., I1_DP[0]:I1_DP[1]], features[..., I1_DV[0]:I1_DV[1]],
        ], dim=-1)
        additional = features[..., ADDITIONAL[0]:ADDITIONAL[1]]

        return torch.cat([
            self.price_encoder(price),
            self.volume_encoder(volume),
            self.trade_encoder(trade),
            additional,
        ], dim=-1)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112). Returns (predictions, state), predictions
        already reconstructed into raw [t0, t1] units - see module docstring.
        """
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        raw = self.head(lstm_out)  # (..., 2) = (t0_hat, s_hat), s_hat targets t0+t1
        t0_hat = raw[..., 0]
        s_hat = raw[..., 1]
        t1_hat = s_hat - t0_hat
        predictions = torch.stack([t0_hat, t1_hat], dim=-1)
        return predictions, state
