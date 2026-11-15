"""Nonlinear-expanded grouped-encoder LSTM.

Same grouped per-feature-type encoder structure as experiment 004 (see its
model.py docstring for the original motivation), but each RAW group
(price/volume/trade/additional) is expanded into 5 channels before its
encoder sees it: the raw value itself plus 4 nonlinear transforms, per
explicit user request (at least 4 nonlinearities, applied to every feature,
proceeding regardless of the latency-budget finding described below).

Transform choice is grounded in analysis/results/feature_statistics.md, not
picked blind: features here are already normalized to roughly [-5.5, 5.5]
and are heavily MIXED-SIGN (frac_negative/frac_positive both nontrivial for
almost every column) - which rules out raw log(x) (undefined for x<=0,
true for roughly half of most columns' rows) and makes raw e^x a poor fit
(finite at this range, but badly scale-mismatched against the other O(1)
channels: e^5.5 = 245 next to channels living in [-5.5, 5.5]). Transforms
used instead, all domain-safe and O(1)-scaled at this input range:

    x                       identity
    asinh(x)                signed, log-like tail compression - same shape
                             as sign(x)*log1p(|x|) but exports as a single
                             native ONNX op instead of a 4-5 op
                             Sign/Abs/Add/Log/Mul decomposition; compresses
                             the fat tails price features specifically show
                             (excess_kurtosis avg ~109-225 there, vs ~1-5
                             for volume/trade, ~0.1 for additional, per
                             feature_statistics.md)
    sin(x), cos(x)          bounded [-1,1], finite and well-behaved here
    tanh(x)                 bounded [-1,1] saturating nonlinearity

giving each group 5x its raw width before its own MLP encoder:

    i0+i1 price   (44*5=220) -> Linear(220,24) -> ReLU               = 24
    i0+i1 volume  (44*5=220) -> Linear(220,24) -> ReLU               = 24
    i0+i1 dp+dv   (16*5=80)  -> Linear(80,12)  -> ReLU               = 12
    additional a0-a7 (8*5=40) -> untouched, passed through as-is     = 40
                                                               total = 100

That 100-dim per-step vector feeds a 2-layer LSTM (hidden_size=128, same
shape as experiment 004), followed by the same 2-layer MLP head.

Latency note: benchmarked at 122.7% of budget uncorrected on
randomly-initialized weights (see EXPERIMENTS.md for the full
investigation, including cheaper variants that were tried and discarded -
scoping to price-only measured 97.5% but was explicitly rejected in favor
of the full 4-nonlinearity/all-groups version below). Proceeding per
explicit user instruction to ignore the latency finding.

Column indices below match `shared/data.py::FEATURE_COLUMNS`'s fixed order,
identical to experiment 004's.
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

N_TRANSFORMS = 5  # x, asinh, sin, cos, tanh


def expand(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*5): [x, asinh(x), sin(x), cos(x), tanh(x)] concatenated."""
    return torch.cat([raw, torch.asinh(raw), torch.sin(raw), torch.cos(raw), torch.tanh(raw)], dim=-1)


class GroupEncoder(nn.Module):
    """Single Linear + ReLU - same shape as 004's (see its revision history for why no hidden layer)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonlinearGroupedEncoderLSTMModel(nn.Module):
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
        """features: (batch, T, 112). Returns (predictions, state)."""
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        predictions = self.head(lstm_out)
        return predictions, state
