"""Backbone for the XGBoost-headed model: grouped feature-type encoders ->
2-layer/128-hidden LSTM, byte-for-byte identical to experiment 004's
architecture up to (and including, for this backbone-training stage) its
2-layer MLP head.

Two distinct uses of this same class in this experiment:
  1. STAGE 1 (this file + train.py): trained normally, end-to-end, via
     backprop through the differentiable head - exactly like every other
     experiment. The head here is only a means to shape good hidden-state
     representations; it is discarded afterward.
  2. STAGE 2 (extract_embeddings.py onward): the trained LSTM+encoder is
     FROZEN and its per-step hidden state `h_t` (not the head's output) is
     used as the input to two independent XGBoost regressors - see
     extract_embeddings.py / train_xgboost.py. XGBoost is not
     differentiable, so it cannot be trained jointly with the LSTM the way
     a neural head can; this two-stage split is the reason why.

`encode_and_recur()` is the method the rest of this experiment's scripts
actually care about - it returns the LSTM's hidden state directly, without
running it through the (stage-1-only) head.
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


class BackboneModel(nn.Module):
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

        # Stage-1-only: shapes the backbone via a real differentiable
        # objective. Discarded before stage 2 (XGBoost) training.
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

    def encode_and_recur(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112) -> hidden states (batch, T, hidden_size).
        This is what stage 2 (XGBoost) actually consumes - the LSTM's own
        hidden state, not anything from the stage-1 head.
        """
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        return lstm_out, state

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """Stage-1 training only: features -> predictions via the
        discardable head. (batch, T, 112) -> (batch, T, 2)."""
        lstm_out, state = self.encode_and_recur(features, state)
        predictions = self.head(lstm_out)
        return predictions, state
