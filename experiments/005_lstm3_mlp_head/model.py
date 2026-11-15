"""Stacked LSTM with an MLP prediction head, instead of experiment 002's
direct linear mapping from the LSTM's hidden state to [t0, t1].

    features (T, 112)
        -> multi-layer LSTM (num_layers=2, hidden_size=128 - same shape as
           the baseline's GRU backbone, isolating the head as the only
           experimental variable vs. experiment 002's plain Linear head)
        -> MLP head: Linear(128,128) -> GELU -> Linear(128,64) -> GELU -> Linear(64,2)

Same design philosophy as experiment 002 (see its model.py): no input
projection, no attention, no windowed cache. A plain recurrent stack's
hidden state is already a complete, fixed-size summary of history, so
training (whole-sequence `nn.LSTM` call) and streaming inference (the same
module called one step at a time) are naturally numerically identical -
adding an MLP head after the LSTM doesn't change that property, since the
head is a pure function of the current step's hidden state with no
additional state of its own.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (num_layers, batch, hidden_size)


class StackedLSTMMLPModel(nn.Module):
    def __init__(
        self,
        n_features: int = 112,
        hidden_size: int = 128,
        num_layers: int = 2,
        head_dims: Tuple[int, ...] = (128, 64),
        n_targets: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        layers = []
        in_dim = hidden_size
        for h in head_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_targets))
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, n_features). Returns (predictions, state).

        `state` lets callers carry (h, c) across calls - same as experiment
        002 - for streaming inference and truncated-BPTT training alike.
        """
        lstm_out, state = self.lstm(features, state)
        predictions = self.head(lstm_out)
        return predictions, state
