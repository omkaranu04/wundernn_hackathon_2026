"""Stacked LSTM model - identical shape to experiment 002's `StackedLSTMModel`
(plain multi-layer nn.LSTM + direct Linear head, no MLP, no input
projection), but narrower and deeper: hidden_size=32, num_layers=6 instead
of hidden_size=128, num_layers=4.

Motivated by a latency sweep (see README.md) across 14 (hidden_size,
num_layers) combinations that found this specific point - not a smooth
width-for-depth tradeoff, but a genuine local optimum - is the only
config that clears the "safe regardless of platform correction factor"
latency bar (<90-95% of budget uncorrected) while still testing whether
depth captures more than width for this task: 60,994 params (88% fewer
than experiment 002's 520,450) with 2 more recurrent layers.

Same design philosophy as experiment 002 (see its model.py): no input
projection, no attention, no windowed cache - training (whole-sequence
`nn.LSTM` call) and streaming inference (the same module called one step
at a time) are naturally numerically identical.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (num_layers, batch, hidden_size)


class StackedLSTMModel(nn.Module):
    def __init__(self, n_features: int = 112, hidden_size: int = 32, num_layers: int = 6, n_targets: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_targets)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, n_features). Returns (predictions, state).

        `state` lets callers carry (h, c) across calls - for streaming
        inference (`export_onnx.py`) and truncated-BPTT training alike.
        """
        lstm_out, state = self.lstm(features, state)
        predictions = self.head(lstm_out)
        return predictions, state
