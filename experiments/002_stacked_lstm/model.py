"""Stacked LSTM model - a deeper/wider analogue of the baseline's 2-layer,
128-dim stacked GRU, using LSTM cells instead of GRU cells.

Deliberately simple, mirroring the baseline's own structure closely:

    features (T, 112)
        -> multi-layer LSTM (num_layers, hidden_size configurable)
        -> linear regression head (hidden_size -> 2)

No input projection, no attention, no windowed cache. Unlike experiment
001 (LSTM + causal windowed attention), a plain recurrent stack's hidden
state IS the complete, fixed-size summary of everything seen so far - so
training (whole-sequence `nn.LSTM` call) and streaming inference (the same
`nn.LSTM` called one step at a time, state threaded through) are naturally
identical, with no reformulation needed for correctness parity the way
attention's growing cache needed one. This also means training doesn't need
"one sample = one full 20,000-step sequence" the way experiment 001 did -
ordinary truncated-BPTT chunks work fine here, since the LSTM state can be
carried across chunk boundaries with no approximation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (num_layers, batch, hidden_size)


class StackedLSTMModel(nn.Module):
    def __init__(self, n_features: int = 112, hidden_size: int = 128, num_layers: int = 2, n_targets: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_targets)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, n_features). Returns (predictions, state).

        `state` lets callers carry (h, c) across calls - both for streaming
        inference (`export_onnx.py`) and for truncated-BPTT training on
        chunks shorter than a full sequence (`train.py`), unlike experiment
        001 where only full-sequence samples gave numerically correct
        windowed attention.
        """
        lstm_out, state = self.lstm(features, state)
        predictions = self.head(lstm_out)
        return predictions, state
