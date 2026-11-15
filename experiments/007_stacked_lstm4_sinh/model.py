"""Experiment 002's exact backbone (4-layer stacked LSTM, hidden_size=128,
direct linear head - no MLP, no input encoder) with one change: the head's
output is passed through `sinh(clamp(z, -8, 8))` instead of being used
raw. This isolates the `sinh` output transform as the *sole* experimental
variable against experiment 002 (our best real-platform-confirmed result:
0.6106 WP, 48m25s / 80.7% of the 60-minute budget), the same way
experiment 003 isolated the MLP head as the sole variable back then.

Motivation for the transform (see experiment 006's model.py for the full
reasoning, unchanged here): targets are signed, ~std-1, but extremely
fat-tailed (t0 excess kurtosis 22.4, t1 excess kurtosis 90.1, ~22% exact
zeros - analysis/results/feature_statistics.md). `sinh`'s inverse,
`arcsinh`, is ~identity near 0 and ~log(2x) for large |x| - it compresses
the rare huge-outlier rows that dominate a squared-error loss without
distorting the bulk of the distribution or requiring `log`'s
positive-only domain. Using `sinh` as the final activation keeps
`shared/losses.py`'s existing loss functions computing loss directly in
raw target units - no separate transformed-space bookkeeping needed.

`sinh` adds one elementwise op over a 2-element vector per step -
computationally negligible relative to the 4-layer LSTM stack that
dominates this model's latency, so this change is not expected to move
the latency picture from experiment 002's own numbers in any meaningful
way. Benchmarked anyway before training, per project convention (see
README.md) - this model was flagged as "already very tight on time" on
the real platform (48m25s of the 60-minute limit) and must not be pushed
closer to that limit.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (num_layers, batch, hidden_size)


class StackedLSTM4SinhModel(nn.Module):
    def __init__(
        self,
        n_features: int = 112,
        hidden_size: int = 128,
        num_layers: int = 4,
        n_targets: int = 2,
        clamp: float = 8.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.clamp = clamp
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_targets)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, n_features). Returns (predictions, state).

        Predictions come back in raw target units - `sinh` undoes the
        implicit arcsinh-space the head's pre-activation output lives in,
        same convention as experiment 006.
        """
        lstm_out, state = self.lstm(features, state)
        z = self.head(lstm_out)
        predictions = torch.sinh(torch.clamp(z, -self.clamp, self.clamp))
        return predictions, state
