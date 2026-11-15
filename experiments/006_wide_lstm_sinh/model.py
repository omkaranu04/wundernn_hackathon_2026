"""Encoder MLP -> single-layer, wide LSTM -> MLP head with a `sinh` final
activation, so the network's raw pre-activation output effectively lives
in an arcsinh-transformed space matched to the fat-tailed, signed target
distribution (see analysis/results/feature_statistics.md: t0 std~1.01 but
range [-125.6, 74.5]; t1 std~1.01, range [-180, 77.3], excess kurtosis 90).

`arcsinh` is ~identity near 0 and ~log(2x) for large |x|, but unlike `log`
it's defined for negative values and exactly 0 at x=0 - both essential
since targets are signed and ~22% are exactly zero (see analysis/README.md
finding on the zero-spike). Putting the transform as the model's own
output activation (`sinh`, the inverse of arcsinh) rather than
transforming targets before the loss keeps `shared/losses.py`'s existing
loss functions computing loss directly in raw units - exactly what the WP
metric scores - with no separate weighting-space bookkeeping needed.

Also departs from experiments 002/003/004/005 by using a *single*, wide
LSTM layer (rather than stacking 2-4 narrower ones) preceded by a small
MLP feature encoder, since 002/003/005 all landed within ~0.002 WP of each
other regardless of stacked depth - suggesting further local gains need a
different lever than more depth.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (num_layers, batch, hidden_size)


class WideLSTMSinhModel(nn.Module):
    def __init__(
        self,
        n_features: int = 112,
        encoder_dim: int = 64,
        hidden_size: int = 224,
        num_layers: int = 1,
        head_dims: Tuple[int, ...] = (128, 64),
        n_targets: int = 2,
        dropout: float = 0.1,
        clamp: float = 8.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.clamp = clamp

        self.encoder = nn.Sequential(nn.Linear(n_features, encoder_dim), nn.GELU())
        self.lstm = nn.LSTM(input_size=encoder_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        layers = []
        in_dim = hidden_size
        for h in head_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_targets))
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, n_features). Returns (predictions, state).

        Predictions come back already in raw target units - `sinh` undoes
        the implicit arcsinh-space the head's pre-activation output lives
        in, so callers (training loop, streaming inference) never need to
        invert anything themselves. The `clamp` (default +-8, `sinh(8) ~=
        1490`, far past the observed |target| max of ~180) is a numerical
        safety net for early training with random weights, not something
        expected to bind once the model has learned.
        """
        encoded = self.encoder(features)
        lstm_out, state = self.lstm(encoded, state)
        z = self.head(lstm_out)
        predictions = torch.sinh(torch.clamp(z, -self.clamp, self.clamp))
        return predictions, state
