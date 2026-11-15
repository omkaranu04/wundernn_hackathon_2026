"""Per-instrument, per-feature-type encoding -> stacked LSTM -> MLP head.

    i0_p  (22) -> 8   \\
    i1_p  (22) -> 8    \\
    i0_v  (22) -> 8     \\
    i1_v  (22) -> 8      >-- concat (44) -> LSTM(hidden=64, layers=4) -> head
    i0_dp (4)  -> 2      /
    i0_dv (4)  -> 2     /
    i1_dp (4)  -> 2    /
    i1_dv (4)  -> 2   /
    a0-a7 (8)  -> 4  /

A finer-grained version of experiment 004's grouped-encoder idea: 004
merged i0+i1 together within each feature type; this keeps i0 and i1
SEPARATE per type (9 independent per-group projections instead of 004's
~5), testing whether letting the model learn instrument-specific
compressions first, before any cross-instrument mixing happens in the
LSTM, captures more than sharing an encoder across both instruments does.

IMPLEMENTATION NOTE - why this isn't 9 separate `nn.Linear` modules:
a first version built with 9 literal `nn.Linear(group_dim, out_dim)`
branches + concat measured 117 ONNX nodes and 137.6% of budget
uncorrected (each `Linear->GELU` branch decomposes into several ONNX
primitives - Gemm, Erf, Mul, Div, Add - so 9 small branches multiply out
to ~90+ nodes before the LSTM even runs). That's a real latency risk for
a branching architecture (see experiment 004's README: branching designs
can't be assumed to get any platform speedup, and need <70% of budget
uncorrected as a safety target, not the ~90-100% that's fine for simple
sequential architectures).

Fix: `BlockDiagonalEncoder` below computes the mathematically IDENTICAL
function - each of the 9 groups gets its own learned linear projection,
with zero cross-group mixing - but as a single `Linear(112, 44)` whose
weight is multiplied by a fixed (non-trainable) block-diagonal 0/1 mask
before use, followed by one shared `GELU` over the whole 44-dim output
(valid because GELU is elementwise: applying it once to the concatenation
is identical to applying it per-group then concatenating). This collapses
the ~90-node multi-branch subgraph down to a handful of nodes (one Mul for
the mask, one Gemm, one GELU), with no change to what the model can
represent - i0 and i1 (and every other group) still never share weights
in this projection, exactly as the original design intended.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

# (name, start, end) slices into the 112-dim feature vector, per
# PROBLEM_STATEMENT.md's documented column layout.
GROUP_SLICES = [
    ("i0_p", 0, 22),
    ("i0_v", 22, 44),
    ("i0_dp", 44, 48),
    ("i0_dv", 48, 52),
    ("i1_p", 52, 74),
    ("i1_v", 74, 96),
    ("i1_dp", 96, 100),
    ("i1_dv", 100, 104),
    ("additional", 104, 112),
]
GROUP_OUT_DIMS = {
    "i0_p": 8, "i0_v": 8, "i0_dp": 2, "i0_dv": 2,
    "i1_p": 8, "i1_v": 8, "i1_dp": 2, "i1_dv": 2,
    "additional": 4,
}
ENCODED_DIM = sum(GROUP_OUT_DIMS[name] for name, _, _ in GROUP_SLICES)  # 44


class BlockDiagonalEncoder(nn.Module):
    """9 independent per-group linear projections, implemented as one
    masked Linear + one shared GELU for a lean ONNX graph. See module
    docstring for why this is equivalent to 9 separate small encoders.
    """

    def __init__(self, n_features: int = 112, encoded_dim: int = ENCODED_DIM):
        super().__init__()
        linear = nn.Linear(n_features, encoded_dim)  # borrow standard PyTorch init
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone())

        mask = torch.zeros(encoded_dim, n_features)
        out_start = 0
        for name, in_start, in_end in GROUP_SLICES:
            out_dim = GROUP_OUT_DIMS[name]
            mask[out_start:out_start + out_dim, in_start:in_end] = 1.0
            out_start += out_dim
        self.register_buffer("mask", mask)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return self.act(F.linear(x, w, self.bias))


class SplitGroupedEncoderLSTMModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 4,
        head_dims: Tuple[int, ...] = (16,),
        n_targets: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = BlockDiagonalEncoder()
        self.lstm = nn.LSTM(input_size=ENCODED_DIM, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        layers = []
        in_dim = hidden_size
        for h in head_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_targets))
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112). Returns (predictions, state)."""
        encoded = self.encoder(features)
        lstm_out, state = self.lstm(encoded, state)
        predictions = self.head(lstm_out)
        return predictions, state
