"""004's grouped-encoder LSTM with the features that permutation importance
(experiment 021) found unused REMOVED: instrument 1's price and volume
(52-96) and additional features a1, a3. Everything else (encoder widths,
LSTM, head) is 004's.

Kept inputs: i0_price (22) -> price encoder; i0_volume (22) -> volume encoder;
trade dp+dv for BOTH instruments (16, i1 trade features still carried a
little signal); additional a0,a2,a4,a5,a6,a7 (6) passed through raw.
The class keeps 004's name (`GroupedEncoderLSTMModel`) so train/export/eval
code is shared unchanged; this directory's model.py is the only difference.
`keep_additional` selects which additional columns survive. Keeping all 8
uses the cheap contiguous Slice; a subset uses ONE index_select (a Gather
node) - measured 2026-10-08 to be SLIGHTLY SLOWER than the Slice it
replaces (96.6% vs 94.4% for 004, despite 4 fewer nodes), so the Gather
variant trades a little latency for dropping a1/a3.

--- ORIGINAL 004 DOCSTRING FOLLOWS ---
Grouped feature-type encoders -> stacked LSTM -> MLP head.

Motivated directly by the data analysis (see analysis/README.md): price,
volume, and trade features have very different scales and shapes (price
features are heavily quantized with low variance ~0.15; volume features
have variance ~0.7; the additional a0-a7 features are on yet another scale,
~1.45, with a5/a6/a7 showing a bimodal sentinel-value pattern). Rather than
feeding all 112 raw, heterogeneously-scaled features directly into the
LSTM's input-to-hidden weight matrix, each feature *type* (price, volume,
trade) gets its own small MLP encoder - concatenating i0 and i1's columns
of that type first, so the encoder can also learn cross-instrument
structure for that type before anything else mixes it in.

    i0+i1 price   (22+22=44) -> Linear(44,16) -> ReLU                = 16
    i0+i1 volume  (22+22=44) -> Linear(44,16) -> ReLU                = 16
    i0+i1 dp+dv   (4+4+4+4=16) -> Linear(16,8) -> ReLU                = 8
    additional a0-a7 (8)     -> untouched, passed through as-is      = 8
                                                              total = 48

That 48-dim per-step vector (down from the raw 112) feeds a 2-layer LSTM
(hidden_size=128, same shape as experiments 002/003's backbone), followed
by a leaner 2-layer MLP head.

Revision history (see EXPERIMENTS.md for the full latency numbers):
  v1: 2-layer encoders per group (price/volume: 44->24->16; trade-price and
      trade-volume as two SEPARATE 8->4 encoders), GELU activations.
      Measured 123.6% of budget (synthetic, uncorrected) - worse than
      experiment 002 despite a smaller LSTM input, because the multi-branch
      slicing/concat/activation structure generates many small ONNX graph
      ops (20 Reshapes, 7 Concats, 3 Splits/step) that each carry real
      per-op dispatch overhead in ONNX Runtime, regardless of their tiny
      individual FLOP count.
  v2: swapped GELU -> ReLU (GELU decomposes into Mul/Div/Erf/Add on this
      opset; ReLU is one native op). Gave only ~1% real speedup despite
      profiling suggesting GELU was ~31% of per-step time - the profiler's
      per-node overhead doesn't translate linearly to wall-clock cost, and
      node COUNT (not op choice) is the real driver here.
  v3 (this version): removed the extra hidden layer in the price/volume
      encoders (44->24->16 -> 44->16 directly) and merged the two separate
      8->4 trade encoders into one combined 16->8 encoder - fewer branches,
      fewer Linear/activation/concat calls overall, same total output width
      (48) so the LSTM itself is unchanged.

Column indices below match `shared/data.py::FEATURE_COLUMNS`'s fixed order
(see PROBLEM_STATEMENT.md's column layout table) - i0's p/v/dp/dv occupy
0-51, i1's occupy 52-103 in the same p/v/dp/dv sub-order, additionals are
104-111. Slicing static ranges and concatenating are both plain,
ONNX-friendly tensor ops - no special handling needed for streaming
inference the way experiment 001's attention cache needed, same as
experiments 002/003.
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
KEEP_ADDITIONAL_FULL = tuple(range(8))
KEEP_ADDITIONAL_NO_A1_A3 = (0, 2, 4, 5, 6, 7)  # a1, a3 dropped (021: slightly harmful)


class GroupEncoder(nn.Module):
    """Single Linear + ReLU - no hidden layer (see revision history above)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupedEncoderLSTMModel(nn.Module):
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
        keep_additional=KEEP_ADDITIONAL_FULL,
    ):
        super().__init__()
        keep_additional = tuple(keep_additional)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.price_encoder = GroupEncoder(22, price_dim, dropout)  # i0 only (021: i1 unused)
        self.volume_encoder = GroupEncoder(22, volume_dim, dropout)  # i0 only (021: i1 unused)
        self.trade_encoder = GroupEncoder(16, trade_dim, dropout)  # merged dp+dv for both instruments

        self.full_additional = keep_additional == KEEP_ADDITIONAL_FULL
        if not self.full_additional:
            self.register_buffer("additional_idx", torch.tensor([ADDITIONAL[0] + k for k in keep_additional], dtype=torch.long), persistent=False)
        lstm_input_size = price_dim + volume_dim + trade_dim + len(keep_additional)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_targets),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., 112) -> (..., lstm_input_size). Shared by forward() and streaming export."""
        price = features[..., I0_PRICE[0]:I0_PRICE[1]]
        volume = features[..., I0_VOLUME[0]:I0_VOLUME[1]]
        trade = torch.cat([
            features[..., I0_DP[0]:I0_DP[1]], features[..., I0_DV[0]:I0_DV[1]],
            features[..., I1_DP[0]:I1_DP[1]], features[..., I1_DV[0]:I1_DV[1]],
        ], dim=-1)
        additional = (features[..., ADDITIONAL[0]:ADDITIONAL[1]] if self.full_additional
                      else torch.index_select(features, -1, self.additional_idx))

        return torch.cat([
            self.price_encoder(price),
            self.volume_encoder(volume),
            self.trade_encoder(trade),
            additional,
        ], dim=-1)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112). Returns (predictions, state)."""
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        predictions = self.head(lstm_out)
        return predictions, state
