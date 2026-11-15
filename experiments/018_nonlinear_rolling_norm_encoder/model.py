"""Nonlinear-expanded + rolling-normalized grouped-encoder LSTM.

Combines the two feature-engineering levers that have actually moved WP in
this project (see EXPERIMENTS.md's "Open directions for future experiments"
section), at FULL capacity for both, rather than testing either in
isolation:

1. **Experiment 012's 5x nonlinear expansion** - every raw group
   (price/volume/trade/additional) expanded into
   `[x, asinh(x), sin(x), cos(x), tanh(x)]` before its own MLP encoder.
   Unchanged from 012: see that experiment's model.py docstring for the
   full justification (feature_statistics.md's mixed-sign/already-
   normalized-range argument against log/e^x).
2. **Experiment 013's rolling EWMA-normalization, price-only, 2 windows
   (50/200 steps), vectorized across windows** - unchanged from 013's
   latency-fixed version (fixes #1-#3 all baked in from the start here,
   not retrofitted after a timeout like 013's own history).

This directly tests the open question neither 012 nor 013 alone answers:
does *temporal* normalization (EWMA, price-only) carry signal beyond what
*static* nonlinear transforms (012, all groups) already capture? 013-alone
(price-only EWMA, no nonlinear expansion: local WP 0.678768) scored BELOW
both plain 004 (0.682425) and 012 (0.682697) - so this is a genuine,
not-pre-decided test, not an assumed win. See EXPERIMENTS.md for the result
once trained.

## How the two are combined for price specifically

Only price gets BOTH treatments; volume/trade/additional get exactly 012's
nonlinear expansion only (EWMA was already shown price-only is the
data-justified scope - see 013's module docstring's excess-kurtosis
argument, unchanged here).

Per-step price encoder input, ADDITIVE not a replacement for either half:

    expand(price_raw)                    44*5 = 220   (012's nonlinear 5x)
    + ewma_normalized_blocks_only         44*2 =  88   (013's 2 EWMA windows,
                                                         RAW block dropped -
                                                         already present as
                                                         expand()'s identity
                                                         channel, no need to
                                                         duplicate it)
    = price_encoder input width                  308

Dropping the redundant raw block from the EWMA output (rather than
reusing 013's `rolling_normalize_batched`/`rolling_normalize_step` output
whole) avoids feeding the model the same 44 raw price values twice under
two different names - a free simplification, not a design compromise.

Volume/trade/additional: byte-for-byte identical to 012 - `expand()` each,
same encoder shapes, additional still has no encoder (raw pass-through).

## Node-count / latency expectation (not yet measured - see the pre-training
## benchmark this experiment's auto_pipeline.sh runs before any training)

012 alone measured 62 ONNX nodes / 112.2% of budget uncorrected (trained
weights). 013's price-only+2-window+vectorized EWMA machinery added ~15
nodes over a 43-node (004-shaped) base. Stacking that same EWMA block onto
012's structure is expected to land somewhere around 75-85 nodes - this
project's own node-count/latency trend (see
analysis/results/timing_correlations.md: r=0.928, roughly ~1%/node in the
60-130 node range) suggests that's plausibly **140-170% of budget
uncorrected BEFORE any further optimization** - i.e. likely still over
budget even after reusing 013's validated node-reduction techniques from
the start. This is a prediction, not a result - `auto_pipeline.sh` benchmarks
on random-init weights before committing to training, and if it lands high,
the next lever to pull (not yet applied here) would be dropping to a single
EWMA window (span=200 only) rather than two, mirroring 013's own
window-count-reduction fix.

## Streaming state

Same as 013: the exported single-step ONNX graph threads the EWMA's
(mean, meansq) state (each shape (N_WINDOWS, N_PRICE) = (2, 44)) alongside
the LSTM's (h, c), all reset to zero at the start of each sequence. The
nonlinear transforms (asinh/sin/cos/tanh) are pointwise and stateless -
they don't add anything to the streaming state, only to the per-step
encoder's input width.

Column indices below match `shared/data.py::FEATURE_COLUMNS`'s fixed order,
identical to every other grouped-encoder experiment (004/009/012/013).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

# (start, end) slices into the 112-dim raw feature vector, per shared/data.py's FEATURE_COLUMNS order
I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)
N_PRICE = 44
N_VOLUME = 44
N_TRADE = 16
N_ADDITIONAL = ADDITIONAL[1] - ADDITIONAL[0]

N_TRANSFORMS = 5  # x, asinh, sin, cos, tanh - experiment 012's expansion

# experiment 013's price-only EWMA, unchanged (2 windows, vectorized)
SPANS = (50, 200)
ALPHAS = tuple(2.0 / (s + 1) for s in SPANS)
KERNEL_SIZES = (300, 1200)  # truncated-kernel length per span, >=99.999% EWMA weight coverage
VAR_EPS = 1e-4
N_WINDOWS = len(SPANS)
ROLLING_STATE_NAMES = ["mean", "meansq"]  # each (N_WINDOWS, N_PRICE)

PRICE_ENCODER_IN = N_PRICE * N_TRANSFORMS + N_PRICE * N_WINDOWS  # 220 + 88 = 308

_ALPHA_VEC = torch.tensor(ALPHAS, dtype=torch.float32).view(-1, 1)
_ONE_MINUS_ALPHA_VEC = 1.0 - _ALPHA_VEC


def expand(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*5): [x, asinh(x), sin(x), cos(x), tanh(x)] concatenated. Identical to 012's."""
    return torch.cat([raw, torch.asinh(raw), torch.sin(raw), torch.cos(raw), torch.tanh(raw)], dim=-1)


def _ewma_kernel(alpha: float, kernel_size: int, device, dtype) -> torch.Tensor:
    k = alpha * (1 - alpha) ** torch.arange(kernel_size, device=device, dtype=torch.float64)
    return k.flip(0).to(dtype)  # most-recent weight last: aligns with causal left-padding + conv1d


def rolling_normalize_batched(price_raw: torch.Tensor) -> torch.Tensor:
    """price_raw: (batch, T, N_PRICE) -> (batch, T, N_PRICE*N_WINDOWS), the
    normalized blocks ONLY (no raw block - unlike 013's version, since the
    caller already has the raw price via expand()'s identity channel).
    Training-time path: truncated causal depthwise conv1d approximation of
    the EWMA - identical technique to 013's, see its module docstring for
    why (a 20,000-step Python recurrence would run un-fused alongside
    cuDNN's LSTM)."""
    batch, T, n_features = price_raw.shape
    x = price_raw.transpose(1, 2)  # (batch, N_PRICE, T)
    x2 = x * x

    outputs = []
    for alpha, kernel_size in zip(ALPHAS, KERNEL_SIZES):
        kernel = _ewma_kernel(alpha, kernel_size, x.device, x.dtype)
        weight = kernel.view(1, 1, -1).repeat(n_features, 1, 1)

        xp = F.pad(x, (kernel_size - 1, 0))
        mean = F.conv1d(xp, weight, groups=n_features)
        x2p = F.pad(x2, (kernel_size - 1, 0))
        meansq = F.conv1d(x2p, weight, groups=n_features)

        var = (meansq - mean * mean).clamp_min(VAR_EPS)
        normalized = (x - mean) / var.sqrt()
        outputs.append(normalized.transpose(1, 2))

    return torch.cat(outputs, dim=-1)


def rolling_normalize_step(
    price_raw: torch.Tensor, state: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """price_raw: (N_PRICE,). state: {'mean': (N_WINDOWS,N_PRICE), 'meansq': (N_WINDOWS,N_PRICE)}.
    Returns (normalized blocks only, (N_PRICE*N_WINDOWS,), new_state) -
    streaming/export path, exact recursive EWMA, vectorized across windows
    (identical technique to 013's latency fix #1). No raw block in the
    output - see module docstring."""
    mean = state["mean"]
    meansq = state["meansq"]
    new_mean = _ONE_MINUS_ALPHA_VEC * mean + _ALPHA_VEC * price_raw
    new_meansq = _ONE_MINUS_ALPHA_VEC * meansq + _ALPHA_VEC * (price_raw * price_raw)
    var = (new_meansq - new_mean * new_mean).clamp_min(VAR_EPS)
    normalized = (price_raw - new_mean) / var.sqrt()  # (N_WINDOWS, N_PRICE)
    new_state = {"mean": new_mean, "meansq": new_meansq}
    return normalized.reshape(-1), new_state


def initial_rolling_state(n_features: int = N_PRICE) -> Dict[str, torch.Tensor]:
    return {"mean": torch.zeros(N_WINDOWS, n_features), "meansq": torch.zeros(N_WINDOWS, n_features)}


class GroupEncoder(nn.Module):
    """Single Linear + ReLU - same shape as 004/012/013's."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonlinearRollingNormGroupedEncoderLSTMModel(nn.Module):
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

        self.price_encoder = GroupEncoder(PRICE_ENCODER_IN, price_dim, dropout)
        self.volume_encoder = GroupEncoder(N_VOLUME * N_TRANSFORMS, volume_dim, dropout)
        self.trade_encoder = GroupEncoder(N_TRADE * N_TRANSFORMS, trade_dim, dropout)

        additional_dim = N_ADDITIONAL * N_TRANSFORMS
        lstm_input_size = price_dim + volume_dim + trade_dim + additional_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_targets),
        )

    def raw_groups(self, features: torch.Tensor):
        """features: (..., 112) -> (price_raw, volume_raw, trade_raw, additional_raw),
        widths (44, 44, 16, 8). Identical slicing to 004/009/012/013."""
        price_raw = torch.cat([features[..., I0_PRICE[0]:I0_PRICE[1]], features[..., I1_PRICE[0]:I1_PRICE[1]]], dim=-1)
        volume_raw = torch.cat([features[..., I0_VOLUME[0]:I0_VOLUME[1]], features[..., I1_VOLUME[0]:I1_VOLUME[1]]], dim=-1)
        trade_raw = torch.cat([
            features[..., I0_DP[0]:I0_DP[1]], features[..., I0_DV[0]:I0_DV[1]],
            features[..., I1_DP[0]:I1_DP[1]], features[..., I1_DV[0]:I1_DV[1]],
        ], dim=-1)
        additional_raw = features[..., ADDITIONAL[0]:ADDITIONAL[1]]
        return price_raw, volume_raw, trade_raw, additional_raw

    def encode_from_price_parts(
        self, price_nonlinear: torch.Tensor, price_ewma: torch.Tensor,
        volume_raw: torch.Tensor, trade_raw: torch.Tensor, additional_raw: torch.Tensor,
    ) -> torch.Tensor:
        """price_nonlinear: (..., N_PRICE*5) = expand(price_raw).
        price_ewma: (..., N_PRICE*N_WINDOWS), already rolling-normalized
        (via rolling_normalize_batched or rolling_normalize_step). Shared
        by forward() and the streaming export path."""
        price_in = torch.cat([price_nonlinear, price_ewma], dim=-1)
        return torch.cat([
            self.price_encoder(price_in),
            self.volume_encoder(expand(volume_raw)),
            self.trade_encoder(expand(trade_raw)),
            expand(additional_raw),
        ], dim=-1)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112). Returns (predictions, state)."""
        price_raw, volume_raw, trade_raw, additional_raw = self.raw_groups(features)
        price_nonlinear = expand(price_raw)
        price_ewma = rolling_normalize_batched(price_raw)
        encoded = self.encode_from_price_parts(price_nonlinear, price_ewma, volume_raw, trade_raw, additional_raw)
        lstm_out, state = self.lstm(encoded, state)
        predictions = self.head(lstm_out)
        return predictions, state
