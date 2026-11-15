"""Rolling-normalized grouped-encoder LSTM.

User's proposal: for each of the 112 raw features x, compute rolling
mean/std over several window lengths and normalize against them, giving
x -> (x, x1, x2, x3, x4) where x1..x4 are the normalized versions - then
feed this into 004's grouped per-feature-type encoder structure (2-layer/
128-hidden LSTM, MLP head, 30 epochs).

UPDATED 2026-09-23 (latency fix #1, vectorization): after this timed out
on the platform (181.9%/179.9% of budget uncorrected), vectorized the
streaming EWMA update across all windows at once (single broadcasted op
on `(N_WINDOWS,112)`-shaped state) instead of a Python loop computing
each window separately - same math, same windows, same full mean+std
normalization, just far fewer ONNX nodes and state I/O ports. Cut latency
128->94 nodes, 179.9%->143.1% uncorrected - a real ~20% reduction, still
over budget.

UPDATED 2026-09-23 (latency fix #2, window-count reduction): trimmed from
4 windows (50/100/150/200) to 2 (50/200 - the two extremes, keeping the
widest short/long coverage for the fewest windows). Nodes 94->76,
143.1%->127.8% uncorrected (random-init). Still over budget.

UPDATED 2026-09-23 (latency fix #3, group-scoping): rather than applying
the rolling normalization to all 4 feature-type groups (price/volume/
trade/additional), scoped it to **price only** - the data-grounded
justification is the same one that drove experiment 012's own price-only
compromise: per-feature-type-section excess_kurtosis in
`analysis/results/feature_statistics.md` shows the fat-tail/"extreme
recent swings" problem rolling normalization is meant to compress is
almost entirely a PRICE phenomenon (avg ~109-225, max 1064) vs. ~1-5 for
volume/trade and ~0.1 for additional - those groups are already close to
normally-distributed, so normalizing them against a rolling mean/std had
little to actually correct.

This is a bigger structural win than "1/4 the work": the ONLY reason the
prior design needed the "expand the full 112-dim vector once, then
re-slice each group's columns out of it" machinery (`_gather_group`, the
dominant remaining `Slice`-node cost after fixes #1/#2) was that MULTIPLE
groups needed access to different sub-ranges of the same globally-
expanded tensor. With only price expanded, that whole cross-group
reslicing structure is gone - it's a straight pipe: slice price's 44 raw
columns -> EWMA-normalize just that 44-dim slice (same vectorized-across-
windows technique, just narrower) -> concat with the raw values -> feed
to price's encoder. Volume/trade/additional go back to exactly what they
were in plain 004/012 - a raw slice straight into their own encoder (or,
for "additional", no encoder at all), untouched by any rolling-stat
machinery. `_gather_group` and the block-major layout it needed are
removed entirely. See EXPERIMENTS.md for the measured latency.

Real tradeoff, not free: this narrows the technique's scope from "every
one of the 112 raw features" down to "price features only" - volume/
trade/additional no longer carry any rolling-normalized information, only
their raw values. Data-justified, but a genuine reduction in fidelity to
the original ask, flagged explicitly rather than glossed over.

Two things needed grounding before implementation (see EXPERIMENTS.md for
the full investigation and the user's own answers to the clarifying
questions this raised):

1. **Window sizes.** The original ask used calendar-day windows
   (7/14/21/28 days). This dataset's 20,000-step sequences are estimated
   at sub-second-to-low-second granularity (analysis/MARKET_GLOSSARY.md) -
   a whole sequence likely spans minutes, so literal day-windows would
   exceed the entire sequence. Rescaled proportionally, preserving the
   original 1:2:3:4 ratio: **50/100/150/200 steps** (~0.25%/0.5%/0.75%/1%
   of a 20,000-step sequence) - short local smoothing windows appropriate
   for this data's actual timescale. (Now trimmed to 50/200 - see above.)
2. **Exact windowed vs. exponentially-weighted (EWMA).** A true fixed-N
   rolling window needs the model to carry a buffer of the last N raw
   values per feature as extra streaming state - a new kind of state for
   this project (nothing like it exists outside 001's attention cache),
   and an O(N) per-step update cost on top of it. EWMA is the standard
   streaming substitute (used throughout real trading-feature-engineering
   practice): O(1) state per feature per window (running mean, running
   mean-of-squares), O(1) per-step update. Chosen per explicit instruction.

## The two-computation-path design (like every other streaming experiment
## here, but with a new subtlety this one specifically needs)

Every experiment's `model.forward()` (used for training, sees a full
20,000-step batch at once) and its exported single-step ONNX graph (used
for streaming inference) compute the SAME function - so far, exactly the
same, to floating-point precision. This experiment is the first exception,
by necessity:

- **Training** needs the EWMA computed over a full (batch, 20000, 44)
  price tensor in one vectorized shot - a 20,000-step Python-level
  recurrence loop (the "obviously correct" way to compute an EWMA) would
  run alongside cuDNN's already-fused LSTM as a second, un-fused
  sequential loop and would likely dominate training time. Instead,
  `forward()` uses `rolling_normalize_batched()`: each window's EWMA is
  computed via a **truncated causal depthwise 1D convolution** -
  mathematically an EWMA is an infinite impulse response filter, so this
  is a finite-impulse-response approximation, with the kernel length
  chosen so the truncated tail carries >99.999% of the total weight (see
  KERNEL_SIZES below).
- **Streaming export** doesn't have this problem - only one step at a
  time, no batch dimension to vectorize over - so `rolling_normalize_step()`
  uses the mathematically **exact** recursive EWMA update instead.

These two paths are validated to agree closely (not identically - see
`export_onnx.py`'s `verify()` and EXPERIMENTS.md for the measured
discrepancy, ~5e-5 in the underlying mean/variance estimates at these
kernel sizes) rather than assumed to agree, since they are genuinely
different computations by construction, unlike every other experiment's
train/export pair which are the exact same computation graph.

## Cold-start behavior (documented, not "fixed")

Both paths initialize each window's running mean/meansq to zero at the
start of a sequence (same convention as the LSTM's own zero initial
state). This means normalized features are biased/compressed during each
sequence's first ~1-4x the window span (up to ~800 steps for the 200-span
window) while the EWMA "warms up" - analogous to, and no worse than, the
LSTM's own well-known warmup behavior. A small variance floor (VAR_EPS)
prevents literal division-by-zero at the very first step (a single data
point has zero variance by construction). No bias-correction (a la Adam's
`m_hat = m_t/(1-beta^t)`) is applied - a scope decision to keep the state
and computation simpler, not a correctness requirement; documented here
so it isn't mistaken for an oversight.
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
N_RAW = 112
N_PRICE = 44  # i0_price(22) + i1_price(22) - the only group that gets rolling-normalized now

SPANS = (50, 200)
ALPHAS = tuple(2.0 / (s + 1) for s in SPANS)
# Truncated-kernel length per span, sized for >=99.999% EWMA weight coverage
# (validated: max abs diff vs. the exact recursion ~5e-5 for mean and
# variance at these sizes - see EXPERIMENTS.md).
KERNEL_SIZES = (300, 1200)
VAR_EPS = 1e-4
N_BLOCKS = len(SPANS) + 1  # raw price + 2 normalized windows (price only, see module docstring)
N_WINDOWS = len(SPANS)
ROLLING_STATE_NAMES = ["mean", "meansq"]  # each (N_WINDOWS, N_PRICE) - see rolling_normalize_step

# Vectorized-across-windows constants for the streaming step (see
# rolling_normalize_step's docstring): (N_WINDOWS, 1), SPANS order,
# precomputed once so (1-alpha) isn't recomputed as a graph node every call.
_ALPHA_VEC = torch.tensor(ALPHAS, dtype=torch.float32).view(-1, 1)
_ONE_MINUS_ALPHA_VEC = 1.0 - _ALPHA_VEC


def _ewma_kernel(alpha: float, kernel_size: int, device, dtype) -> torch.Tensor:
    k = alpha * (1 - alpha) ** torch.arange(kernel_size, device=device, dtype=torch.float64)
    return k.flip(0).to(dtype)  # most-recent weight last: aligns with causal left-padding + conv1d


def rolling_normalize_batched(price_raw: torch.Tensor) -> torch.Tensor:
    """price_raw: (batch, T, N_PRICE) -> (batch, T, N_PRICE*N_BLOCKS) block-major
    [price, norm_50, norm_200]. Training-time path (see module docstring
    for why this uses a truncated convolution, not a Python-level
    recurrence). PRICE ONLY as of the 2026-09-23 group-scoping fix - see
    module docstring."""
    batch, T, n_features = price_raw.shape
    x = price_raw.transpose(1, 2)  # (batch, N_PRICE, T) - conv1d wants channels second
    x2 = x * x

    outputs = [price_raw]
    for alpha, kernel_size in zip(ALPHAS, KERNEL_SIZES):
        kernel = _ewma_kernel(alpha, kernel_size, x.device, x.dtype)
        weight = kernel.view(1, 1, -1).repeat(n_features, 1, 1)  # depthwise: identical kernel per channel

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
    Returns (block-major expanded (N_PRICE*N_BLOCKS,), new_state) -
    streaming/export path, exact recursive EWMA (update-then-normalize,
    matching the causal convention the training-time convolution also
    uses). PRICE ONLY as of the 2026-09-23 group-scoping fix.

    VECTORIZED ACROSS WINDOWS (2026-09-23, latency fix #1): computes all
    windows' updates as ONE set of ops on (N_WINDOWS, N_PRICE)-shaped
    tensors, broadcasting `price_raw` (N_PRICE,) against the
    (N_WINDOWS, 1) alpha vector, instead of a Python loop doing it once
    per window. Same math, same windows, same full mean+std z-score
    normalization - only the graph shape changed.
    """
    mean = state["mean"]      # (N_WINDOWS, N_PRICE)
    meansq = state["meansq"]  # (N_WINDOWS, N_PRICE)
    new_mean = _ONE_MINUS_ALPHA_VEC * mean + _ALPHA_VEC * price_raw
    new_meansq = _ONE_MINUS_ALPHA_VEC * meansq + _ALPHA_VEC * (price_raw * price_raw)
    var = (new_meansq - new_mean * new_mean).clamp_min(VAR_EPS)
    normalized = (price_raw - new_mean) / var.sqrt()  # (N_WINDOWS, N_PRICE)
    # normalized.reshape(-1) on a row-major (N_WINDOWS,N_PRICE) tensor gives
    # exactly [norm_50(N_PRICE), norm_200(N_PRICE)] concatenated - the same
    # block order rolling_normalize_batched produces.
    expanded = torch.cat([price_raw, normalized.reshape(-1)], dim=-1)
    new_state = {"mean": new_mean, "meansq": new_meansq}
    return expanded, new_state


def initial_rolling_state(n_features: int = N_PRICE) -> Dict[str, torch.Tensor]:
    return {"mean": torch.zeros(N_WINDOWS, n_features), "meansq": torch.zeros(N_WINDOWS, n_features)}


class GroupEncoder(nn.Module):
    """Single Linear + ReLU - same shape as 004's (see its revision history for why no hidden layer)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RollingNormGroupedEncoderLSTMModel(nn.Module):
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

        # Only price gets the N_BLOCKS-wide (rolling-normalized) input now;
        # volume/trade/additional are raw, untouched - same shape as 004/012.
        self.price_encoder = GroupEncoder(N_PRICE * N_BLOCKS, price_dim, dropout)
        self.volume_encoder = GroupEncoder(44, volume_dim, dropout)
        self.trade_encoder = GroupEncoder(16, trade_dim, dropout)

        additional_dim = ADDITIONAL[1] - ADDITIONAL[0]
        lstm_input_size = price_dim + volume_dim + trade_dim + additional_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_targets),
        )

    def raw_groups(self, features: torch.Tensor):
        """features: (..., 112) -> (price_raw, volume_raw, trade_raw, additional_raw),
        widths (N_PRICE, 44, 16, 8). Shared by both the batched (training)
        and streaming (export) paths - see forward() and export_onnx.py."""
        price_raw = torch.cat([features[..., I0_PRICE[0]:I0_PRICE[1]], features[..., I1_PRICE[0]:I1_PRICE[1]]], dim=-1)
        volume_raw = torch.cat([features[..., I0_VOLUME[0]:I0_VOLUME[1]], features[..., I1_VOLUME[0]:I1_VOLUME[1]]], dim=-1)
        trade_raw = torch.cat([
            features[..., I0_DP[0]:I0_DP[1]], features[..., I0_DV[0]:I0_DV[1]],
            features[..., I1_DP[0]:I1_DP[1]], features[..., I1_DV[0]:I1_DV[1]],
        ], dim=-1)
        additional_raw = features[..., ADDITIONAL[0]:ADDITIONAL[1]]
        return price_raw, volume_raw, trade_raw, additional_raw

    def encode_from_price_expanded(
        self, price_expanded: torch.Tensor, volume_raw: torch.Tensor, trade_raw: torch.Tensor, additional_raw: torch.Tensor
    ) -> torch.Tensor:
        """price_expanded: (..., N_PRICE*N_BLOCKS) already rolling-normalized
        (via rolling_normalize_batched or rolling_normalize_step, depending
        on the caller). volume_raw/trade_raw/additional_raw: raw, untouched."""
        return torch.cat([
            self.price_encoder(price_expanded),
            self.volume_encoder(volume_raw),
            self.trade_encoder(trade_raw),
            additional_raw,
        ], dim=-1)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112). Returns (predictions, state)."""
        price_raw, volume_raw, trade_raw, additional_raw = self.raw_groups(features)
        price_expanded = rolling_normalize_batched(price_raw)
        encoded = self.encode_from_price_expanded(price_expanded, volume_raw, trade_raw, additional_raw)
        lstm_out, state = self.lstm(encoded, state)
        predictions = self.head(lstm_out)
        return predictions, state
