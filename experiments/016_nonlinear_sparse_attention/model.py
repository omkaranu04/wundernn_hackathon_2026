"""Nonlinear-expanded grouped-encoder + 2-layer LSTM + dilated (log-)sparse
self-attention.

User's request: same as experiment 015 (012's input encoding + output
structure unchanged), but with 2 LSTM layers (not 1) and **sparse**
attention instead of a fixed window - specifically, dilated/strided sparse
attention at offsets (1, 2, 4, 8, 16, ..., 2048): each query attends to a
SPARSE set of past positions at those geometric strides (plus itself),
instead of every position within a contiguous window. This is the
"LogSparse"/dilated attention pattern used in some long-sequence time-series
transformer work: O(len(OFFSETS)) attended positions per query - 13 fixed
positions here - instead of O(window), giving cheap access to very
long-range history (up to 2048 steps back) without attending to everything
in between.

Validated (see EXPERIMENTS.md) against an exact streaming recursion before
building around it: batched (shifted-K/V gather) and streaming (circular
buffer, size = max(OFFSETS)+1 - NOT max(OFFSETS), a genuine off-by-one that
was caught and fixed during validation, see below) agree to float32
precision (~5e-7 max diff on a synthetic test). Unlike experiment 013's
EWMA, this needs no train/export approximation gap: gathering exact values
at fixed offsets is trivially exact in both batched and streaming form.

Buffer-size subtlety (read before changing OFFSETS): with a circular buffer
of size exactly max(OFFSETS), the current step's write clobbers the exact
slot the max-offset lookup needs at that same step (step t's write goes to
slot t%N; offset=N lookup at step t wants slot (t-N)%N = t%N - the SAME
slot, already overwritten). Buffer size must be max(OFFSETS)+1.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)

N_TRANSFORMS = 5  # x, asinh, sin, cos, tanh

OFFSETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
BUFFER_SIZE = max(OFFSETS) + 1  # 2049 - see module docstring for why +1 is required


def expand(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*5): [x, asinh(x), sin(x), cos(x), tanh(x)] concatenated."""
    return torch.cat([raw, torch.asinh(raw), torch.sin(raw), torch.cos(raw), torch.tanh(raw)], dim=-1)


class GroupEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedSparseSelfAttention(nn.Module):
    """Multi-head self-attention where each query attends only to keys at
    the fixed relative offsets in `offsets` (causal: offset=0 is self,
    larger offsets look further into the past). Batched forward is used for
    training; see export_onnx.py for the exact-equivalent streaming form."""

    def __init__(self, d_model: int, n_heads: int, offsets=OFFSETS, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.offsets = offsets

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, d_model). Returns (batch, T, d_model)."""
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, heads, T, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        gathered_k, gathered_v, masks = [], [], []
        for o in self.offsets:
            if o == 0:
                gathered_k.append(k)
                gathered_v.append(v)
                masks.append(torch.ones(seq_len, dtype=torch.bool, device=x.device))
            else:
                shifted_k = F.pad(k, (0, 0, o, 0))[..., :seq_len, :]
                shifted_v = F.pad(v, (0, 0, o, 0))[..., :seq_len, :]
                gathered_k.append(shifted_k)
                gathered_v.append(shifted_v)
                masks.append(torch.arange(seq_len, device=x.device) >= o)

        K = torch.stack(gathered_k, dim=3)  # (batch, heads, T, n_offsets, head_dim)
        V = torch.stack(gathered_v, dim=3)
        mask = torch.stack(masks, dim=-1)   # (T, n_offsets)

        scores = torch.einsum("bhtd,bhtsd->bhts", q, K) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, -1), float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhts,bhtsd->bhtd", attn, V)  # (batch, heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonlinearSparseAttentionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        price_dim: int = 24,
        volume_dim: int = 24,
        trade_dim: int = 12,
        n_heads: int = 4,
        ff_multiplier: int = 4,
        n_targets: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.price_encoder = GroupEncoder(44 * N_TRANSFORMS, price_dim, dropout)
        self.volume_encoder = GroupEncoder(44 * N_TRANSFORMS, volume_dim, dropout)
        self.trade_encoder = GroupEncoder(16 * N_TRANSFORMS, trade_dim, dropout)

        additional_dim = (ADDITIONAL[1] - ADDITIONAL[0]) * N_TRANSFORMS
        lstm_input_size = price_dim + volume_dim + trade_dim + additional_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attention = DilatedSparseSelfAttention(hidden_size, n_heads, dropout=dropout)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, hidden_size * ff_multiplier, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_targets),
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
            self.price_encoder(expand(price)),
            self.volume_encoder(expand(volume)),
            self.trade_encoder(expand(trade)),
            expand(additional),
        ], dim=-1)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, 112). Returns (predictions, state)."""
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)

        attended = self.attention(self.attn_norm(lstm_out))
        x = lstm_out + attended
        x = x + self.feed_forward(self.ff_norm(x))

        predictions = self.head(x)
        return predictions, state
