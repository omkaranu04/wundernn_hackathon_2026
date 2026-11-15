"""Nonlinear-expanded grouped-encoder + 1-layer LSTM + causal windowed
self-attention (window=512).

User's request: take experiment 012's exact input encoding (all 4 feature
groups expanded 5x via `[x, asinh, sin, cos, tanh]`) and output structure
(2-layer MLP head predicting (t0, t1) directly) unchanged, but replace 012's
plain 2-layer LSTM backbone with a 1-layer LSTM followed by experiment 001's
attention block (LayerNorm -> causal windowed self-attention, window=512 ->
residual -> LayerNorm -> feed-forward -> residual), the same "LSTM +
attention on top" pattern 001 introduced.

The attention mechanism itself (`CausalSelfAttention`, windowed/banded, not
sparse) is copied verbatim from experiment 001's model.py - see that file's
docstring for why it's windowed (not full O(T^2)) and the OOM story behind
the current chunked implementation (unfold() silently materialized a full
tensor, measured 41.9GB; this chunked version keeps peak memory at a few
hundred MB for the same shapes).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)

N_TRANSFORMS = 5  # x, asinh, sin, cos, tanh


def expand(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*5): [x, asinh(x), sin(x), cos(x), tanh(x)] concatenated."""
    return torch.cat([raw, torch.asinh(raw), torch.sin(raw), torch.cos(raw), torch.tanh(raw)], dim=-1)


class GroupEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention, always causal, bounded look-back window.
    Verbatim copy of experiment 001's `CausalSelfAttention` (window mode
    only - the unbounded full-attention path is dropped since this
    experiment always uses a finite window)."""

    def __init__(self, d_model: int, n_heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, d_model). Returns (batch, T, d_model)."""
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        out = self._windowed_causal_attention(q, k, v, self.window)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)

    def _windowed_causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
        """Block-recurrent banded causal attention - see experiment 001's
        model.py for the full explanation and the unfold()-OOM story."""
        batch, heads, seq_len, head_dim = q.shape
        chunk_len = min(window, seq_len)

        outputs = []
        cache_k = k.new_zeros(batch, heads, 0, head_dim)
        cache_v = v.new_zeros(batch, heads, 0, head_dim)

        for start in range(0, seq_len, chunk_len):
            end = min(start + chunk_len, seq_len)
            q_chunk = q[:, :, start:end]

            context_k = torch.cat([cache_k, k[:, :, start:end]], dim=2)
            context_v = torch.cat([cache_v, v[:, :, start:end]], dim=2)
            ctx_len = context_k.shape[2]
            cache_len = cache_k.shape[2]

            scores = torch.matmul(q_chunk, context_k.transpose(-2, -1)) / math.sqrt(head_dim)

            q_abs = start + torch.arange(end - start, device=q.device).unsqueeze(1)
            k_abs = (start - cache_len) + torch.arange(ctx_len, device=q.device).unsqueeze(0)
            distance = q_abs - k_abs
            invalid = (distance < 0) | (distance >= window)
            scores = scores.masked_fill(invalid.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = self.dropout(torch.softmax(scores, dim=-1))
            outputs.append(torch.matmul(attn, context_v))

            keep = min(window - 1, ctx_len)
            cache_k = context_k[:, :, ctx_len - keep:, :]
            cache_v = context_v[:, :, ctx_len - keep:, :]

        return torch.cat(outputs, dim=2)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonlinearWindowedAttentionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        price_dim: int = 24,
        volume_dim: int = 24,
        trade_dim: int = 12,
        n_heads: int = 4,
        attention_window: int = 512,
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
        self.attention = CausalSelfAttention(hidden_size, n_heads, window=attention_window, dropout=dropout)
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
