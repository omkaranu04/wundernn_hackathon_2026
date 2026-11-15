"""LSTM encoder + dilated (log-)sparse self-attention, at experiment 001's
full originally-intended scale (d_model=128).

User's request: "like experiment 1, apply attention, but instead of the
full attention window of 512, try sparse attention for (1,2,4,8,...,2048)
and increase the LSTM layers to 2 (as against experiment 1's actually-
trained 1-layer config), rest all stays the same in the architecture."
"Rest stays the same" is taken from `model.py`'s own class defaults
(d_model=128, n_heads=4, ff_multiplier=4, dropout=0.1) - the originally-
intended design that was never actually trained at full scale because it
measured ~7.8x over the latency budget in PyTorch eager mode (see
experiment 001's train.py). Per explicit user instruction, this experiment
builds/trains that scale anyway, with the attention mechanism swapped for
sparse/dilated (see below) and lstm_layers=2 instead of 001's actually-
scored lstm_layers=1.

Architecture:
    features (T, 112)
        -> Linear input projection (112 -> 128)
        -> 2-layer LSTM
        -> LayerNorm -> dilated sparse self-attention (offsets 1,2,4,...,2048)
        -> residual add
        -> LayerNorm -> feed-forward
        -> residual add
        -> regression head (Linear 128 -> 2)

Same overall block shape as 001's `LSTMAttentionModel`, but no grouped/
nonlinear input encoder (001 never had one - raw 112 features go straight
into the input projection) and the attention mechanism is
`DilatedSparseSelfAttention` (see experiment 016's model.py for the design
rationale and validation story, copied verbatim here - offsets, buffer
size, and the off-by-one it avoids are all identical).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

LSTMState = Tuple[torch.Tensor, torch.Tensor]

OFFSETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
BUFFER_SIZE = max(OFFSETS) + 1  # 2049 - see experiment 016's model.py for why +1 is required


class DilatedSparseSelfAttention(nn.Module):
    """Verbatim copy of experiment 016's attention module - see that file
    for the full design rationale and numerical validation story."""

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
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

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

        K = torch.stack(gathered_k, dim=3)
        V = torch.stack(gathered_v, dim=3)
        mask = torch.stack(masks, dim=-1)

        scores = torch.einsum("bhtd,bhtsd->bhts", q, K) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, -1), float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhts,bhtsd->bhtd", attn, V)

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


class SparseAttentionLSTMModel(nn.Module):
    def __init__(
        self,
        n_features: int = 112,
        d_model: int = 128,
        lstm_layers: int = 2,
        n_heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        n_targets: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.lstm_layers = lstm_layers

        self.input_proj = nn.Linear(n_features, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=lstm_layers, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = DilatedSparseSelfAttention(d_model, n_heads, dropout=dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_model * ff_multiplier, dropout=dropout)
        self.head = nn.Linear(d_model, n_targets)

    def forward(
        self, features: torch.Tensor, lstm_state: Optional[LSTMState] = None
    ) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, n_features). Returns (predictions, lstm_state)."""
        x = self.input_proj(features)
        lstm_out, lstm_state = self.lstm(x, lstm_state)

        attended = self.attention(self.attn_norm(lstm_out))
        x = lstm_out + attended
        x = x + self.feed_forward(self.ff_norm(x))

        predictions = self.head(x)
        return predictions, lstm_state
