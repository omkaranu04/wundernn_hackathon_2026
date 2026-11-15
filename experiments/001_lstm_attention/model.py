"""LSTM encoder + masked (causal) self-attention model for the Connectome challenge.

Architecture
------------
    features (T, 112)
        -> input projection (Linear 112 -> d_model)
        -> multi-layer LSTM                          (recurrent state: h, c)
        -> LayerNorm -> causal self-attention          (attends only to steps <= current)
        -> residual add
        -> LayerNorm -> feed-forward
        -> residual add
        -> regression head (Linear d_model -> 2)

The LSTM already only sees past inputs by construction (it's recurrent,
left-to-right). The self-attention block on top lets the model directly look
back at any earlier step's LSTM output (not just what's been compressed into
the current recurrent state) - the causal mask is what stops it from ever
attending to a future position, which would leak information and violate the
competition's rules (a prediction may only use the current and earlier rows).

Windowed attention, not full O(T^2)
------------------------------------
Sequences are 20,000 steps long. A naive causal-attention implementation
materializes a full (T, T) score matrix per head - for T=20,000 that's 4*10^8
entries *per head, per sequence*, which is not remotely feasible on the
16 GB / 1-vCPU budget this eventually has to run inside (see BASELINE.md /
PROBLEM_STATEMENT.md in the repo root). `CausalSelfAttention` instead
implements a *local windowed causal attention*: each step attends only to the
last `window` steps (inclusive of itself), computed via an `unfold`-based
banded attention that costs O(T * window) instead of O(T^2), and never
materializes the full score matrix.

Train/inference parity
-----------------------
This module defines the "process a whole sequence/chunk at once" forward pass
used for training (see `train.py`, `dataset.py`). The competition instead
calls a step-by-step `predict()` once per row (see `inference.py`,
`solution.py`). For the two to produce identical numbers, `inference.py`
caches each step's *key/value projections* (not just the LSTM state) and
replicates the exact same windowed attention computation incrementally - see
the module docstring there. Training deliberately uses one full 20,000-step
sequence per sample (see `dataset.py`) rather than shorter chunks, precisely
so the windowed look-back is always measured from the sequence's true start,
matching what streaming inference sees.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def causal_mask(size: int, device=None) -> torch.Tensor:
    """Additive attention mask of shape (size, size): 0 where key <= query
    (allowed), -inf where key > query (future, blocked). Only used by the
    unbounded full-attention path - see `CausalSelfAttention.window`.
    """
    mask = torch.full((size, size), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention, always causal, with a bounded look-back window.

    window=None: full causal attention (attends to the entire past). Only
    intended for short chunks/tests - O(T^2) memory, infeasible at T=20,000.
    Guarded with an assertion below.

    window=W: attend only to the last W steps (inclusive of the current one).
    Implemented without ever building a (T, T) matrix - see
    `_windowed_causal_attention`.
    """

    _MAX_SEQ_FOR_FULL_ATTENTION = 4096  # sanity guard, see window=None branch

    def __init__(self, d_model: int, n_heads: int, window: Optional[int] = None, dropout: float = 0.0):
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
        qkv = self.qkv(x)  # (batch, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, heads, T, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if self.window is None:
            if seq_len > self._MAX_SEQ_FOR_FULL_ATTENTION:
                raise ValueError(
                    f"full (unwindowed) causal attention requested on a sequence of length "
                    f"{seq_len} > {self._MAX_SEQ_FOR_FULL_ATTENTION} - this would need a "
                    f"(batch, heads, {seq_len}, {seq_len}) score matrix. Set `window` to a "
                    f"bounded value instead."
                )
            out = self._full_causal_attention(q, k, v)
        else:
            out = self._windowed_causal_attention(q, k, v, self.window)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)

    def _full_causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = q.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, T, T)
        scores = scores + causal_mask(seq_len, device=q.device)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, v)

    def _windowed_causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
        """Banded causal attention: query i attends to keys in [i-window+1, i].

        Block-recurrent implementation: the sequence is processed in chunks
        of `window` steps, carrying forward only the trailing `window - 1`
        keys/values as a small cache between chunks (the same idea as
        `inference.py`'s streaming key/value cache, just batched over a chunk
        at a time instead of one row at a time). Each chunk's score tensor is
        only (batch, heads, chunk_len, <= 2*window-1) - small and bounded,
        reused/freed chunk to chunk.

        An earlier version used `k.unfold(...)` to build a (batch, heads, T,
        window, head_dim) tensor in one shot, reasoning that it was
        O(T*window) rather than O(T^2). That FLOP-count reasoning was correct
        but the *memory* reasoning was not: unfold's overlapping-window
        strides can't be reshaped into the contiguous layout `einsum`'s
        batched matmul needs, so PyTorch silently materializes the full
        expanded tensor anyway. Measured peak: ~41.9 GB for batch=4,
        T=20,000, window=512, d_model=128 - confirmed by an isolated forward
        pass and the OOM kill of the first real training attempt (see
        EXPERIMENTS.md). This version was verified to produce numerically
        identical output to that one (see the correctness check referenced
        in the experiment README) while keeping peak memory at a few hundred
        MB for the same shapes.
        """
        batch, heads, seq_len, head_dim = q.shape
        chunk_len = min(window, seq_len)

        outputs = []
        cache_k = k.new_zeros(batch, heads, 0, head_dim)
        cache_v = v.new_zeros(batch, heads, 0, head_dim)

        for start in range(0, seq_len, chunk_len):
            end = min(start + chunk_len, seq_len)
            q_chunk = q[:, :, start:end]  # (batch, heads, chunk, head_dim)

            context_k = torch.cat([cache_k, k[:, :, start:end]], dim=2)  # (batch, heads, ctx, head_dim)
            context_v = torch.cat([cache_v, v[:, :, start:end]], dim=2)
            ctx_len = context_k.shape[2]
            cache_len = cache_k.shape[2]

            scores = torch.matmul(q_chunk, context_k.transpose(-2, -1)) / math.sqrt(head_dim)  # (batch, heads, chunk, ctx)

            # absolute position of query i (row) and context key j (column);
            # allowed iff 0 <= q_abs - k_abs < window (causal + windowed).
            q_abs = start + torch.arange(end - start, device=q.device).unsqueeze(1)          # (chunk, 1)
            k_abs = (start - cache_len) + torch.arange(ctx_len, device=q.device).unsqueeze(0)  # (1, ctx)
            distance = q_abs - k_abs
            invalid = (distance < 0) | (distance >= window)
            scores = scores.masked_fill(invalid.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = self.dropout(torch.softmax(scores, dim=-1))
            outputs.append(torch.matmul(attn, context_v))  # (batch, heads, chunk, head_dim)

            # carry forward only the trailing (window - 1) positions for the next chunk
            keep = min(window - 1, ctx_len)
            cache_k = context_k[:, :, ctx_len - keep:, :]
            cache_v = context_v[:, :, ctx_len - keep:, :]

        return torch.cat(outputs, dim=2)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (num_layers, batch, d_model)


def build_lstm_cells(lstm: nn.LSTM) -> nn.ModuleList:
    """Build one `nn.LSTMCell` per layer of a trained `nn.LSTM`, with weights
    copied over so the result is numerically identical for single-step calls.

    `nn.LSTM` is optimized for processing a whole sequence in one call;
    invoking it with seq_len=1 repeatedly (as any row-by-row inference path
    must) pays its fixed per-call overhead every single time - `nn.LSTMCell`
    is built for exactly the single-step case and is markedly faster for it
    (measured ~2.7x per call, same math, same weights - see EXPERIMENTS.md).
    `nn.LSTMCell` uses the same per-layer weight shapes and gate order as
    `nn.LSTM`, so copying is a direct 1:1 parameter mapping, no math changes.

    Used by both `inference.py::StreamingPredictor` (PyTorch-eager streaming)
    and `export_onnx.py` (ONNX export) so there is one implementation of this
    weight-copy logic to keep in sync with `nn.LSTM`'s parameter layout.
    """
    cells = nn.ModuleList(
        [nn.LSTMCell(lstm.input_size if layer == 0 else lstm.hidden_size, lstm.hidden_size)
         for layer in range(lstm.num_layers)]
    )
    with torch.no_grad():
        for layer, cell in enumerate(cells):
            cell.weight_ih.copy_(getattr(lstm, f"weight_ih_l{layer}"))
            cell.weight_hh.copy_(getattr(lstm, f"weight_hh_l{layer}"))
            cell.bias_ih.copy_(getattr(lstm, f"bias_ih_l{layer}"))
            cell.bias_hh.copy_(getattr(lstm, f"bias_hh_l{layer}"))
    cells.eval()
    return cells


class LSTMAttentionModel(nn.Module):
    """LSTM encoder followed by one causal windowed self-attention block.

    Config is intentionally modest: the competition scores 37M+ individual
    inference steps against a hard 60-minute wall-clock limit on 1 vCPU (see
    PROBLEM_STATEMENT.md), so a wider/deeper model trades accuracy for step
    latency in a way that can make a submission time out entirely.
    """

    def __init__(
        self,
        n_features: int = 112,
        d_model: int = 128,
        lstm_layers: int = 2,
        n_heads: int = 4,
        attention_window: Optional[int] = 512,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        n_targets: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, window=attention_window, dropout=dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_model * ff_multiplier, dropout=dropout)
        self.head = nn.Linear(d_model, n_targets)

    def forward(
        self, features: torch.Tensor, lstm_state: Optional[LSTMState] = None
    ) -> Tuple[torch.Tensor, LSTMState]:
        """features: (batch, T, n_features).

        Returns (predictions, lstm_state):
          predictions: (batch, T, n_targets)
          lstm_state:  (h, c) after processing this chunk - pass back in to
                       continue the same sequence in a later call. Note that
                       only the *LSTM* state is fixed-size/carryable this way;
                       the attention block's windowed look-back is computed
                       fresh from the given `features` each call, so this
                       `forward()` is only exactly correct for a chunk that
                       starts at the true beginning of a sequence (see
                       `dataset.py`'s "one sample = one full sequence" choice)
                       or when the caller doesn't care about attention seeing
                       the true full window near the chunk's start.
        """
        x = self.input_proj(features)
        lstm_out, lstm_state = self.lstm(x, lstm_state)

        attended = self.attention(self.attn_norm(lstm_out))
        x = lstm_out + attended

        x = x + self.feed_forward(self.ff_norm(x))

        predictions = self.head(x)
        return predictions, lstm_state
