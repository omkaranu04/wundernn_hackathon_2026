"""Streaming (row-by-row) inference wrapper around `LSTMAttentionModel`.

`model.py`'s `forward()` processes a whole sequence/chunk at once, which is
efficient for training but is *not* the contract the competition uses at
submission time: the platform calls `predict()` once per row (see the
`DataPoint` contract in PROBLEM_STATEMENT.md / the starter pack's `utils.py`),
strictly left-to-right, with one state reset per new `seq_ix`. This module
reproduces the exact same computation incrementally, row by row, so streaming
predictions are numerically identical to what `model.forward()` would have
produced given the same prefix of a sequence.

State carried between calls:
  - the LSTM's (h, c) recurrent state - fixed-size, O(1) to update, standard.
  - a rolling cache of the last `attention_window` steps' *key/value
    projections* (not the raw hidden vectors - see below). Unlike the LSTM
    state, attention's look-back is not a fixed-size summary, so this is a
    genuine bounded-size buffer that fills up to `attention_window` entries
    and then evicts the oldest on each new step (a sliding window).

Why cache key/value projections rather than raw hidden vectors: each cached
entry's (key, value) is a fixed function of that step's own LSTM output
(via the shared `attn_norm` + `qkv` linear layer) and never changes once
computed - projecting it again on a later step would recompute the exact same
numbers. So each step computes its own (q, k, v) once, uses q immediately
against the cache, and pushes (k, v) onto the cache for future steps to use.
This makes each step's incremental cost O(window) (attend over the cached
window) instead of O(window * d_model^2) (re-projecting the whole window
every step).

Cache implementation: a preallocated circular tensor buffer of shape
(window, n_heads, head_dim), written in place, rather than a Python
`collections.deque` of per-step tensors rebuilt into a fresh stacked tensor
every call. An earlier version used the deque + `torch.stack(...)` approach -
correct, but measured at ~1.44 ms/step (attention_window=512), which
extrapolates to ~15 hours for a valid-set-scale test set against the
competition's 60-minute budget. The Python-level list comprehension and
re-allocation on every single row (up to 512 entries, every step) was the
actual cost, not the underlying attention math. Softmax attention is a
permutation-invariant weighted sum over the cached set - it doesn't care what
order the cache holds its entries in - so a circular buffer (no need to keep
anything in chronological order, just overwrite the oldest slot each step)
is both correct and avoids that overhead entirely. See `model.py`'s
"correctness check" note - this was verified against the deque version for
identical output before replacing it.

This file has no training/data-loading logic - see `train.py` / `dataset.py`
for that, and `solution.py` for the actual competition submission entrypoint
that wraps this class.
"""
from __future__ import annotations

import torch

from model import LSTMAttentionModel, build_lstm_cells


class StreamingPredictor:
    def __init__(self, model: LSTMAttentionModel):
        model.eval()
        self.model = model
        self.d_model = model.d_model
        self.n_heads = model.attention.n_heads
        self.head_dim = model.attention.head_dim
        self.window = model.attention.window
        if self.window is None:
            raise ValueError(
                "Streaming inference requires a finite attention window - "
                "unbounded causal attention would need to cache the entire "
                "sequence history (up to 19,999 steps), which defeats the "
                "point of a bounded, fixed-cost-per-step online predictor."
            )

        # `nn.LSTM` is built (and cuDNN/MKL-optimized) for processing whole
        # sequences in one call; invoking it with seq_len=1 on every row pays
        # its fixed per-call overhead 20,000 times over. `nn.LSTMCell` is
        # built for exactly this single-step use case - measured ~2.7x
        # faster per call for identical math (see EXPERIMENTS.md). Training
        # (`model.py::forward`) still uses batched `nn.LSTM` - that direction
        # benefits from processing the whole sequence in one cuDNN/MKL call,
        # the opposite of the streaming case. See `model.py::build_lstm_cells`.
        self.lstm_cells = build_lstm_cells(model.lstm)

        self._reset_state()

    def _reset_state(self) -> None:
        num_layers = len(self.lstm_cells)
        self.lstm_h = [torch.zeros(1, self.d_model) for _ in range(num_layers)]
        self.lstm_c = [torch.zeros(1, self.d_model) for _ in range(num_layers)]
        # circular buffer: cache_k[i]/cache_v[i] holds the key/value projection
        # from some past step; cache_pos is the next slot to overwrite. Order
        # within the buffer is irrelevant (see module docstring) - only the
        # *set* of the last `window` steps' entries matters.
        self.cache_k = torch.zeros(self.window, self.n_heads, self.head_dim)
        self.cache_v = torch.zeros(self.window, self.n_heads, self.head_dim)
        self.cache_len = 0   # number of valid entries so far (caps at self.window)
        self.cache_pos = 0   # next slot to write (wraps around)

    def reset(self) -> None:
        """Call whenever `seq_ix` changes - see `solution.py`."""
        self._reset_state()

    @torch.no_grad()
    def step(self, features: torch.Tensor) -> torch.Tensor:
        """features: (n_features,) float32 tensor for one row. Returns a (2,) prediction tensor.

        Must be called exactly once per row, strictly in sequence order,
        within one sequence - matches `solution.py`'s `predict()` contract
        (and, in turn, the competition's).
        """
        x = self.model.input_proj(features.view(1, -1))  # (1, d_model)
        layer_input = x
        for i, cell in enumerate(self.lstm_cells):
            self.lstm_h[i], self.lstm_c[i] = cell(layer_input, (self.lstm_h[i], self.lstm_c[i]))
            layer_input = self.lstm_h[i]
        lstm_out = layer_input.view(self.d_model)  # (d_model,) - final layer's hidden state

        normed = self.model.attn_norm(lstm_out)
        qkv = self.model.attention.qkv(normed)  # (3*d_model,)
        q, k, v = qkv.split(self.d_model)
        q = q.view(self.n_heads, self.head_dim)
        k = k.view(self.n_heads, self.head_dim)
        v = v.view(self.n_heads, self.head_dim)

        self.cache_k[self.cache_pos] = k  # includes the current step - matches training's
        self.cache_v[self.cache_pos] = v  # "rightmost key of the window is always position i"
        self.cache_len = min(self.cache_len + 1, self.window)
        self.cache_pos = (self.cache_pos + 1) % self.window

        active_k = self.cache_k if self.cache_len == self.window else self.cache_k[: self.cache_len]
        active_v = self.cache_v if self.cache_len == self.window else self.cache_v[: self.cache_len]

        scores = torch.einsum("hd,chd->ch", q, active_k) / (self.head_dim ** 0.5)  # (cache_len, n_heads)
        attn = torch.softmax(scores, dim=0)  # softmax over the cached (causal, windowed) key set
        attended = torch.einsum("ch,chd->hd", attn, active_v).reshape(self.d_model)

        attended = self.model.attention.out_proj(attended)
        x = lstm_out + attended  # residual - matches model.py's `lstm_out + attended`

        x = x + self.model.feed_forward(self.model.ff_norm(x))
        prediction = self.model.head(x)
        return prediction
