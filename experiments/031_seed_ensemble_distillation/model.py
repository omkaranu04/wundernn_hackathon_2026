"""Lean cos-expanded grouped-encoder LSTM (experiment 029 = 028's model + an
OPTIONAL `bidirectional` switch).

New in 029: `bidirectional=True` makes the LSTM see the FUTURE as well as the
past. Such a model can never be shipped (inference is one row at a time); it
exists only as a "hindsight teacher" to measure how much better a model could
do with future context, i.e. the ceiling for distillation. export_onnx.py
refuses to export it.

--- 028 docstring follows ---

Lean cos-expanded grouped-encoder LSTM (experiment 028 = 027's model + an
OPTIONAL auxiliary head).

New in 028: `aux_scored=True` adds `aux_head = Linear(hidden, 1)` on the LSTM
output that predicts the logit of "this row is scored". It is a TRAINING-ONLY
auxiliary task (forces the recurrent state to represent the scored-row regime);
the ONNX export never touches it, so the shipped graph and latency are
identical to 027's. `forward(..., return_aux=True)` returns the logits.

--- 027 docstring follows ---

Lean cos-expanded grouped-encoder LSTM (experiment 027).

A deliberately slimmed variant of experiment 012:
- INPUTS (68 of the 112 features): instrument-0 price (22) + instrument-0
  volume (22) + ALL trade features dp/dv of both instruments (16) + the 8
  `additional` features. Instrument 1's price and volume (44 features) are
  DROPPED (experiments 021/022: effectively unused).
- EXPANSION: each kept group is expanded to `[x, cos(x)]` (2 channels)
  instead of 012's 5 - experiment 021's channel test found `x` and `cos(x)`
  are the channels 012 actually relies on (asinh/sin/tanh were redundant).
- ENCODERS: grouped MLPs exactly as 012 (Linear + ReLU + Dropout):
  price 44->24, volume 44->24, trade 32->12. The `additional` group is
  expanded to 16 dims and passed straight to the LSTM, as in 012/004 (an
  interpretation choice: the spec said "grouped MLPs" and 012's additional
  group had no MLP; a 16->k MLP for additional is a one-line change).
- RECURRENT CORE: 2-layer LSTM, hidden 128, then 012's head
  Linear(128,64)-ReLU-Dropout-Linear(64,2) predicting (t0, t1) directly.
Per-step vector: 24 + 24 + 12 + 16 = 76 dims (012: 100).
The training loss for this experiment's main arm is pure global weighted
Pearson - see train.py / README.md.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

LSTMState = Tuple[torch.Tensor, torch.Tensor]

# (start, end) slices into the 112-dim raw feature vector, per shared/data.py's FEATURE_COLUMNS order
I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)

N_TRANSFORMS = 2  # x, cos


def expand(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*2): [x, cos(x)] concatenated."""
    return torch.cat([raw, torch.cos(raw)], dim=-1)


class GroupEncoder(nn.Module):
    """Single Linear + ReLU - same shape as 004's (see its revision history for why no hidden layer)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeanCosGroupedEncoderLSTMModel(nn.Module):
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
        aux_scored: bool = False,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.price_encoder = GroupEncoder(22 * N_TRANSFORMS, price_dim, dropout)   # i0 price only
        self.volume_encoder = GroupEncoder(22 * N_TRANSFORMS, volume_dim, dropout)  # i0 volume only
        self.trade_encoder = GroupEncoder(16 * N_TRANSFORMS, trade_dim, dropout)

        additional_dim = (ADDITIONAL[1] - ADDITIONAL[0]) * N_TRANSFORMS
        lstm_input_size = price_dim + volume_dim + trade_dim + additional_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(out_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_targets),
        )
        self.aux_head = nn.Linear(out_dim, 1) if aux_scored else None

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., 112) -> (..., lstm_input_size). Shared by forward() and streaming export."""
        price_raw = features[..., I0_PRICE[0]:I0_PRICE[1]]      # instrument 0 only
        volume_raw = features[..., I0_VOLUME[0]:I0_VOLUME[1]]   # instrument 0 only
        trade_raw = torch.cat([
            features[..., I0_DP[0]:I0_DP[1]], features[..., I0_DV[0]:I0_DV[1]],
            features[..., I1_DP[0]:I1_DP[1]], features[..., I1_DV[0]:I1_DV[1]],
        ], dim=-1)
        additional_raw = features[..., ADDITIONAL[0]:ADDITIONAL[1]]

        return torch.cat([
            self.price_encoder(expand(price_raw)),
            self.volume_encoder(expand(volume_raw)),
            self.trade_encoder(expand(trade_raw)),
            expand(additional_raw),
        ], dim=-1)

    def forward(self, features: torch.Tensor, state: Optional[LSTMState] = None, return_aux: bool = False):
        """features: (batch, T, 112). Returns (predictions, state) or, with
        return_aux=True, (predictions, state, aux_logits or None)."""
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        predictions = self.head(lstm_out)
        if return_aux:
            aux = self.aux_head(lstm_out).squeeze(-1) if self.aux_head is not None else None
            return predictions, state, aux
        return predictions, state


# ---------------------------------------------------------------------------
# Round-4 teacher diversity (2026-10-25): experiment 012's architecture
# (5-way nonlinear expansion [x, asinh, sin, cos, tanh] of ALL 112 inputs, both
# instruments merged per group) with an LSTM (012/026) or a GRU (024) core, plus
# the optional training-only auxiliary head so train.py can use them unchanged.
# Teachers only: never exported (students stay the lean model).
# ---------------------------------------------------------------------------
NL_TRANSFORMS = 5


def expand_nl(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*5): [x, asinh(x), sin(x), cos(x), tanh(x)]."""
    return torch.cat([raw, torch.asinh(raw), torch.sin(raw), torch.cos(raw), torch.tanh(raw)], dim=-1)


class NonlinearGroupedEncoderRNNModel(nn.Module):
    def __init__(self, rnn_type: str = "lstm", hidden_size: int = 128, num_layers: int = 2, price_dim: int = 24,
                 volume_dim: int = 24, trade_dim: int = 12, head_hidden: int = 64, n_targets: int = 2,
                 dropout: float = 0.1, aux_scored: bool = False, bidirectional: bool = False):
        super().__init__()
        assert rnn_type in ("lstm", "gru") and not bidirectional
        self.rnn_type, self.hidden_size, self.num_layers, self.bidirectional = rnn_type, hidden_size, num_layers, False
        self.price_encoder = GroupEncoder(44 * NL_TRANSFORMS, price_dim, dropout)
        self.volume_encoder = GroupEncoder(44 * NL_TRANSFORMS, volume_dim, dropout)
        self.trade_encoder = GroupEncoder(16 * NL_TRANSFORMS, trade_dim, dropout)
        in_size = price_dim + volume_dim + trade_dim + (ADDITIONAL[1] - ADDITIONAL[0]) * NL_TRANSFORMS
        rnn = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, n_targets))
        self.aux_head = nn.Linear(hidden_size, 1) if aux_scored else None

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        price = torch.cat([features[..., I0_PRICE[0]:I0_PRICE[1]], features[..., I1_PRICE[0]:I1_PRICE[1]]], dim=-1)
        volume = torch.cat([features[..., I0_VOLUME[0]:I0_VOLUME[1]], features[..., I1_VOLUME[0]:I1_VOLUME[1]]], dim=-1)
        trade = torch.cat([features[..., I0_DP[0]:I0_DP[1]], features[..., I0_DV[0]:I0_DV[1]],
                           features[..., I1_DP[0]:I1_DP[1]], features[..., I1_DV[0]:I1_DV[1]]], dim=-1)
        add = features[..., ADDITIONAL[0]:ADDITIONAL[1]]
        return torch.cat([self.price_encoder(expand_nl(price)), self.volume_encoder(expand_nl(volume)),
                          self.trade_encoder(expand_nl(trade)), expand_nl(add)], dim=-1)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False):
        out, state = self.rnn(self.encode(features), state)
        predictions = self.head(out)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


# ---------------------------------------------------------------------------
# Look-ahead teacher (2026-10-30; a rival's +0.0012 public step): the lean causal
# model plus a branch that reads ONLY the next K rows (t..t+K) through gated
# dilated convolutions, added to the LSTM output through a zero-initialised
# projection (so training starts from the plain causal model). Its predictions
# are a "denoised" label between the causal teachers' (~0.69) and the truth: the
# 029 bidirectional teacher saw the WHOLE future (0.997) and was useless as a
# label source; a short window is the point. Teacher only: never exported.
# ---------------------------------------------------------------------------
LOOKAHEAD_DILATIONS = {6: (1, 2, 3), 12: (1, 2, 4, 5), 25: (1, 2, 4, 8, 10), 50: (1, 2, 4, 8, 16, 19)}  # receptive field 1 + sum = K + 1 rows: t..t+K


class GatedFutureConv(nn.Module):
    """Gated dilated conv (kernel 2) that looks FORWARD: the output at t uses t and t+d only."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.d = dilation
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size=2, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        a, b = self.conv(F.pad(x, (0, self.d))).chunk(2, dim=1)  # right zero-padding: past the end = 0
        return x + torch.tanh(a) * torch.sigmoid(b)


class LookaheadLeanModel(LeanCosGroupedEncoderLSTMModel):
    def __init__(self, lookahead: int = 25, future_dim: int = 64, **kw):
        super().__init__(**kw)
        self.lookahead = lookahead
        enc_dim = self.lstm.input_size
        self.future_in = nn.Linear(2 * enc_dim, future_dim)  # [encoded x_t, encoded x_t - encoded x_{t-1}]
        self.future_convs = nn.ModuleList([GatedFutureConv(future_dim, d) for d in LOOKAHEAD_DILATIONS[lookahead]])
        self.future_out = nn.Linear(future_dim, self.hidden_size)
        nn.init.zeros_(self.future_out.weight); nn.init.zeros_(self.future_out.bias)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False):
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        diff = torch.cat([torch.zeros_like(encoded[:, :1]), encoded[:, 1:] - encoded[:, :-1]], dim=1)
        f = self.future_in(torch.cat([encoded, diff], dim=-1)).transpose(1, 2)
        for conv in self.future_convs:
            f = conv(f)
        out = lstm_out + self.future_out(f.transpose(1, 2))
        predictions = self.head(out)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


class EventLookaheadModel(LeanCosGroupedEncoderLSTMModel):
    """2026-11-07 (039): look-ahead TEACHER in EVENT time. Targets only change on rows where an i0 price column changes,
    and (sunArise notes) the reference price settles within ~8-16 quote changes. This teacher sees the encoded book at
    each of the next `n_events` i0-price-change rows (as change vs row t, plus the row gap), ignoring events more than
    `cap` rows ahead: ~12 rows on fine tick units, up to `cap` on the coarse unit. Teacher only: never exported."""

    def __init__(self, n_events: int = 6, cap: int = 50, future_dim: int = 64, **kw):
        super().__init__(**kw)
        self.n_events, self.cap = n_events, cap
        enc_dim = self.lstm.input_size
        self.future_mlp = nn.Sequential(nn.Linear(n_events * (enc_dim + 2), future_dim), nn.ReLU(), nn.Linear(future_dim, future_dim), nn.ReLU())
        self.future_out = nn.Linear(future_dim, self.hidden_size)
        nn.init.zeros_(self.future_out.weight); nn.init.zeros_(self.future_out.bias)

    def event_index(self, features: torch.Tensor) -> torch.Tensor:
        """(B, T, n_events) row index of the k-th i0 price change after t (T = none)."""
        B, T = features.shape[:2]
        price = features[..., I0_PRICE[0]:I0_PRICE[1]]
        chg = torch.zeros(B, T, dtype=torch.bool, device=features.device)
        chg[:, 1:] = (price[:, 1:] != price[:, :-1]).any(-1)
        ar = torch.arange(T, device=features.device).expand(B, T)
        idx = torch.where(chg, ar, torch.full_like(ar, T))
        nxt_from = torch.flip(torch.cummin(torch.flip(idx, [1]), dim=1).values, [1])      # first change at >= t
        nxt1 = torch.cat([nxt_from[:, 1:], torch.full((B, 1), T, device=features.device, dtype=nxt_from.dtype)], 1)  # > t
        nxt1_ext = torch.cat([nxt1, torch.full((B, 1), T, device=features.device, dtype=nxt1.dtype)], 1)  # index T -> T
        out = [nxt1]
        for _ in range(self.n_events - 1):
            out.append(torch.gather(nxt1_ext, 1, out[-1]))
        return torch.stack(out, -1)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False):
        encoded = self.encode(features)
        lstm_out, state = self.lstm(encoded, state)
        B, T, E = encoded.shape
        ev = self.event_index(features)                                                  # (B, T, N)
        t = torch.arange(T, device=features.device).view(1, T, 1)
        gap = (ev - t).float()
        valid = (ev < T) & (gap <= self.cap)
        enc_ext = torch.cat([encoded, torch.zeros(B, 1, E, device=encoded.device, dtype=encoded.dtype)], 1)
        g = torch.gather(enc_ext, 1, ev.reshape(B, T * self.n_events, 1).expand(-1, -1, E)).view(B, T, self.n_events, E)
        vf = valid.unsqueeze(-1).to(encoded.dtype)
        fut = torch.cat([(g - encoded.unsqueeze(2)) * vf, vf, (gap / self.cap).clamp(max=1.0).unsqueeze(-1) * vf], -1)
        out = lstm_out + self.future_out(self.future_mlp(fut.reshape(B, T, -1)))
        predictions = self.head(out)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


class LeanGRUModel(LeanCosGroupedEncoderLSTMModel):
    """2026-11-02: the lean encoder + head with a GRU core instead of the LSTM - a structurally different ensemble
    member (the rival: different structures +0.002, seeds of one structure +0.0003). Streams with ONE state tensor."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lstm = nn.GRU(input_size=self.lstm.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)


class LeanXModel(LeanCosGroupedEncoderLSTMModel):
    """2026-11-02 (push for the top 8): the lean LSTM student with three switchable additions, each cheap at inference:
      use_i1    - instrument 1's price + volume back in (two more [x, cos x] group encoders, like i0's)
      res_mlp   - a residual MLP stage after the LSTM: h + W2 relu(W1 h) (the rival: "a second nonlinear stage")
      pos_input - position in the sequence: [s, sin(2 pi s), cos(2 pi s)], s = step / 20000 (step from the caller)
    LSTM input order: [price, volume, trade, (i1 price, i1 volume), additional, (pos)]."""

    def __init__(self, use_i1: bool = False, res_mlp: bool = False, pos_input: bool = False, **kw):
        super().__init__(**kw)
        self.use_i1, self.use_res, self.use_pos = use_i1, res_mlp, pos_input
        dropout = kw.get("dropout", 0.1)
        if use_i1:
            self.i1_price_encoder = GroupEncoder(22 * N_TRANSFORMS, kw.get("price_dim", 24), dropout)
            self.i1_volume_encoder = GroupEncoder(22 * N_TRANSFORMS, kw.get("volume_dim", 24), dropout)
        in_size = self.lstm.input_size + (2 * 0 if not use_i1 else kw.get("price_dim", 24) + kw.get("volume_dim", 24)) + (3 if pos_input else 0)
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        if res_mlp:
            self.res = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size))
            nn.init.zeros_(self.res[2].weight); nn.init.zeros_(self.res[2].bias)

    def encode_x(self, features: torch.Tensor, step0: int = 0) -> torch.Tensor:
        base = self.encode(features)  # [price, volume, trade, additional]
        parts = [base[..., :-16]]
        if self.use_i1:
            parts += [self.i1_price_encoder(expand(features[..., I1_PRICE[0]:I1_PRICE[1]])),
                      self.i1_volume_encoder(expand(features[..., I1_VOLUME[0]:I1_VOLUME[1]]))]
        parts.append(base[..., -16:])
        if self.use_pos:
            T = features.shape[-2]
            s = (torch.arange(T, device=features.device, dtype=features.dtype) + step0) / 20000.0
            pos = torch.stack([s, torch.sin(2 * torch.pi * s), torch.cos(2 * torch.pi * s)], dim=-1)
            parts.append(pos.expand(*features.shape[:-1], 3))
        return torch.cat(parts, dim=-1)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False, step0: int = 0):
        out, state = self.lstm(self.encode_x(features, step0), state)
        if self.use_res:
            out = out + self.res(out)
        predictions = self.head(out)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


class LeanDistModel(LeanCosGroupedEncoderLSTMModel):
    """2026-11-03 (032): the lean LSTM student with a DISTRIBUTIONAL head - per target, logits over n_bins values
    c_b evenly spaced on [-2, 2] (the metric's clip range; the targets live on a tick lattice, mostly 0 / +-1 tick on
    the large unit). Point prediction = the |y|-weighted mean  sum p|c|c / sum p|c|, which is exactly the value that
    maximises a |y|-weighted Pearson (the regression student has to approximate it). train.py --dist-ce adds a
    cross-entropy on the TRUE labels (HL-Gauss smoothed) next to the usual distillation loss on the point output."""

    def __init__(self, n_bins: int = 81, **kw):
        super().__init__(**kw)
        self.n_bins = n_bins
        hh = self.head[0].out_features
        self.head = nn.Sequential(nn.Linear(self.hidden_size, hh), nn.ReLU(), nn.Dropout(kw.get("dropout", 0.1)), nn.Linear(hh, 2 * n_bins))
        self.register_buffer("centers", torch.linspace(-2.0, 2.0, n_bins))
        self.last_logits = None

    def point(self, logits):
        p = torch.softmax(logits, dim=-1); c = self.centers
        return (p * c.abs() * c).sum(-1) / ((p * c.abs()).sum(-1) + 1e-3)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False):
        out, state = self.lstm(self.encode(features), state)
        logits = self.head(out).view(*out.shape[:-1], 2, self.n_bins)
        self.last_logits = logits
        predictions = self.point(logits)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


def hl_gauss(y: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """HL-Gauss target distribution over bins for values y (..., ): a Gaussian at y integrated over each bin."""
    half = (centers[1] - centers[0]) / 2
    edges = torch.cat([centers - half, centers[-1:] + half])
    cdf = torch.special.ndtr((edges - y.clamp(-2, 2).unsqueeze(-1)) / sigma)
    mass = cdf[..., 1:] - cdf[..., :-1]
    return mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)


FUSION_RATES = (0.7, 0.97)       # per-row EMA decays of the fusion member's two moving averages
FUSION_TAPS = (60, 600)          # causal-filter lengths used in training (decay^taps < 1e-8)


def causal_ema(x: torch.Tensor, a: float, taps: int) -> torch.Tensor:
    """x (B, T, C) -> EMA along T with ema_0 = x_0, ema_t = a ema_{t-1} + (1-a) x_t, as one causal conv
    (left padding replicates x_0, which reproduces the ema_0 = x_0 start exactly; truncation error a^taps)."""
    B, T, C = x.shape
    k = (1 - a) * a ** torch.arange(taps, dtype=x.dtype, device=x.device)
    k[-1] = a ** (taps - 1)   # the last tap carries the whole remaining tail (exact when the tail is x_0 padding)
    xp = torch.cat([x[:, :1].expand(B, taps - 1, C), x], dim=1).transpose(1, 2).reshape(B * C, 1, T + taps - 1)
    out = F.conv1d(xp, k.flip(0).view(1, 1, taps))
    return out.view(B, C, T).transpose(1, 2)


class FusionGRUModel(nn.Module):
    """2026-11-04 (032 workstream 1): a structurally DIFFERENT pair member, after the rival's "fusion" members
    (difference + EMA features, 64-unit embedding, 128-unit GRU, skip into the head; their 2nd structure +0.0017).
    Per feature: [x, x - x_prev, x - ema_fast, x - ema_slow, cos x] (5 x 112 = 560) -> Linear 64 + ReLU (embedding)
    -> ONE GRU layer (128) -> head on [h, embedding]. All 112 inputs (both instruments). Streams with extra state
    (x_prev, two EMAs, a started flag); export_slim.build_fusion writes the ONNX graph."""

    def __init__(self, hidden_size: int = 128, head_hidden: int = 64, dropout: float = 0.1, aux_scored: bool = False,
                 embed_dim: int = 64, use_sin: bool = False, **_):
        super().__init__()
        self.hidden_size, self.num_layers, self.bidirectional, self.embed_dim = hidden_size, 1, False, embed_dim
        self.use_sin = use_sin    # 033: a 6th channel sin x per feature (fusion_gru_sin)
        self.embed = nn.Linear((6 if use_sin else 5) * 112, embed_dim)
        self.lstm = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)  # named lstm for train.py helpers
        self.head = nn.Sequential(nn.Linear(hidden_size + embed_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 2))
        self.aux_head = nn.Linear(hidden_size, 1) if aux_scored else None

    def channels_full(self, x):
        xprev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        ef = causal_ema(x, FUSION_RATES[0], FUSION_TAPS[0]); es = causal_ema(x, FUSION_RATES[1], FUSION_TAPS[1])
        return torch.cat([x, x - xprev, x - ef, x - es, torch.cos(x)] + ([torch.sin(x)] if self.use_sin else []), dim=-1)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False):
        if state is None:
            ch = self.channels_full(features); h0 = None; new_aux_state = None
        else:  # streaming (verification only): state = (h, x_prev, ema_f, ema_s, started)
            h0, xp, ef, es, st = state
            chs = []
            for t in range(features.shape[1]):
                x = features[:, t]
                xp_eff = st * xp + (1 - st) * x
                ef = FUSION_RATES[0] * ef + (1 - FUSION_RATES[0]) * x + (1 - st) * FUSION_RATES[0] * (x - ef)
                es = FUSION_RATES[1] * es + (1 - FUSION_RATES[1]) * x + (1 - st) * FUSION_RATES[1] * (x - es)
                chs.append(torch.cat([x, x - xp_eff, x - ef, x - es, torch.cos(x)] + ([torch.sin(x)] if self.use_sin else []), dim=-1)); xp = x; st = torch.ones_like(st)
            ch = torch.stack(chs, dim=1); new_aux_state = (xp, ef, es, st)
        emb = torch.relu(self.embed(ch))
        out, h = self.lstm(emb, h0)
        predictions = self.head(torch.cat([out, emb], dim=-1))
        new_state = h if new_aux_state is None else (h, *new_aux_state)
        if return_aux:
            return predictions, new_state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, new_state


def expand3(raw: torch.Tensor) -> torch.Tensor:
    """raw: (..., d) -> (..., d*3): [x, cos x, sin x]."""
    return torch.cat([raw, torch.cos(raw), torch.sin(raw)], dim=-1)


class FullFeatureModel(nn.Module):
    """2026-11-05 (user request, experiment 033): the best model's design on ALL 112 inputs with a 3-way expansion
    [x, cos x, sin x]. Group encoders (Linear + ReLU): i0 price 66->24, i0 volume 66->24, i1 price 66->24,
    i1 volume 66->24, trade (both instruments) 48->12; the 8 additional columns expanded (24) pass straight on.
    RNN input 132. core = "lstm" (member A) or "gru" (member B); 2 layers x 128; head 128->64->2; aux head."""

    def __init__(self, core: str = "lstm", hidden_size: int = 128, num_layers: int = 2, price_dim: int = 24, volume_dim: int = 24,
                 trade_dim: int = 12, head_hidden: int = 64, dropout: float = 0.1, aux_scored: bool = False, **_):
        super().__init__()
        self.core, self.hidden_size, self.num_layers, self.bidirectional = core, hidden_size, num_layers, False
        self.price_encoder = GroupEncoder(66, price_dim, dropout); self.volume_encoder = GroupEncoder(66, volume_dim, dropout)
        self.i1_price_encoder = GroupEncoder(66, price_dim, dropout); self.i1_volume_encoder = GroupEncoder(66, volume_dim, dropout)
        self.trade_encoder = GroupEncoder(48, trade_dim, dropout)
        in_size = 2 * price_dim + 2 * volume_dim + trade_dim + 24
        rnn = nn.LSTM if core == "lstm" else nn.GRU
        self.lstm = rnn(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 2))
        self.aux_head = nn.Linear(hidden_size, 1) if aux_scored else None

    def encode(self, f: torch.Tensor) -> torch.Tensor:
        trade = torch.cat([f[..., I0_DP[0]:I0_DP[1]], f[..., I0_DV[0]:I0_DV[1]], f[..., I1_DP[0]:I1_DP[1]], f[..., I1_DV[0]:I1_DV[1]]], dim=-1)
        return torch.cat([self.price_encoder(expand3(f[..., I0_PRICE[0]:I0_PRICE[1]])), self.volume_encoder(expand3(f[..., I0_VOLUME[0]:I0_VOLUME[1]])),
                          self.i1_price_encoder(expand3(f[..., I1_PRICE[0]:I1_PRICE[1]])), self.i1_volume_encoder(expand3(f[..., I1_VOLUME[0]:I1_VOLUME[1]])),
                          self.trade_encoder(expand3(trade)), expand3(f[..., ADDITIONAL[0]:ADDITIONAL[1]])], dim=-1)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False):
        out, state = self.lstm(self.encode(features), state)
        predictions = self.head(out)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


class FullLookaheadModel(FullFeatureModel):
    """2026-11-06 (034): the look-ahead TEACHER on all 112 inputs with [x, cos x, sin x] - FullFeatureModel's encoders
    and LSTM + LookaheadLeanModel's gated forward convs over the next K rows (zero-initialised output, as la*).
    The all-feature LSTM student beat the lean one on the platform (0.66597 vs 0.66554); this asks whether its
    TEACHER gains too. Held-out check: valid fold 0, same split / seed / recipe as LA_la12 (ema_wp 0.741101)."""

    def __init__(self, lookahead: int = 12, future_dim: int = 64, **kw):
        super().__init__(core="lstm", **kw)
        self.lookahead = lookahead
        enc_dim = self.lstm.input_size
        self.future_in = nn.Linear(2 * enc_dim, future_dim)
        self.future_convs = nn.ModuleList([GatedFutureConv(future_dim, d) for d in LOOKAHEAD_DILATIONS[lookahead]])
        self.future_out = nn.Linear(future_dim, self.hidden_size)
        nn.init.zeros_(self.future_out.weight); nn.init.zeros_(self.future_out.bias)

    def forward(self, features: torch.Tensor, state=None, return_aux: bool = False):
        encoded = self.encode(features)
        out, state = self.lstm(encoded, state)
        diff = torch.cat([torch.zeros_like(encoded[:, :1]), encoded[:, 1:] - encoded[:, :-1]], dim=1)
        f = self.future_in(torch.cat([encoded, diff], dim=-1)).transpose(1, 2)
        for conv in self.future_convs:
            f = conv(f)
        out = out + self.future_out(f.transpose(1, 2))
        predictions = self.head(out)
        if return_aux:
            return predictions, state, (self.aux_head(out).squeeze(-1) if self.aux_head is not None else None)
        return predictions, state


class JointPairModel(nn.Module):
    """2026-11-03 (032): an LSTM member `a` (lean) and a GRU member `b` (lean_gru) trained TOGETHER; train.py's
    --joint-pair loss rewards their average as well as each member on its own targets. Ships as an ordinary mixed
    pair: split_joint.py writes the two members as separate checkpoints for export_slim.py --multi."""

    def __init__(self, **kw):
        super().__init__()
        self.a = LeanCosGroupedEncoderLSTMModel(**kw)
        self.b = LeanGRUModel(**kw)
        self.hidden_size, self.num_layers, self.bidirectional = self.a.hidden_size, self.a.num_layers, False

    def members(self, features):
        pa, _, xa = self.a(features, return_aux=True)
        pb, _, xb = self.b(features, return_aux=True)
        return pa, pb, xa, xb

    def forward(self, features, state=None, return_aux: bool = False):
        pa, pb, xa, xb = self.members(features)
        pred = 0.5 * (pa + pb)
        if return_aux:
            return pred, None, (0.5 * (xa + xb) if xa is not None else None)
        return pred, None


def build_model(arch: str = "lean", **kw) -> nn.Module:
    """'lean' = LeanCosGroupedEncoderLSTMModel (students, exportable); 'nl_lstm' / 'nl_gru' = 012-style teachers;
    'la25' / 'la50' = look-ahead teachers (lean + the next 25 / 50 rows)."""
    if arch == "lean":
        return LeanCosGroupedEncoderLSTMModel(**kw)
    if arch == "lean_gru":
        return LeanGRUModel(**kw)
    if arch == "joint_pair":
        return JointPairModel(**kw)
    if arch == "lean_dist":
        return LeanDistModel(**kw)
    if arch == "fusion_gru":
        return FusionGRUModel(**kw)
    if arch == "fusion_gru_sin":
        return FusionGRUModel(use_sin=True, **kw)
    if arch in ("full_lstm", "full_gru"):
        return FullFeatureModel(core=arch[5:], **kw)
    if arch.startswith("leanx"):  # leanx_<flags>: i = use_i1, r = res_mlp, p = pos_input (e.g. leanx_irp)
        f = arch.split("_", 1)[1] if "_" in arch else ""
        return LeanXModel(use_i1="i" in f, res_mlp="r" in f, pos_input="p" in f, **kw)
    if arch.startswith("ela"):  # ela<N>: event-time look-ahead teacher over the next N i0 price changes, cap 50 rows (039)
        return EventLookaheadModel(n_events=int(arch[3:]), **kw)
    if arch.startswith("fla"):  # fla12 / fla25 / fla50: all-feature look-ahead teachers (034)
        return FullLookaheadModel(lookahead=int(arch[3:]), **kw)
    if arch.startswith("la"):
        return LookaheadLeanModel(lookahead=int(arch[2:]), **kw)
    return NonlinearGroupedEncoderRNNModel(rnn_type=arch[3:], **kw)
