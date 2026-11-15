"""Fused numba inference engine for an ENSEMBLE of lean students (2026-11-02).

Why: the ONNX graphs pay a large fixed cost per call (~70 us/row for one width-128 student), and a merged
pair wastes half its recurrent work on block-diagonal zeros (100 us/row). Diverse pairs gave +0.0006-0.0007
on the platform, so more members should pay - if they fit in the time budget. This engine runs K members
(lean encoder + LSTM or GRU core + head, any mix) in ONE compiled call per row, with no wasted zeros.

Members are packed into one .npz (`pack`), the competition `solution.py` (solution_numba.py) loads it and
falls back to ONNX if numba is missing. Weights are float32; the ensemble prediction is the plain mean.

    python numba_engine.py pack --out ens.npz --models results/checkpoints_R6_LA25_A:ema.pt results/checkpoints_R6_LA12_A:ema.pt
    python numba_engine.py check --npz ens.npz --models ...     # vs torch forward, plus latency
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # the engine needs numba; packing/checking does not use it directly
    njit = None

# 112-dim raw feature layout (model.py)
I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_DP, I1_DV = (96, 100), (100, 104)
ADDITIONAL = (104, 112)


# ----------------------------------------------------------------------------- packing (torch side)
def pack(specs, out):
    """specs: ['CKPT_DIR:WEIGHTS', ...] of lean / lean_gru students with EQUAL widths per core type."""
    import torch

    lstm, gru = [], []
    for spec in specs:
        d, w = spec.rsplit(":", 1)
        cfg = json.load(open(Path(d) / "config.json"))
        arch = cfg.get("arch", "lean")
        assert arch in ("lean", "lean_gru"), f"{spec}: arch {arch} not supported by the engine"
        assert cfg["num_layers"] == 2, "engine assumes 2 recurrent layers"
        sd = {k: v.float().numpy() for k, v in torch.load(Path(d) / w, map_location="cpu").items()}
        m = {"Wp": sd["price_encoder.net.0.weight"], "bp": sd["price_encoder.net.0.bias"],
             "Wv": sd["volume_encoder.net.0.weight"], "bv": sd["volume_encoder.net.0.bias"],
             "Wt": sd["trade_encoder.net.0.weight"], "bt": sd["trade_encoder.net.0.bias"],
             "HW1": sd["head.0.weight"], "Hb1": sd["head.0.bias"], "HW2": sd["head.3.weight"], "Hb2": sd["head.3.bias"]}
        for l in (0, 1):
            wih, whh = sd[f"lstm.weight_ih_l{l}"], sd[f"lstm.weight_hh_l{l}"]
            bih, bhh = sd[f"lstm.bias_ih_l{l}"], sd[f"lstm.bias_hh_l{l}"]
            if arch == "lean":   # fused [W_ih | W_hh], one bias
                m[f"Wl{l}"] = np.concatenate([wih, whh], axis=1); m[f"bl{l}"] = bih + bhh
            else:                # GRU: the n-gate's hidden bias sits inside r * (...), keep the parts apart
                m[f"Wi{l}"], m[f"Wh{l}"], m[f"bi{l}"], m[f"bh{l}"] = wih, whh, bih, bhh
        (lstm if arch == "lean" else gru).append(m)
    arrays = {}
    for tag, group in (("L", lstm), ("G", gru)):
        if group:
            for k in group[0]:
                arrays[f"{tag}_{k}"] = np.ascontiguousarray(np.stack([g[k] for g in group]).astype(np.float32))
    arrays["n_lstm"] = np.array(len(lstm)); arrays["n_gru"] = np.array(len(gru))
    np.savez(out, **arrays)
    print(f"packed {len(lstm)} LSTM + {len(gru)} GRU members -> {out}")


# ----------------------------------------------------------------------------- kernels
def _kernels():
    @njit(cache=True, fastmath=True)
    def encode(x, Wp, bp, Wv, bv, Wt, bt, out):
        # expansion [x, cos x] per group, grouped Linear + ReLU; `additional` expanded and passed straight on
        ep = np.empty(44, np.float32); ev = np.empty(44, np.float32); et = np.empty(32, np.float32)
        for j in range(22):
            ep[j] = x[j]; ep[22 + j] = np.cos(x[j])
            ev[j] = x[22 + j]; ev[22 + j] = np.cos(x[22 + j])
        for j in range(4):
            et[j] = x[44 + j]; et[4 + j] = x[48 + j]; et[8 + j] = x[96 + j]; et[12 + j] = x[100 + j]
        for j in range(16):
            et[16 + j] = np.cos(et[j])
        for o in range(Wp.shape[0]):
            s = bp[o]
            for j in range(44):
                s += Wp[o, j] * ep[j]
            out[o] = s if s > 0 else 0.0
        P = Wp.shape[0]
        for o in range(Wv.shape[0]):
            s = bv[o]
            for j in range(44):
                s += Wv[o, j] * ev[j]
            out[P + o] = s if s > 0 else 0.0
        V = Wv.shape[0]
        for o in range(Wt.shape[0]):
            s = bt[o]
            for j in range(32):
                s += Wt[o, j] * et[j]
            out[P + V + o] = s if s > 0 else 0.0
        T = Wt.shape[0]
        for j in range(8):
            out[P + V + T + j] = x[104 + j]; out[P + V + T + 8 + j] = np.cos(x[104 + j])

    @njit(cache=True, fastmath=True)
    def head(h, W1, b1, W2, b2, pred):
        z = np.empty(W1.shape[0], np.float32)
        for o in range(W1.shape[0]):
            s = b1[o]
            for j in range(h.shape[0]):
                s += W1[o, j] * h[j]
            z[o] = s if s > 0 else 0.0
        for o in range(2):
            s = b2[o]
            for j in range(z.shape[0]):
                s += W2[o, j] * z[j]
            pred[o] += s

    @njit(cache=True, fastmath=True, inline="always")
    def gemv(W, b, v, out):
        # out = W @ v + b, W C-contiguous (rows, len(v)): a flat dot per row, which LLVM vectorises
        n = v.shape[0]
        for o in range(W.shape[0]):
            s = np.float32(0.0)
            for j in range(n):
                s += W[o, j] * v[j]
            out[o] = s + b[o]

    @njit(cache=True, fastmath=True)
    def lstm_members(x, H, C, Wp, bp, Wv, bv, Wt, bt, W0, b0, W1l, b1l, W1, b1, W2, b2, pred):
        K = W0.shape[0]; Hd = H.shape[2]; enc_dim = W0.shape[2] - Hd
        enc = np.empty(enc_dim, np.float32); g = np.empty(4 * Hd, np.float32)
        v0 = np.empty(enc_dim + Hd, np.float32); v1 = np.empty(2 * Hd, np.float32)
        for m in range(K):
            encode(x, Wp[m], bp[m], Wv[m], bv[m], Wt[m], bt[m], enc)
            for l in range(2):
                if l == 0:
                    v0[:enc_dim] = enc; v0[enc_dim:] = H[m, 0]
                    gemv(W0[m], b0[m], v0, g)
                else:
                    v1[:Hd] = H[m, 0]; v1[Hd:] = H[m, 1]
                    gemv(W1l[m], b1l[m], v1, g)
                for j in range(Hd):  # PyTorch gate order i, f, g, o
                    i = 1.0 / (1.0 + np.exp(-g[j])); f = 1.0 / (1.0 + np.exp(-g[Hd + j]))
                    c = f * C[m, l, j] + i * np.tanh(g[2 * Hd + j])
                    C[m, l, j] = c
                    H[m, l, j] = (1.0 / (1.0 + np.exp(-g[3 * Hd + j]))) * np.tanh(c)
            head(H[m, 1], W1[m], b1[m], W2[m], b2[m], pred)

    @njit(cache=True, fastmath=True)
    def gru_members(x, H, Wp, bp, Wv, bv, Wt, bt, Wi0, Wh0, bi0, bh0, Wi1, Wh1, bi1, bh1, W1, b1, W2, b2, pred):
        K = Wi0.shape[0]; Hd = H.shape[2]
        enc = np.empty(Wi0.shape[2], np.float32); gi = np.empty(3 * Hd, np.float32); gh = np.empty(3 * Hd, np.float32)
        h = np.empty(Hd, np.float32)
        for m in range(K):
            encode(x, Wp[m], bp[m], Wv[m], bv[m], Wt[m], bt[m], enc)
            for l in range(2):
                h[:] = H[m, l]
                if l == 0:
                    gemv(Wi0[m], bi0[m], enc, gi); gemv(Wh0[m], bh0[m], h, gh)
                else:
                    gemv(Wi1[m], bi1[m], H[m, 0].copy(), gi); gemv(Wh1[m], bh1[m], h, gh)
                for j in range(Hd):  # PyTorch gate order r, z, n
                    r = 1.0 / (1.0 + np.exp(-(gi[j] + gh[j]))); z = 1.0 / (1.0 + np.exp(-(gi[Hd + j] + gh[Hd + j])))
                    n = np.tanh(gi[2 * Hd + j] + r * gh[2 * Hd + j])
                    H[m, l, j] = (1.0 - z) * n + z * h[j]
            head(H[m, 1], W1[m], b1[m], W2[m], b2[m], pred)

    return lstm_members, gru_members


class Engine:
    """Streaming ensemble: step(features112) -> mean prediction (2,), state kept per member; reset() per sequence."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.nl, self.ng = int(z["n_lstm"]), int(z["n_gru"])
        self.L = {k[2:]: z[k] for k in z.files if k.startswith("L_")}
        self.G = {k[2:]: z[k] for k in z.files if k.startswith("G_")}
        self.lstm_members, self.gru_members = _kernels()
        self.pred = np.zeros(2, np.float32)
        self.reset()

    def reset(self):
        if self.nl:
            Hd = self.L["HW1"].shape[2]
            self.LH = np.zeros((self.nl, 2, Hd), np.float32); self.LC = np.zeros((self.nl, 2, Hd), np.float32)
        if self.ng:
            Hd = self.G["HW1"].shape[2]
            self.GH = np.zeros((self.ng, 2, Hd), np.float32)

    def step(self, x):
        p = self.pred; p[:] = 0.0
        if self.nl:
            L = self.L
            self.lstm_members(x, self.LH, self.LC, L["Wp"], L["bp"], L["Wv"], L["bv"], L["Wt"], L["bt"],
                              L["Wl0"], L["bl0"], L["Wl1"], L["bl1"], L["HW1"], L["Hb1"], L["HW2"], L["Hb2"], p)
        if self.ng:
            G = self.G
            self.gru_members(x, self.GH, G["Wp"], G["bp"], G["Wv"], G["bv"], G["Wt"], G["bt"],
                             G["Wi0"], G["Wh0"], G["bi0"], G["bh0"], G["Wi1"], G["Wh1"], G["bi1"], G["bh1"],
                             G["HW1"], G["Hb1"], G["HW2"], G["Hb2"], p)
        return p / (self.nl + self.ng)


# ----------------------------------------------------------------------------- check: vs torch, and latency
def check(npz, specs, steps=600, bench=20000):
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from teacher_preds import load_model
    eng = Engine(npz)
    torch.manual_seed(0)
    x = torch.randn(1, steps, 112)
    with torch.no_grad():
        ref = sum(load_model(s, torch.device("cpu"))(x)[0][0] for s in specs).numpy() / len(specs)
    xs = x[0].numpy().astype(np.float32)
    eng.reset(); got = np.stack([eng.step(xs[t]).copy() for t in range(steps)])
    print(f"engine vs torch mean of {len(specs)} members: max abs diff {np.abs(got - ref).max():.3e} over {steps} steps")
    F = np.random.default_rng(0).standard_normal((bench, 112)).astype(np.float32)
    eng.reset()
    for t in range(300):
        eng.step(F[t])
    t0 = time.perf_counter()
    for t in range(bench):
        eng.step(F[t])
    us = (time.perf_counter() - t0) / bench * 1e6
    print(f"latency {us:.1f} us/step (engine call only, {eng.nl} LSTM + {eng.ng} GRU members)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "check"])
    ap.add_argument("--out"); ap.add_argument("--npz"); ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.models, a.out)
    else:
        check(a.npz, a.models)
