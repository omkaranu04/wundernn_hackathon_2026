"""Numba streaming engine v2 for mixed ensembles (039, 2026-11-07).

Why: the sunArise log runs 3 members (928k params) at 46 us/row on the platform with numba kernels; our ONNX pair
costs ~77 us/row there (50m47s). Locally a numba fp32 kernel at our best pair's shapes runs the matrix work in ~50 us
vs 103 us for the whole IOBinding ONNX step -> room for a THIRD member at about today's pair runtime.

Members supported (any mix, each weighted): full_lstm / full_gru (FullFeatureModel: 112 inputs, [x, cos, sin],
i0/i1 price+volume + trade encoders, additional passed through) and lean / lean_gru (68 inputs, [x, cos]).
Recurrent + head matrices stored TRANSPOSED as int16 with a per-output fp32 scale (bandwidth-bound products; rel. error ~3e-5); encoders fp32; tanh/sigmoid via Eigen's rational approximation (no SVML needed). One compiled call per member per row.

    python numba_engine2.py pack  --out ens.npz --models DIR:ema.pt DIR:ema.pt [--weights 0.5 0.5]
    python numba_engine2.py check --npz ens.npz --models ...       # vs torch on a real valid sequence + latency
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)
TRADE = list(range(*I0_DP)) + list(range(*I0_DV)) + list(range(*I1_DP)) + list(range(*I1_DV))


def _groups(arch):
    """(encoder name or None for pass-through, raw column list), in the model's concatenation order; n transforms."""
    if arch in ("full_lstm", "full_gru"):
        return [("price_encoder", list(range(*I0_PRICE))), ("volume_encoder", list(range(*I0_VOLUME))),
                ("i1_price_encoder", list(range(*I1_PRICE))), ("i1_volume_encoder", list(range(*I1_VOLUME))),
                ("trade_encoder", TRADE), (None, list(range(*ADDITIONAL)))], 3
    if arch in ("lean", "lean_gru"):
        return [("price_encoder", list(range(*I0_PRICE))), ("volume_encoder", list(range(*I0_VOLUME))),
                ("trade_encoder", TRADE), (None, list(range(*ADDITIONAL)))], 2
    raise SystemExit(f"arch {arch} not supported")


# ----------------------------------------------------------------------------- packing (torch side)
def pack(specs, out, weights=None):
    import torch
    weights = weights or [1.0 / len(specs)] * len(specs)
    arrays, meta = {}, []
    for m, spec in enumerate(specs):
        d, w = spec.rsplit(":", 1)
        cfg = json.load(open(Path(d) / "config.json")); arch = cfg.get("arch", "lean")
        assert cfg["num_layers"] == 2
        sd = {k: v.float().numpy() for k, v in torch.load(Path(d) / w, map_location="cpu").items()}
        groups, nt = _groups(arch)
        cols, gdesc, Wflat, bflat = [], [], [], []
        enc_off = 0; w_off = 0
        for name, gc in groups:
            n_in = len(gc) * nt
            if name is None:
                gdesc.append([len(cols), len(gc), enc_off, 0, 0]); enc_off += n_in
            else:
                W = sd[f"{name}.net.0.weight"]; b = sd[f"{name}.net.0.bias"]
                assert W.shape[1] == n_in, (name, W.shape, n_in)
                gdesc.append([len(cols), len(gc), enc_off, W.shape[0], w_off])
                Wflat.append(W.reshape(-1)); w_off += W.size
                bflat.append(b); enc_off += W.shape[0]
            cols += gc
        bvec = np.zeros(enc_off, np.float32)
        for (cs, nc, eo, no, wo), name_gc in zip(gdesc, groups):
            if no:
                bvec[eo:eo + no] = sd[f"{name_gc[0]}.net.0.bias"]
        is_lstm = arch in ("full_lstm", "lean")
        p = f"m{m}_"
        arrays[p + "cols"] = np.array(cols, np.int32); arrays[p + "gdesc"] = np.array(gdesc, np.int32)
        arrays[p + "Wenc"] = np.concatenate(Wflat).astype(np.float32); arrays[p + "benc"] = bvec
        for l in (0, 1):
            wih, whh = sd[f"lstm.weight_ih_l{l}"], sd[f"lstm.weight_hh_l{l}"]
            bih, bhh = sd[f"lstm.bias_ih_l{l}"], sd[f"lstm.bias_hh_l{l}"]
            if is_lstm:  # fused [W_ih | W_hh]^T, one bias
                arrays[p + f"W{l}t"] = np.ascontiguousarray(np.concatenate([wih, whh], 1).T); arrays[p + f"b{l}"] = bih + bhh
            else:        # GRU: keep the parts (the n gate's hidden term sits inside r * (...))
                arrays[p + f"Wi{l}t"] = np.ascontiguousarray(wih.T); arrays[p + f"Wh{l}t"] = np.ascontiguousarray(whh.T)
                arrays[p + f"bi{l}"] = bih; arrays[p + f"bh{l}"] = bhh
        arrays[p + "HW1t"] = np.ascontiguousarray(sd["head.0.weight"].T); arrays[p + "Hb1"] = sd["head.0.bias"]
        arrays[p + "HW2"] = np.ascontiguousarray(sd["head.3.weight"]); arrays[p + "Hb2"] = sd["head.3.bias"]
        meta.append({"spec": spec, "arch": arch, "lstm": is_lstm, "nt": nt, "enc_dim": enc_off, "hidden": int(sd["lstm.weight_hh_l0"].shape[1]),
                     "weight": float(weights[m])})
    arrays = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in arrays.items()}
    for k in [k for k in arrays if k.endswith("t") and k.split("_", 1)[1] in ("W0t", "W1t", "Wi0t", "Wh0t", "Wi1t", "Wh1t", "HW1t")]:
        Wt = arrays[k]                               # (n_in, n_out): one scale per OUTPUT
        sc = np.maximum(np.abs(Wt).max(0), 1e-12) / 32767.0
        arrays[k] = np.ascontiguousarray(np.round(Wt / sc[None, :]).astype(np.int16)); arrays[k + "_s"] = sc.astype(np.float32)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(out, **arrays)
    print(f"packed {len(specs)} members -> {out}: " + ", ".join(f"{x['arch']} (w {x['weight']:.3f})" for x in meta))


# ----------------------------------------------------------------------------- kernels
def _kernels():
    @njit(fastmath=True, inline="always")
    def ftanh(x):
        # Eigen's float tanh: clamp + 13/6 rational approximation (a few ulp), branch-free -> vectorisable
        x = min(max(x, np.float32(-7.99881172180175781)), np.float32(7.99881172180175781))
        x2 = x * x
        p = x2 * np.float32(-2.76076847742355e-16) + np.float32(2.00018790482477e-13)
        p = x2 * p + np.float32(-8.60467152213735e-11)
        p = x2 * p + np.float32(5.12229709037114e-08)
        p = x2 * p + np.float32(1.48572235717979e-05)
        p = x2 * p + np.float32(6.37261928875436e-04)
        p = x2 * p + np.float32(4.89352455891786e-03)
        p = x * p
        q = x2 * np.float32(1.19825839466702e-06) + np.float32(1.18534705686654e-04)
        q = x2 * q + np.float32(2.26843463243900e-03)
        q = x2 * q + np.float32(4.89352518554385e-03)
        return p / q

    @njit(fastmath=True, inline="always")
    def fsig(x):
        return np.float32(0.5) * ftanh(np.float32(0.5) * x) + np.float32(0.5)

    @njit(fastmath=True, inline="always")
    def axpy_gemv(Wq, sc, b, v, out):
        # out = W @ v + b with W^T stored as int16 (n_in, n_out) and one fp32 scale per output: the matrix-vector
        # products are bound by weight bandwidth, int16 halves it (rel. error ~3e-5); 4 inputs per pass over `out`
        n = v.shape[0]; m = out.shape[0]
        for o in range(m):
            out[o] = 0.0
        j = 0
        while j + 4 <= n:
            v0 = v[j]; v1 = v[j + 1]; v2 = v[j + 2]; v3 = v[j + 3]
            for o in range(m):
                out[o] += (np.float32(Wq[j, o]) * v0 + np.float32(Wq[j + 1, o]) * v1
                           + np.float32(Wq[j + 2, o]) * v2 + np.float32(Wq[j + 3, o]) * v3)
            j += 4
        while j < n:
            vj = v[j]
            for o in range(m):
                out[o] += np.float32(Wq[j, o]) * vj
            j += 1
        for o in range(m):
            out[o] = out[o] * sc[o] + b[o]

    @njit(fastmath=True, inline="always")
    def encode(x, cols, gdesc, nt, Wenc, benc, e, enc):
        for g in range(gdesc.shape[0]):
            cs = gdesc[g, 0]; nc = gdesc[g, 1]; eo = gdesc[g, 2]; no = gdesc[g, 3]; wo = gdesc[g, 4]
            n_in = nc * nt
            for j in range(nc):  # expansion [x, cos x, (sin x)], each block contiguous, like expand()/expand3()
                v = x[cols[cs + j]]
                e[j] = v; e[nc + j] = np.cos(v)
                if nt == 3:
                    e[2 * nc + j] = np.sin(v)
            if no == 0:
                for j in range(n_in):
                    enc[eo + j] = e[j]
            else:
                for o in range(no):
                    s = benc[eo + o]
                    base = wo + o * n_in
                    for j in range(n_in):
                        s += Wenc[base + j] * e[j]
                    enc[eo + o] = s if s > 0.0 else 0.0

    @njit(fastmath=True, inline="always")
    def head(h, HW1t, HW1t_s, Hb1, HW2, Hb2, z, wgt, pred):
        axpy_gemv(HW1t, HW1t_s, Hb1, h, z)
        for o in range(z.shape[0]):
            if z[o] < 0.0:
                z[o] = 0.0
        for k in range(2):
            s = Hb2[k]
            for o in range(z.shape[0]):
                s += HW2[k, o] * z[o]
            pred[k] += wgt * s

    @njit(fastmath=True)
    def lstm_step(x, cols, gdesc, nt, Wenc, benc, W0t, W0t_s, b0, W1t, W1t_s, b1, HW1t, HW1t_s, Hb1, HW2, Hb2, H, C, buf, wgt, pred):
        Hd = H.shape[1]; E = W0t.shape[0] - Hd
        e = buf[0:200]; enc = buf[200:200 + E]
        encode(x, cols, gdesc, nt, Wenc, benc, e, enc)
        v0 = np.empty(E + Hd, np.float32); g = np.empty(4 * Hd, np.float32)
        v0[:E] = enc; v0[E:] = H[0]
        for l in range(2):
            if l == 0:
                axpy_gemv(W0t, W0t_s, b0, v0, g)
            else:
                v1 = np.empty(2 * Hd, np.float32); v1[:Hd] = H[0]; v1[Hd:] = H[1]
                axpy_gemv(W1t, W1t_s, b1, v1, g)
            for j in range(Hd):  # gate order i, f, g, o
                i = fsig(g[j]); f = fsig(g[Hd + j])
                c = f * C[l, j] + i * ftanh(g[2 * Hd + j]); C[l, j] = c
                H[l, j] = fsig(g[3 * Hd + j]) * ftanh(c)
        z = np.empty(HW1t.shape[1], np.float32)
        head(H[1], HW1t, HW1t_s, Hb1, HW2, Hb2, z, wgt, pred)

    @njit(fastmath=True)
    def gru_step(x, cols, gdesc, nt, Wenc, benc, Wi0t, Wi0t_s, Wh0t, Wh0t_s, bi0, bh0, Wi1t, Wi1t_s, Wh1t, Wh1t_s, bi1, bh1, HW1t, HW1t_s, Hb1, HW2, Hb2, H, buf, wgt, pred):
        Hd = H.shape[1]; E = Wi0t.shape[0]
        e = buf[0:200]; enc = buf[200:200 + E]
        encode(x, cols, gdesc, nt, Wenc, benc, e, enc)
        gi = np.empty(3 * Hd, np.float32); gh = np.empty(3 * Hd, np.float32); hin = np.empty(Hd, np.float32)
        for l in range(2):
            if l == 0:
                axpy_gemv(Wi0t, Wi0t_s, bi0, enc, gi); axpy_gemv(Wh0t, Wh0t_s, bh0, H[0], gh)
            else:
                hin[:] = H[0]
                axpy_gemv(Wi1t, Wi1t_s, bi1, hin, gi); axpy_gemv(Wh1t, Wh1t_s, bh1, H[1], gh)
            for j in range(Hd):  # gate order r, z, n
                r = fsig(gi[j] + gh[j]); zz = fsig(gi[Hd + j] + gh[Hd + j])
                n = ftanh(gi[2 * Hd + j] + r * gh[2 * Hd + j])
                H[l, j] = (1.0 - zz) * n + zz * H[l, j]
        z = np.empty(HW1t.shape[1], np.float32)
        head(H[1], HW1t, HW1t_s, Hb1, HW2, Hb2, z, wgt, pred)

    return lstm_step, gru_step


class Engine:
    """Streaming ensemble: step(features112) -> weighted prediction (2,); reset() per sequence."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.meta = json.loads(bytes(z["meta"]).decode())
        self.lstm_step, self.gru_step = _kernels()
        self.members = []
        for m, mm in enumerate(self.meta):
            p = f"m{m}_"; g = lambda k: np.ascontiguousarray(z[p + k])
            common = (g("cols"), g("gdesc"), np.int64(mm["nt"]), g("Wenc"), g("benc"))
            if mm["lstm"]:
                ws = (g("W0t"), g("W0t_s"), g("b0"), g("W1t"), g("W1t_s"), g("b1"))
            else:
                ws = (g("Wi0t"), g("Wi0t_s"), g("Wh0t"), g("Wh0t_s"), g("bi0"), g("bh0"), g("Wi1t"), g("Wi1t_s"), g("Wh1t"), g("Wh1t_s"), g("bi1"), g("bh1"))
            head = (g("HW1t"), g("HW1t_s"), g("Hb1"), g("HW2"), g("Hb2"))
            self.members.append([mm["lstm"], common + ws + head, mm["hidden"], np.float32(mm["weight"])])
        self.buf = np.zeros(400, np.float32)
        self.pred = np.zeros(2, np.float32)
        self.reset()
        x = np.zeros(112, np.float32)
        self.step(x); self.reset()   # compile both kernels now, not on the first scored row

    def reset(self):
        self.H = [np.zeros((2, m[2]), np.float32) for m in self.members]
        self.C = [np.zeros((2, m[2]), np.float32) if m[0] else None for m in self.members]

    def step(self, x):
        p = self.pred; p[0] = 0.0; p[1] = 0.0
        for k, (is_lstm, args, _, wgt) in enumerate(self.members):
            if is_lstm:
                self.lstm_step(x, *args, self.H[k], self.C[k], self.buf, wgt, p)
            else:
                self.gru_step(x, *args, self.H[k], self.buf, wgt, p)
        return p


# ----------------------------------------------------------------------------- check vs torch + latency
def check(npz, specs, weights=None, steps=3000, seq=11):
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import pyarrow.parquet as pq
    from data import file_columns, read_sequence
    from teacher_preds import load_model
    weights = weights or [1.0 / len(specs)] * len(specs)
    pf = pq.ParquetFile(Path(__file__).resolve().parents[2] / "wnn_connectome_starterpack/datasets/valid.parquet")
    hs, ht = file_columns(pf)
    X = read_sequence(pf, seq, has_is_scored=hs, has_targets=ht).features[:steps].astype(np.float32)
    with torch.no_grad():
        ref = sum(w * load_model(s, torch.device("cpu"))(torch.from_numpy(X)[None])[0][0].numpy() for s, w in zip(specs, weights))
    eng = Engine(npz)
    got = np.stack([eng.step(X[t]).copy() for t in range(steps)])
    print(f"engine vs torch (weighted mean of {len(specs)} members, real valid sequence {seq}, {steps} rows): "
          f"max abs diff {np.abs(got - ref).max():.3e}, mean |pred| {np.abs(ref).mean():.3f}")
    eng.reset()
    for rep in range(3):
        t0 = time.perf_counter()
        for t in range(steps):
            eng.step(X[t])
        print(f"latency {(time.perf_counter() - t0) / steps * 1e6:.1f} us/step (engine call only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "check"])
    ap.add_argument("--out"); ap.add_argument("--npz"); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.models, a.out, a.weights)
    else:
        check(a.npz, a.models, a.weights)
