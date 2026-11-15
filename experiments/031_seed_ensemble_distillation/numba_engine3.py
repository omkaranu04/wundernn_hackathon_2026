"""Numba streaming engine v3 (040, 2026-11-07): ONE compiled call per row for the whole ensemble.

v2 (039) scored 0.66676 in 41m21s with three members. Profiling v2 locally (pinned core, 4,000 real rows):
member kernel alone 35.7 / 24.4 / 28.4 us (full LSTM / lean GRU / lean LSTM) but 41.8 / 29.7 / 33.3 us through
numba's Python dispatcher: ~5.5 us per member per row is argument unboxing (25 array arguments per call).
v3 reads the SAME npz (no repacking) and flattens every member into four arrays (fp32 / int16 / int32 / int64
layout table), so one call with 7 arguments runs all members. Also: cos/sin of the 112 inputs computed once per
row and shared by members (v2 recomputed them per member). The arithmetic of every member is v2's, in v2's order,
so predictions are bit-identical to v2 (checked by `check`).

    python numba_engine3.py check --npz results/numba/triple_A.npz     # bit-equality vs v2 + latency
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

# layout-table columns (int64 per member)
(K_LSTM, K_NT, K_E, K_HD, K_Z, K_NG, K_COLS, K_GDESC, K_WENC, K_BENC,
 K_A0, K_A0S, K_B0, K_A1, K_A1S, K_B1,            # LSTM: fused W0t/W1t; GRU: Wi0t / Wi1t (+ bi)
 K_H0, K_H0S, K_HB0, K_H1, K_H1S, K_HB1,          # GRU only: Wh0t / Wh1t (+ bh)
 K_HW1, K_HW1S, K_HB1V, K_HW2, K_HB2, K_ST, K_W, K_A0N, K_A1N) = range(31)
NK = 31


def flatten(npz_path):
    z = np.load(npz_path)
    meta = json.loads(bytes(z["meta"]).decode())
    F, Q, I = [], [], []
    nf = nq = ni = 0; nst = 0
    L = np.zeros((len(meta), NK), np.int64)

    def addf(a):
        nonlocal nf
        a = np.ascontiguousarray(a, np.float32).reshape(-1); F.append(a); nf += a.size; return nf - a.size

    def addq(a):
        nonlocal nq
        a = np.ascontiguousarray(a, np.int16).reshape(-1); Q.append(a); nq += a.size; return nq - a.size

    def addi(a):
        nonlocal ni
        a = np.ascontiguousarray(a, np.int32).reshape(-1); I.append(a); ni += a.size; return ni - a.size

    for m, mm in enumerate(meta):
        p = f"m{m}_"; g = lambda k: z[p + k]
        r = L[m]; hd = mm["hidden"]
        r[K_LSTM] = 1 if mm["lstm"] else 0; r[K_NT] = mm["nt"]; r[K_E] = mm["enc_dim"]; r[K_HD] = hd
        r[K_NG] = g("gdesc").shape[0]
        r[K_COLS] = addi(g("cols")); r[K_GDESC] = addi(g("gdesc")); r[K_WENC] = addf(g("Wenc")); r[K_BENC] = addf(g("benc"))
        if mm["lstm"]:
            r[K_A0] = addq(g("W0t")); r[K_A0S] = addf(g("W0t_s")); r[K_B0] = addf(g("b0")); r[K_A0N] = g("W0t").shape[0]
            r[K_A1] = addq(g("W1t")); r[K_A1S] = addf(g("W1t_s")); r[K_B1] = addf(g("b1")); r[K_A1N] = g("W1t").shape[0]
            assert g("W0t").shape == (mm["enc_dim"] + hd, 4 * hd) and g("W1t").shape == (2 * hd, 4 * hd)
        else:
            r[K_A0] = addq(g("Wi0t")); r[K_A0S] = addf(g("Wi0t_s")); r[K_B0] = addf(g("bi0")); r[K_A0N] = g("Wi0t").shape[0]
            r[K_A1] = addq(g("Wi1t")); r[K_A1S] = addf(g("Wi1t_s")); r[K_B1] = addf(g("bi1")); r[K_A1N] = g("Wi1t").shape[0]
            r[K_H0] = addq(g("Wh0t")); r[K_H0S] = addf(g("Wh0t_s")); r[K_HB0] = addf(g("bh0"))
            r[K_H1] = addq(g("Wh1t")); r[K_H1S] = addf(g("Wh1t_s")); r[K_HB1] = addf(g("bh1"))
            assert g("Wi0t").shape == (mm["enc_dim"], 3 * hd) and g("Wh0t").shape == (hd, 3 * hd)
        r[K_Z] = g("HW1t").shape[1]
        r[K_HW1] = addq(g("HW1t")); r[K_HW1S] = addf(g("HW1t_s")); r[K_HB1V] = addf(g("Hb1"))
        r[K_HW2] = addf(g("HW2")); r[K_HB2] = addf(g("Hb2"))
        r[K_W] = addf(np.array([mm["weight"]], np.float32))
        r[K_ST] = nst; nst += 4 * hd                   # H (2, hd) then C (2, hd)
    return (np.concatenate(F), np.concatenate(Q), np.concatenate(I), L, nst, meta)


def _kernels():
    @njit(fastmath=True, inline="always")
    def ftanh(x):
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
        # identical to v2: out = W @ v + b, W^T int16 (n_in, n_out), one scale per output, 4 inputs per pass
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
    def encode(cx, sx, x, cols, gdesc, nt, Wenc, benc, e, enc):
        # v2's encode with cos/sin read from the per-row tables (same values: np.cos/np.sin of the same float32)
        for g in range(gdesc.shape[0]):
            cs = gdesc[g, 0]; nc = gdesc[g, 1]; eo = gdesc[g, 2]; no = gdesc[g, 3]; wo = gdesc[g, 4]
            n_in = nc * nt
            for j in range(nc):
                c = cols[cs + j]
                e[j] = x[c]; e[nc + j] = cx[c]
                if nt == 3:
                    e[2 * nc + j] = sx[c]
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
    def ens_step(x, F, Q, I, L, S, pred, need_sin):
        n = x.shape[0]
        cx = np.empty(n, np.float32); sx = np.empty(n, np.float32)
        for c in range(n):
            cx[c] = np.cos(x[c])
        if need_sin:
            for c in range(n):
                sx[c] = np.sin(x[c])
        pred[0] = 0.0; pred[1] = 0.0
        e = np.empty(200, np.float32)
        for m in range(L.shape[0]):
            r = L[m]
            nt = r[K_NT]; E = r[K_E]; Hd = r[K_HD]; Z = r[K_Z]; ng = r[K_NG]
            cols = I[r[K_COLS]:r[K_COLS] + 400]
            gdesc = I[r[K_GDESC]:r[K_GDESC] + 5 * ng].reshape((ng, 5))
            Wenc = F[r[K_WENC]:]; benc = F[r[K_BENC]:r[K_BENC] + E]
            enc = np.empty(E, np.float32)
            encode(cx, sx, x, cols, gdesc, nt, Wenc, benc, e, enc)
            H = S[r[K_ST]:r[K_ST] + 2 * Hd].reshape((2, Hd))
            if r[K_LSTM] == 1:
                C = S[r[K_ST] + 2 * Hd:r[K_ST] + 4 * Hd].reshape((2, Hd))
                G = 4 * Hd
                v0 = np.empty(E + Hd, np.float32); g = np.empty(G, np.float32)
                v0[:E] = enc; v0[E:] = H[0]
                for l in range(2):
                    if l == 0:
                        a = r[K_A0]; nin = r[K_A0N]
                        axpy_gemv(Q[a:a + nin * G].reshape((nin, G)), F[r[K_A0S]:r[K_A0S] + G], F[r[K_B0]:r[K_B0] + G], v0, g)
                    else:
                        v1 = np.empty(2 * Hd, np.float32); v1[:Hd] = H[0]; v1[Hd:] = H[1]
                        a = r[K_A1]; nin = r[K_A1N]
                        axpy_gemv(Q[a:a + nin * G].reshape((nin, G)), F[r[K_A1S]:r[K_A1S] + G], F[r[K_B1]:r[K_B1] + G], v1, g)
                    for j in range(Hd):
                        i = fsig(g[j]); f = fsig(g[Hd + j])
                        c = f * C[l, j] + i * ftanh(g[2 * Hd + j]); C[l, j] = c
                        H[l, j] = fsig(g[3 * Hd + j]) * ftanh(c)
            else:
                G = 3 * Hd
                gi = np.empty(G, np.float32); gh = np.empty(G, np.float32); hin = np.empty(Hd, np.float32)
                for l in range(2):
                    if l == 0:
                        a = r[K_A0]; nin = r[K_A0N]; h = r[K_H0]
                        axpy_gemv(Q[a:a + nin * G].reshape((nin, G)), F[r[K_A0S]:r[K_A0S] + G], F[r[K_B0]:r[K_B0] + G], enc, gi)
                        axpy_gemv(Q[h:h + Hd * G].reshape((Hd, G)), F[r[K_H0S]:r[K_H0S] + G], F[r[K_HB0]:r[K_HB0] + G], H[0], gh)
                    else:
                        hin[:] = H[0]
                        a = r[K_A1]; nin = r[K_A1N]; h = r[K_H1]
                        axpy_gemv(Q[a:a + nin * G].reshape((nin, G)), F[r[K_A1S]:r[K_A1S] + G], F[r[K_B1]:r[K_B1] + G], hin, gi)
                        axpy_gemv(Q[h:h + Hd * G].reshape((Hd, G)), F[r[K_H1S]:r[K_H1S] + G], F[r[K_HB1]:r[K_HB1] + G], H[1], gh)
                    for j in range(Hd):
                        rr = fsig(gi[j] + gh[j]); zz = fsig(gi[Hd + j] + gh[Hd + j])
                        nn = ftanh(gi[2 * Hd + j] + rr * gh[2 * Hd + j])
                        H[l, j] = (1.0 - zz) * nn + zz * H[l, j]
            z = np.empty(Z, np.float32)
            hw = r[K_HW1]
            head(H[1], Q[hw:hw + Hd * Z].reshape((Hd, Z)), F[r[K_HW1S]:r[K_HW1S] + Z], F[r[K_HB1V]:r[K_HB1V] + Z],
                 F[r[K_HW2]:r[K_HW2] + 2 * Z].reshape((2, Z)), F[r[K_HB2]:r[K_HB2] + 2], z, F[r[K_W]], pred)

    return ens_step


class Engine:
    """Streaming ensemble, one compiled call per row: step(features112) -> weighted prediction (2,); reset()."""

    def __init__(self, npz_path):
        self.F, self.Q, self.I, self.L, nst, self.meta = flatten(npz_path)
        self.need_sin = bool(any(m["nt"] == 3 for m in self.meta))
        self.S = np.zeros(nst, np.float32)
        self.pred = np.zeros(2, np.float32)
        self._step = _kernels()
        self.step(np.zeros(112, np.float32)); self.reset()   # compile now, not on the first scored row

    def reset(self):
        self.S[:] = 0.0

    def step(self, x):
        self._step(x, self.F, self.Q, self.I, self.L, self.S, self.pred, self.need_sin)
        return self.pred


def check(npz, steps=4000, seqs=(11, 500)):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import pyarrow.parquet as pq
    from data import file_columns, read_sequence
    import numba_engine2
    pf = pq.ParquetFile(Path(__file__).resolve().parents[2] / "wnn_connectome_starterpack/datasets/valid.parquet")
    hs, ht = file_columns(pf)
    Xs = [read_sequence(pf, s, has_is_scored=hs, has_targets=ht).features[:steps].astype(np.float32) for s in seqs]
    e2, e3 = numba_engine2.Engine(npz), Engine(npz)
    d = 0.0
    for X in Xs:
        e2.reset(); e3.reset()
        for t in range(len(X)):
            d = max(d, float(np.abs(e2.step(X[t]) - e3.step(X[t])).max()))
    print(f"v3 vs v2 max abs diff over {len(Xs)} sequences x {steps} rows: {d:.3e}")
    X = Xs[0]
    for name, e in (("v2", e2), ("v3", e3), ("v2", e2), ("v3", e3)):
        e.reset(); t0 = time.perf_counter()
        for t in range(len(X)):
            e.step(X[t])
        print(f"{name}: {(time.perf_counter() - t0) / len(X) * 1e6:.1f} us/step")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"]); ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    check(a.npz)
