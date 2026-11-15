"""Numba streaming engine v6 (2026-11-14): v5 with INTEGER encoders.

Profile of v5 on the 0.66754 quint (local): ~105 us/row = integer gemvs ~75 + encoders ~20 + rest ~10; the fp32
encoder ran ~1.5 GMAC/s. v6 runs each encoder group (24 or 12 outputs x <= 66 inputs) through v5's integer pair
multiply-add: its fp32 weights are quantized to int16 per output (scale max|w|/32767), inputs padded with zeros to a
multiple of 4 and quantized per row with the overflow-proof qmax; then * scale + bias, ReLU. Everything else is v5.

    python numba_engine6.py check --npz results/numba/q49_swapD25_plus_s1.npz     # diff vs v5 + latency
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit, types
    from numba.extending import intrinsic
    from numba.core import cgutils
    from llvmlite import ir
except ImportError:
    njit = None

# layout-table columns (int64 per member)
(K_LSTM, K_NT, K_E, K_HD, K_Z, K_NG, K_COLS, K_GDESC, K_WENC, K_BENC,
 K_A0, K_A0S, K_B0, K_A1, K_A1S, K_B1,            # LSTM: fused W0t/W1t; GRU: Wi0t / Wi1t (+ bi)
 K_H0, K_H0S, K_HB0, K_H1, K_H1S, K_HB1,          # GRU only: Wh0t / Wh1t (+ bh)
 K_HW1, K_HW1S, K_HB1V, K_HW2, K_HB2, K_ST, K_W, K_A0N, K_A1N, K_QM) = range(32)
NK = 32
# K_QM: fp32 offset of 7 per-segment qmax values: LSTM [l0 enc, l0 h0, l1 h0, l1 h1, -, -, head];
#       GRU [gi0 enc, gh0 h0, gi1 h0, gh1 h1, -, -, head]


def _flatten4(npz_path):
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
        Wenc = np.array(g("Wenc"), np.float32).reshape(-1).copy()
        for cs, nc, eo, no, wo in g("gdesc"):                 # (no, n_in) -> (n_in, no) per group, same offsets
            if no > 0:
                n_in = nc * mm["nt"]
                Wenc[wo:wo + no * n_in] = Wenc[wo:wo + no * n_in].reshape(no, n_in).T.reshape(-1)
        r[K_COLS] = addi(g("cols")); r[K_GDESC] = addi(g("gdesc")); r[K_WENC] = addf(Wenc); r[K_BENC] = addf(g("benc"))
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


def segments(r):
    """[(slot, Q offset of the segment block, rows, n_out)] for one member's layout row."""
    hd = int(r[K_HD]); E = int(r[K_E])
    if r[K_LSTM]:
        G = 4 * hd
        out = [(0, r[K_A0], E, G), (1, r[K_A0] + E * G, hd, G), (2, r[K_A1], hd, G), (3, r[K_A1] + hd * G, hd, G)]
    else:
        G = 3 * hd
        out = [(0, r[K_A0], int(r[K_A0N]), G), (1, r[K_H0], hd, G), (2, r[K_A1], int(r[K_A1N]), G), (3, r[K_H1], hd, G)]
    return [(k, int(a), int(n), int(m)) for k, a, n, m in out] + [(6, int(r[K_HW1]), hd, int(r[K_Z]))]


def flatten(npz_path):
    F, Q, I, L4, nst, meta = _flatten4(npz_path)
    L = np.zeros((L4.shape[0], NK), np.int64); L[:, :L4.shape[1]] = L4
    Q = Q.copy(); qms = []
    for m in range(L.shape[0]):
        qm = np.ones(7, np.float32)
        for slot, a, n, mo in segments(L[m]):
            assert n % 4 == 0 and mo % 8 == 0, (n, mo)
            W = Q[a:a + n * mo].reshape(n, mo)
            worst = int(np.abs(W.astype(np.int64)).sum(0).max())
            qm[slot] = float(min(32767, (2 ** 31 - 1) // max(worst, 1)))
            Q[a:a + n * mo] = np.ascontiguousarray(W.reshape(n // 2, 2, mo).transpose(0, 2, 1)).reshape(-1)
        L[m, K_QM] = F.size + 7 * m; qms.append(qm)
    F = np.concatenate([F] + qms)
    Fx, Qx, Ix = [F], [Q], [I]; nf, nq, ni = F.size, Q.size, I.size
    for m in range(L.shape[0]):
        r = L[m]; nt = int(r[K_NT]); ng = int(r[K_NG])
        gd = I[r[K_GDESC]:r[K_GDESC] + 5 * ng].reshape(ng, 5).astype(np.int64)
        g9 = np.zeros((ng, 9), np.int32); g9[:, :5] = gd
        for gi, (cs, nc, eo, no, wo) in enumerate(gd):
            if no == 0:
                continue
            n_in = int(nc) * nt; npad = (n_in + 3) // 4 * 4
            W = F[r[K_WENC] + wo:r[K_WENC] + wo + n_in * no].reshape(n_in, no).astype(np.float64)   # (n_in, no), v4 layout
            mo = (int(no) + 7) // 8 * 8                       # outputs padded to a multiple of 8 (zero weights)
            sc = np.ones(mo); sc[:no] = np.abs(W).max(0) / 32767.0; sc[sc == 0] = 1.0
            Wq = np.zeros((npad, mo), np.int16); Wq[:n_in, :no] = np.round(W / sc[:no]).astype(np.int16)
            worst = int(np.abs(Wq.astype(np.int64)).sum(0).max())
            g9[gi, 5] = nq; Qx.append(np.ascontiguousarray(Wq.reshape(npad // 2, 2, mo).transpose(0, 2, 1)).reshape(-1)); nq += npad * mo
            g9[gi, 6] = nf; Fx.append(sc.astype(np.float32)); nf += mo
            g9[gi, 7] = npad; g9[gi, 8] = min(32767, (2 ** 31 - 1) // max(worst, 1))
        L[m, K_GDESC] = ni; Ix.append(g9.reshape(-1)); ni += g9.size
    return np.concatenate(Fx), np.concatenate(Qx), np.concatenate(Ix).astype(np.int32), L, nst, meta


def _kernels():
    @njit(fastmath=True, error_model="numpy", inline="always")
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

    @njit(fastmath=True, error_model="numpy", inline="always")
    def fsig(x):
        return np.float32(0.5) * ftanh(np.float32(0.5) * x) + np.float32(0.5)

    def _pm(builder, base, off, b32):
        i16, i32 = ir.IntType(16), ir.IntType(32)
        v16, v8 = ir.VectorType(i32, 16), ir.VectorType(i32, 8)
        w = builder.load(builder.bitcast(builder.gep(base, [off]), ir.PointerType(ir.VectorType(i16, 16))), align=2)
        bv = builder.insert_element(v8(ir.Undefined), b32, ir.Constant(i32, 0))
        bv = builder.shuffle_vector(bv, v8(ir.Undefined), ir.Constant(v8, [0] * 8))
        p = builder.mul(builder.sext(w, v16), builder.sext(builder.bitcast(bv, ir.VectorType(i16, 16)), v16), flags=["nsw"])
        ev = builder.shuffle_vector(p, v16(ir.Undefined), ir.Constant(v8, list(range(0, 16, 2))))
        od = builder.shuffle_vector(p, v16(ir.Undefined), ir.Constant(v8, list(range(1, 16, 2))))
        return builder.add(ev, od, flags=["nsw"])

    @intrinsic
    def madd2(typingctx, acc, ao, W, w0, w1, b0, b1):
        # acc[ao:ao+8] += pairsum(W[w0:w0+16] * bcast16x2(b0)) + pairsum(W[w1:w1+16] * bcast16x2(b1))   (int32)
        sig = types.void(acc, ao, W, w0, w1, b0, b1)

        def codegen(context, builder, signature, args):
            acc_a, ao_v, W_a, w0_v, w1_v, b0_v, b1_v = args
            accd = cgutils.create_struct_proxy(signature.args[0])(context, builder, value=acc_a).data
            Wd = cgutils.create_struct_proxy(signature.args[2])(context, builder, value=W_a).data
            i32 = ir.IntType(32)
            t = lambda v: builder.trunc(v, i32) if v.type.width > 32 else v
            s = builder.add(_pm(builder, Wd, w0_v, t(b0_v)), _pm(builder, Wd, w1_v, t(b1_v)))
            ap = builder.bitcast(builder.gep(accd, [ao_v]), ir.PointerType(ir.VectorType(i32, 8)))
            builder.store(builder.add(builder.load(ap, align=4), s), ap, align=4)
            return context.get_dummy_value()
        return sig, codegen

    @njit(fastmath=True, error_model="numpy", inline="always")
    def qseg(Q, a, m, v, qmax, vq, acc, out, first):
        # out[:m] (=|+=) s * (W_seg^T @ round(v / s)) with W_seg the pair-laid-out int16 block at Q[a:], s = amax/qmax
        n = v.shape[0]
        amax = np.float32(0.0)
        for j in range(n):
            x = abs(v[j])
            amax = x if x > amax else amax
        if amax == 0.0:
            if first:
                for o in range(m):
                    out[o] = 0.0
            return
        inv = qmax / amax; s = amax / qmax
        for j in range(n):
            x = v[j] * inv
            vq[j] = np.int16(x + np.float32(0.5)) if x >= 0.0 else np.int16(x - np.float32(0.5))
        for o in range(m):
            acc[o] = 0
        P = n // 2; p = 0
        while p < P:
            b0 = (np.int64(vq[2 * p]) & 0xFFFF) | ((np.int64(vq[2 * p + 1]) & 0xFFFF) << 16)
            b1 = (np.int64(vq[2 * p + 2]) & 0xFFFF) | ((np.int64(vq[2 * p + 3]) & 0xFFFF) << 16)
            base0 = a + p * m * 2; base1 = base0 + m * 2
            for ob in range(0, m, 8):
                madd2(acc, ob, Q, base0 + 2 * ob, base1 + 2 * ob, b0, b1)
            p += 2
        if first:
            for o in range(m):
                out[o] = np.float32(acc[o]) * s
        else:
            for o in range(m):
                out[o] += np.float32(acc[o]) * s

    @njit(fastmath=True, error_model="numpy", inline="always")
    def fin(sc, b, out):
        for o in range(out.shape[0]):
            out[o] = out[o] * sc[o] + b[o]

    @njit(fastmath=True, error_model="numpy", inline="always")
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

    @njit(fastmath=True, error_model="numpy", inline="always")
    def encode(cx, sx, x, cols, gdesc, nt, Q, F, benc, e, enc, vq, acc, ob):
        # gdesc rows: cs, nc, eo, no, wo, qoff, soff, npad, qm
        for g in range(gdesc.shape[0]):
            cs = gdesc[g, 0]; nc = gdesc[g, 1]; eo = gdesc[g, 2]; no = gdesc[g, 3]
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
                npad = gdesc[g, 7]
                for j in range(n_in, npad):
                    e[j] = 0.0
                mo = (no + 7) // 8 * 8
                qseg(Q, gdesc[g, 5], mo, e[:npad], np.float32(gdesc[g, 8]), vq, acc, ob, True)
                so = gdesc[g, 6]
                for o in range(no):
                    v = ob[o] * F[so + o] + benc[eo + o]
                    enc[eo + o] = v if v > 0.0 else np.float32(0.0)

    @njit(fastmath=True, error_model="numpy", inline="always")
    def head(h, Q, a, qm, HW1t_s, Hb1, HW2, Hb2, z, wgt, pred, vq, acc):
        qseg(Q, a, z.shape[0], h, qm, vq, acc, z, True)
        fin(HW1t_s, Hb1, z)
        for o in range(z.shape[0]):
            if z[o] < 0.0:
                z[o] = 0.0
        for k in range(2):
            s = Hb2[k]
            for o in range(z.shape[0]):
                s += HW2[k, o] * z[o]
            pred[k] += wgt * s

    @njit(fastmath=True, error_model="numpy")
    def ens_step(x, F, Q, I, L, S, pred, need_sin, T, TQ, TA):
        # T: float32 scratch (>= 3072): cx 0:112, sx 128:240, e 256:456, enc 512:768, v 768:1280, g 1280:1792,
        #    gh 1792:2304, t 2304:2816, z 2816:3072
        n = x.shape[0]
        cx = T[0:112]; sx = T[128:240]; e = T[256:456]
        for c in range(n):
            cx[c] = np.cos(x[c])
        if need_sin:
            for c in range(n):
                sx[c] = np.sin(x[c])
        pred[0] = 0.0; pred[1] = 0.0
        for m in range(L.shape[0]):
            r = L[m]
            nt = r[K_NT]; E = r[K_E]; Hd = r[K_HD]; Z = r[K_Z]; ng = r[K_NG]
            cols = I[r[K_COLS]:r[K_COLS] + 400]
            gdesc = I[r[K_GDESC]:r[K_GDESC] + 9 * ng].reshape((ng, 9))
            Wenc = F[r[K_WENC]:]; benc = F[r[K_BENC]:r[K_BENC] + E]
            enc = T[512:512 + E]
            qm = F[r[K_QM]:r[K_QM] + 7]
            encode(cx, sx, x, cols, gdesc, nt, Q, F, benc, e, enc, TQ, TA, T[2880:2912])
            H = S[r[K_ST]:r[K_ST] + 2 * Hd].reshape((2, Hd))
            if r[K_LSTM] == 1:
                C = S[r[K_ST] + 2 * Hd:r[K_ST] + 4 * Hd].reshape((2, Hd))
                G = 4 * Hd
                g = T[1280:1280 + G]; tc = T[2304:2304 + Hd]
                for l in range(2):
                    if l == 0:
                        a = r[K_A0]
                        qseg(Q, a, G, enc, qm[0], TQ, TA, g, True)
                        qseg(Q, a + E * G, G, H[0], qm[1], TQ, TA, g, False)
                        fin(F[r[K_A0S]:r[K_A0S] + G], F[r[K_B0]:r[K_B0] + G], g)
                    else:
                        a = r[K_A1]
                        qseg(Q, a, G, H[0], qm[2], TQ, TA, g, True)
                        qseg(Q, a + Hd * G, G, H[1], qm[3], TQ, TA, g, False)
                        fin(F[r[K_A1S]:r[K_A1S] + G], F[r[K_B1]:r[K_B1] + G], g)
                    for j in range(2 * Hd):                       # i, f
                        g[j] = fsig(g[j])
                    for j in range(2 * Hd, 3 * Hd):               # cell candidate
                        g[j] = ftanh(g[j])
                    for j in range(3 * Hd, 4 * Hd):               # o
                        g[j] = fsig(g[j])
                    Cl = C[l]; Hl = H[l]
                    for j in range(Hd):
                        c = g[Hd + j] * Cl[j] + g[j] * g[2 * Hd + j]
                        Cl[j] = c; tc[j] = c
                    for j in range(Hd):
                        tc[j] = ftanh(tc[j])
                    for j in range(Hd):
                        Hl[j] = g[3 * Hd + j] * tc[j]
            else:
                G = 3 * Hd
                gi = T[1280:1280 + G]; gh = T[1792:1792 + G]; t = T[2304:2304 + G]
                for l in range(2):
                    if l == 0:
                        qseg(Q, r[K_A0], G, enc, qm[0], TQ, TA, gi, True)
                        fin(F[r[K_A0S]:r[K_A0S] + G], F[r[K_B0]:r[K_B0] + G], gi)
                        qseg(Q, r[K_H0], G, H[0], qm[1], TQ, TA, gh, True)
                        fin(F[r[K_H0S]:r[K_H0S] + G], F[r[K_HB0]:r[K_HB0] + G], gh)
                    else:
                        qseg(Q, r[K_A1], G, H[0], qm[2], TQ, TA, gi, True)
                        fin(F[r[K_A1S]:r[K_A1S] + G], F[r[K_B1]:r[K_B1] + G], gi)
                        qseg(Q, r[K_H1], G, H[1], qm[3], TQ, TA, gh, True)
                        fin(F[r[K_H1S]:r[K_H1S] + G], F[r[K_HB1]:r[K_HB1] + G], gh)
                    for j in range(2 * Hd):                       # r, z
                        t[j] = fsig(gi[j] + gh[j])
                    for j in range(Hd):                           # n
                        t[2 * Hd + j] = ftanh(gi[2 * Hd + j] + t[j] * gh[2 * Hd + j])
                    Hl = H[l]
                    for j in range(Hd):
                        zz = t[Hd + j]
                        Hl[j] = (1.0 - zz) * t[2 * Hd + j] + zz * Hl[j]
            z = T[2816:2816 + Z]
            head(H[1], Q, r[K_HW1], qm[6], F[r[K_HW1S]:r[K_HW1S] + Z], F[r[K_HB1V]:r[K_HB1V] + Z],
                 F[r[K_HW2]:r[K_HW2] + 2 * Z].reshape((2, Z)), F[r[K_HB2]:r[K_HB2] + 2], z, F[r[K_W]], pred, TQ, TA)

    return ens_step


class Engine:
    """Streaming ensemble, one compiled call per row: step(features112) -> weighted prediction (2,); reset()."""

    def __init__(self, npz_path):
        self.F, self.Q, self.I, self.L, nst, self.meta = flatten(npz_path)
        self.need_sin = bool(any(m["nt"] == 3 for m in self.meta))
        self.S = np.zeros(nst, np.float32)
        self.pred = np.zeros(2, np.float32)
        assert all(m["hidden"] <= 128 and m["enc_dim"] <= 256 for m in self.meta)
        self.T = np.zeros(3072, np.float32)
        self.TQ = np.zeros(512, np.int16); self.TA = np.zeros(1024, np.int32)
        self._step = _kernels()
        self.step(np.zeros(112, np.float32)); self.reset()   # compile now, not on the first scored row

    def reset(self):
        self.S[:] = 0.0

    def step(self, x):
        self._step(x, self.F, self.Q, self.I, self.L, self.S, self.pred, self.need_sin, self.T, self.TQ, self.TA)
        return self.pred


def check(npz, steps=4000, seqs=(11, 500)):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import pyarrow.parquet as pq
    from data import file_columns, read_sequence
    import numba_engine5
    pf = pq.ParquetFile(Path(__file__).resolve().parents[2] / "wnn_connectome_starterpack/datasets/valid.parquet")
    hs, ht = file_columns(pf)
    Xs = [read_sequence(pf, s, has_is_scored=hs, has_targets=ht).features[:steps].astype(np.float32) for s in seqs]
    e2, e3 = numba_engine5.Engine(npz), Engine(npz)
    d = 0.0
    for X in Xs:
        e2.reset(); e3.reset()
        for t in range(len(X)):
            d = max(d, float(np.abs(e2.step(X[t]) - e3.step(X[t])).max()))
    print(f"v6 vs v5 max abs diff over {len(Xs)} sequences x {steps} rows: {d:.3e}")
    X = Xs[0]
    for name, e in (("v5", e2), ("v6", e3), ("v5", e2), ("v6", e3)):
        e.reset(); t0 = time.perf_counter()
        for t in range(len(X)):
            e.step(X[t])
        print(f"{name}: {(time.perf_counter() - t0) / len(X) * 1e6:.1f} us/step")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"]); ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    check(a.npz)
