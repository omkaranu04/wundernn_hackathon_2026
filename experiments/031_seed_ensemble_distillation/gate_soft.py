"""035 (2026-11-06): gate regression soft targets with the tick classifier's P(tick) on 0.6569-unit sequences.

    gated = P(tick) x mu                       (the proposal: "SoftTarget_new = P(tick) x mu")
    mode "seq": then rescale each sequence/target so its RMS over prediction rows equals mu's (mu is already an
                expectation that includes the no-tick case, so the product shrinks these sequences relative to the
                other units - and the metric is ONE correlation pooled over all sequences)
Other units are copied unchanged.

  write:  python gate_soft.py write BASE.npy PTICK_LOGITS.npy UNITS.npy MODE OUT.npy
  eval:   python gate_soft.py eval  MU_valid.npy PTICK_LOGITS_valid.npy   (honest: valid fold 0 scored rows, global WP
          and the 0.6569 unit's own WP, for mu / gated raw / gated seq-rescaled; MU must be held out on fold 0, e.g.
          valid_LA12.npy from the la12 teacher; prints the better mode on the last line as "BEST <mode>")
"""
import sys

import numpy as np

UNIT = 3
W0 = 99  # first prediction row


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gate(mu, logit, mode):
    g = sigmoid(logit.astype(np.float32)) * mu
    if mode == "seq":
        r_mu = np.sqrt((mu[W0:] ** 2).mean(0)); r_g = np.sqrt((g[W0:] ** 2).mean(0))
        g = g * (r_mu / np.maximum(r_g, 1e-8))
    return g


def wp(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    out = []
    for t in range(2):
        ww = w[:, t]; W = ww.sum(); my = (ww * y[:, t]).sum() / W; mp = (ww * p[:, t]).sum() / W
        c = (ww * (y[:, t] - my) * (p[:, t] - mp)).sum(); vy = (ww * (y[:, t] - my) ** 2).sum(); vp = (ww * (p[:, t] - mp) ** 2).sum()
        out.append(c / np.sqrt(vy * vp))
    return float(np.mean(out))


def write(base, logits, units, mode, out):
    A = np.load(base, mmap_mode="r"); L = np.load(logits, mmap_mode="r"); U = np.load(units)
    assert A.shape == L.shape and A.shape[0] == len(U), (A.shape, L.shape, len(U))
    O = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=A.shape)
    for i in range(A.shape[0]):
        mu = A[i].astype(np.float32)
        O[i] = (gate(mu, L[i], mode) if U[i] == UNIT else mu).astype(np.float16)
    O.flush()
    print(f"wrote {out}: {A.shape}, mode {mode}, gated {(U == UNIT).sum()} of {len(U)} sequences")


def evaluate(mu_path, logit_path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile("../../wnn_connectome_starterpack/datasets/valid.parquet")
    M = np.load(mu_path, mmap_mode="r"); L = np.load(logit_path, mmap_mode="r"); U = np.load("soft/units_valid.npy")
    Y, P, Uu = [], {m: [] for m in ("mu", "raw", "seq")}, []
    for i in range(0, pf.num_row_groups, 5):  # fold 0
        t = pf.read_row_group(i, columns=["need_prediction", "is_scored", "t0", "t1"])
        sel = t.column("need_prediction").to_numpy(zero_copy_only=False).astype(bool) & t.column("is_scored").to_numpy(zero_copy_only=False).astype(bool)
        Y.append(np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1)[sel]); Uu.append(np.full(sel.sum(), U[i]))
        mu = M[i].astype(np.float32)
        P["mu"].append(mu[sel])
        for m in ("raw", "seq"):
            P[m].append((gate(mu, L[i], m) if U[i] == UNIT else mu)[sel])
    Y = np.concatenate(Y); Uu = np.concatenate(Uu); P = {m: np.concatenate(v) for m, v in P.items()}
    u = Uu == UNIT
    tick = np.abs(Y[u]) > 0.1
    print(f"valid fold 0 scored rows: {len(Y):,} ({u.sum():,} in the 0.6569 unit, tick share {tick.mean():.1%})")
    res = {}
    for m in ("mu", "raw", "seq"):
        res[m] = wp(Y, P[m])
        print(f"{m:4s}: global WP {res[m]:.6f} | 0.6569-unit WP {wp(Y[u], P[m][u]):.6f} | other units WP {wp(Y[~u], P[m][~u]):.6f}")
    best = max(("raw", "seq"), key=lambda m: res[m])
    print(f"gain vs mu: raw {res['raw'] - res['mu']:+.6f}, seq {res['seq'] - res['mu']:+.6f}")
    print(f"BEST {best} {res[best] - res['mu']:+.6f}")


if __name__ == "__main__":
    if sys.argv[1] == "write":
        write(*sys.argv[2:7])
    elif sys.argv[1] == "eval":
        evaluate(sys.argv[2], sys.argv[3])
