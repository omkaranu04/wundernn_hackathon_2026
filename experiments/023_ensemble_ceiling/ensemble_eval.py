"""Offline ensemble evaluation from stored predictions (extract_preds.py).

All WP numbers use the corrected global metric computed on the scored rows
only (clip +-2, weights |clipped target|, one pooled correlation per target,
mean of the two), in float64. Ensembles average RAW outputs then the metric
clips the result.

Optimism control: any selection or weight-fitting is CROSS-FITTED across
sequences (fit on even sequences, score on odd, and vice versa; reported
number = score on the held-out half pooled), so choosing members/weights on
valid doesn't inflate the result. Simple averages of pre-specified sets
involve no fitting. (All models were already early-stopped on valid, a
mild optimism shared by every row, including single models.)
"""
from __future__ import annotations

import argparse, itertools, json
from pathlib import Path

import numpy as np

CLIP = 2.0


def wp(y, p):
    """y,p: (N,2) -> dict(t0,t1,wp), pooled weighted Pearson per target."""
    y = np.clip(y, -CLIP, CLIP).astype(np.float64); p = np.clip(p, -CLIP, CLIP).astype(np.float64)
    w = np.abs(y); W = w.sum(0)
    ym = (w * y).sum(0) / W; pm = (w * p).sum(0) / W
    cov = (w * (y - ym) * (p - pm)).sum(0) / W
    vy = (w * (y - ym) ** 2).sum(0) / W; vp = (w * (p - pm) ** 2).sum(0) / W
    c = cov / np.sqrt(vy * vp)
    return {"t0": float(c[0]), "t1": float(c[1]), "wp": float(c.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.preds)
    meta = np.load(d / "meta.npz"); Y = meta["targets"]; seq = meta["seq"]
    names = sorted(p.stem[5:] for p in d.glob("pred_*.npy"))
    P = {n: np.load(d / f"pred_{n}.npy").astype(np.float64) for n in names}
    even, odd = (seq % 2 == 0), (seq % 2 == 1)
    res = {"n_rows": int(len(Y)), "singles": {}}

    print(f"{len(Y):,} scored rows, {len(np.unique(seq))} sequences, {len(names)} models\n")
    print("SINGLE MODELS (corrected global WP on valid):")
    for n in sorted(names, key=lambda n: -wp(Y, P[n])["wp"]):
        r = wp(Y, P[n]); res["singles"][n] = r
        print(f"  {n:<16} wp={r['wp']:.5f}  t0={r['t0']:.5f}  t1={r['t1']:.5f}")
    best_single = max(res["singles"].items(), key=lambda kv: kv[1]["wp"])
    print(f"\nbest single: {best_single[0]} = {best_single[1]['wp']:.5f}")

    def avg(members): return np.mean([P[m] for m in members], axis=0)

    sets = {
        "012 + 017": ["012", "017"],
        "012 + 004": ["012", "004"],
        "012 + 016": ["012", "016"],
        "012 + 018": ["012", "018"],
        "012 + 017 + 018": ["012", "017", "018"],
        "nonlinear family (012,017,018,015,016)": ["012", "017", "018", "015", "016"],
        "grouped family (004,010,019_control,019_pooled50,019_pooled60,022_arm2)": ["004", "010", "019_control", "019_pooled50", "019_pooled60", "022_arm2"],
        "raw-LSTM family (002,003,005,008,009,006)": ["002", "003", "005", "008", "009", "006"],
        "ALL models": names,
    }
    print("\nSIMPLE AVERAGES of pre-specified sets (no fitting):")
    res["sets"] = {}
    for label, mem in sets.items():
        mem = [m for m in mem if m in P]
        r = wp(Y, avg(mem)); res["sets"][label] = {**r, "members": mem}
        print(f"  {r['wp']:.5f} ({r['wp']-best_single[1]['wp']:+.5f} vs best single)  {label}  [n={len(mem)}]")

    # all pairs
    pairs = sorted(((wp(Y, avg(c))["wp"], c) for c in itertools.combinations(names, 2)), reverse=True)
    print("\nTOP 8 PAIRS (equal weight; chosen on valid, so mildly optimistic):")
    for s, c in pairs[:8]:
        print(f"  {s:.5f} ({s-best_single[1]['wp']:+.5f})  {c[0]} + {c[1]}")
    res["top_pairs"] = [{"wp": s, "members": list(c)} for s, c in pairs[:8]]

    # cross-fitted greedy forward selection (with replacement allowed -> implicit weights)
    def greedy(mask, steps=8):
        chosen = []
        cur = None
        for _ in range(steps):
            best = None
            for n in names:
                cand = chosen + [n]
                s = wp(Y[mask], np.mean([P[m][mask] for m in cand], axis=0))["wp"]
                if best is None or s > best[0]: best = (s, n)
            chosen.append(best[1])
        return chosen
    print("\nCROSS-FITTED greedy selection (pick up to 8 members on one half of the sequences, score on the other):")
    held_p = np.zeros_like(Y); picks = {}
    for tr, te, tag in ((even, odd, "fit even -> score odd"), (odd, even, "fit odd -> score even")):
        ch = greedy(tr); picks[tag] = ch
        held_p[te] = np.mean([P[m][te] for m in ch], axis=0)
        print(f"  {tag}: {ch}")
    r = wp(Y, held_p)
    single_held = wp(Y, P[best_single[0]])  # for reference
    print(f"  held-out pooled WP = {r['wp']:.5f} ({r['wp']-best_single[1]['wp']:+.5f} vs best single {best_single[0]})")
    res["crossfit_greedy"] = {"wp": r["wp"], "picks": picks}

    # cross-fitted stacking: per-target non-negative-ish least squares on the weighted problem
    def fit_w(mask, t, cols):
        w = np.abs(np.clip(Y[mask, t], -CLIP, CLIP))
        X = np.stack([np.clip(P[m][mask, t], -CLIP, CLIP) for m in cols], axis=1)
        sw = np.sqrt(w)[:, None]
        lam = 1e-3 * np.trace((X * sw).T @ (X * sw)) / len(cols)
        A = (X * sw).T @ (X * sw) + lam * np.eye(len(cols))
        b = (X * sw).T @ (np.clip(Y[mask, t], -CLIP, CLIP) * sw[:, 0])
        return np.linalg.solve(A, b)
    print("\nCROSS-FITTED weighted ridge stacking over ALL models (weights fit on one half, scored on the other):")
    held = np.zeros_like(Y)
    for tr, te in ((even, odd), (odd, even)):
        for t in (0, 1):
            wts = fit_w(tr, t, names)
            held[te, t] = np.stack([P[m][te, t] for m in names], axis=1) @ wts
    r = wp(Y, held)
    print(f"  held-out pooled WP = {r['wp']:.5f} ({r['wp']-best_single[1]['wp']:+.5f} vs best single)  t0={r['t0']:.5f} t1={r['t1']:.5f}")
    res["crossfit_stacking_all"] = r

    # diversity: mean pairwise correlation of predictions (t0) among a few key models
    key = [n for n in ("004", "012", "017", "018", "016", "014", "001_d32", "002") if n in P]
    C = np.corrcoef(np.stack([np.clip(P[n][:, 0], -CLIP, CLIP) for n in key]))
    print("\nPREDICTION CORRELATION (t0) between key models:")
    print("          " + " ".join(f"{n[:7]:>8}" for n in key))
    for i, n in enumerate(key):
        print(f"  {n[:7]:<8}" + " ".join(f"{C[i,j]:8.3f}" for j in range(len(key))))
    res["corr_key_t0"] = {"models": key, "matrix": C.tolist()}
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
