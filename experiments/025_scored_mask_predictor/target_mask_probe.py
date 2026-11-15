"""Can the hidden `is_scored` flag be recovered from the TARGETS? (valid only
has both.) If yes, train.parquet - which has targets but no flag - could be
labelled (near-)exactly instead of with the feature-based predictor.

NOTE: this deliberately breaks the mask predictor's "no targets as inputs"
rule. That is safe for ONE use only: deriving per-row LOSS WEIGHTS for
training rows (the loss already uses |target| as a weight). Target-derived
quantities must never be fed to the model as inputs - they do not exist at
inference time.
"""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "shared")); sys.path.insert(0, str(HERE.parent.parent / "wnn_connectome_starterpack"))
from data import file_columns, read_sequence
pf = pq.ParquetFile(sys.argv[1]); hs, _ = file_columns(pf)
tr, te = list(range(2, 1873, 31))[:60], list(range(17, 1873, 83))[:22]

def feats(y):
    """y: (T,2) -> per-row target-derived statistics, (T,k), plus names."""
    T = len(y); z = np.zeros((1, 2), np.float32)
    dprev = np.abs(np.vstack([z, np.diff(y, axis=0)])); dnext = np.abs(np.vstack([np.diff(y, axis=0), z]))
    chg_prev = (dprev > 0).any(1); chg_next = (dnext > 0).any(1)
    # steps since the target last changed / until it next changes
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(chg_prev, idx, 0)); since = idx - last
    nxt = np.minimum.accumulate(np.where(chg_next, idx, T)[::-1])[::-1]; until = nxt - idx
    cols = [np.abs(y), dprev, dnext, (y == 0).astype(np.float32), chg_prev[:, None], chg_next[:, None],
            np.log1p(since)[:, None], np.log1p(until)[:, None]]
    names = ["|t0|", "|t1|", "|dprev t0|", "|dprev t1|", "|dnext t0|", "|dnext t1|", "t0==0", "t1==0",
             "changed vs prev row", "changes at next row", "log steps since last change", "log steps until next change"]
    return np.hstack([c.astype(np.float32) for c in cols]), names

def build(ids, window=3):
    F, W, Y, stats = [], [], [], []
    for g in ids:
        s = read_sequence(pf, g, has_is_scored=hs, has_targets=True); y = s.targets; m = s.need_prediction
        f, names = feats(y)
        # raw target window t-3..t+3 for a flexible model
        pad = np.pad(y, ((window, window), (0, 0)), mode="edge")
        w = np.hstack([pad[k:k + len(y)] for k in range(2 * window + 1)])
        F.append(f[m]); W.append(np.hstack([f, w])[m]); Y.append(s.is_scored[m])
        stats.append(((f[m][:, 8] > 0).mean()))
    return np.vstack(F), np.vstack(W), np.concatenate(Y).astype(np.int8), names, float(np.mean(stats))

F, W, Y, names, frac_chg = build(tr); Ft, Wt, Yt, _, _ = build(te)
print(f"rows: fit {len(Y):,} ({Y.mean():.3f} scored), held-out {len(Yt):,} ({Yt.mean():.3f} scored)")
print(f"fraction of rows where the target differs from the previous row: {frac_chg:.3f}")
print("\nSingle target-derived statistics vs is_scored (held-out AUC; 0.5 = nothing):")
for j, n in enumerate(names):
    a = roc_auc_score(Yt, Ft[:, j]); print(f"  {n:<30} AUC {max(a, 1 - a):.3f}")
c = Ft[:, 8] > 0
print(f"\nP(scored | target changed vs prev) = {Yt[c].mean():.3f}   P(scored | unchanged) = {Yt[~c].mean() if (~c).any() else float('nan'):.3f}   share of scored rows that changed = {c[Yt == 1].mean():.3f}")
sub = np.random.default_rng(0).choice(len(Y), min(len(Y), 600_000), replace=False)
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08).fit(W[sub], Y[sub])
p = clf.predict_proba(Wt)[:, 1]
print(f"\nBoosted trees on TARGET-ONLY inputs (statistics + raw target window t-3..t+3): held-out AUC {roc_auc_score(Yt, p):.4f}  AP {average_precision_score(Yt, p):.4f}  (base {Yt.mean():.3f})")
print("for comparison, feature-only predictors: boosted trees on one row 0.859 / 0.42; bidirectional LSTM 0.952 / 0.728")
