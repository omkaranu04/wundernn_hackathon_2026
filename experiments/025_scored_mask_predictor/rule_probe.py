"""What makes a row "scored"? Rule probe on valid (has the true is_scored flag).
For every raw feature, the AUC of several simple per-row statistics against
is_scored: the value, its magnitude, the magnitude of its change from the
previous row (causal), and to the next row (non-causal). Then a shallow
decision tree on [x, |dx_prev|, |dx_next|] to see whether a simple rule exists.
Targets are never used."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "shared")); sys.path.insert(0, str(HERE.parent.parent / "wnn_connectome_starterpack"))
from data import FEATURE_COLUMNS, file_columns, read_sequence
pf = pq.ParquetFile(sys.argv[1]); hs, _ = file_columns(pf)
tr, te = list(range(1, 1873, 37))[:50], list(range(11, 1873, 97))[:19]
def build(ids):
    X, P, N, Y = [], [], [], []
    for g in ids:
        s = read_sequence(pf, g, has_is_scored=hs, has_targets=True); x = s.features
        dp = np.abs(np.vstack([np.zeros((1, 112), np.float32), np.diff(x, axis=0)]))
        dn = np.abs(np.vstack([np.diff(x, axis=0), np.zeros((1, 112), np.float32)]))
        m = s.need_prediction
        X.append(x[m]); P.append(dp[m]); N.append(dn[m]); Y.append(s.is_scored[m])
    return np.vstack(X), np.vstack(P), np.vstack(N), np.concatenate(Y).astype(np.int8)
X, P, N, Y = build(tr); Xt, Pt, Nt, Yt = build(te)
print(f"rows: train {len(Y):,} ({Y.mean():.3f} scored), test {len(Yt):,} ({Yt.mean():.3f} scored)")
rows = []
for j, name in enumerate(FEATURE_COLUMNS):
    def auc(v):
        a = roc_auc_score(Yt, v); return max(a, 1 - a)
    rows.append((name, auc(Xt[:, j]), auc(np.abs(Xt[:, j])), auc(Pt[:, j]), auc(Nt[:, j])))
print("\nTop 12 features by best single-statistic AUC (0.5 = no information; direction ignored):")
print(f"{'feature':<8} {'value':>7} {'|value|':>8} {'|chg prev|':>11} {'|chg next|':>11}")
for r in sorted(rows, key=lambda r: -max(r[1:]))[:12]:
    print(f"{r[0]:<8} {r[1]:7.3f} {r[2]:8.3f} {r[3]:11.3f} {r[4]:11.3f}")
grp = lambda n: 'additional' if n.startswith('a') else n.split('_')[0] + '_' + ''.join(c for c in n.split('_')[1] if c.isalpha())
import collections
G = collections.defaultdict(list)
for r in rows: G[grp(r[0])].append(max(r[1:]))
print("\nBest single-statistic AUC by group (max over the group's features):")
for k, v in sorted(G.items(), key=lambda kv: -max(kv[1])): print(f"  {k:<11} max {max(v):.3f}  mean {np.mean(v):.3f}")
# fraction of rows where ANY trade feature changed vs scored
trade = [j for j, n in enumerate(FEATURE_COLUMNS) if '_dp' in n or '_dv' in n]
for label, M in (("any trade feature changed from previous row", (Pt[:, trade] > 0).any(1)), ("any trade feature nonzero", (Xt[:, trade] != 0).any(1)),
                 ("any i0 price feature changed from previous row", (Pt[:, 0:22] > 0).any(1)), ("any feature changed from previous row", (Pt > 0).any(1))):
    print(f"{label}: true on {M.mean():.3f} of rows | P(scored | true)={Yt[M].mean() if M.any() else float('nan'):.3f}  P(scored | false)={Yt[~M].mean() if (~M).any() else float('nan'):.3f}")
names = [f"{n}" for n in FEATURE_COLUMNS] + [f"|d_prev {n}|" for n in FEATURE_COLUMNS] + [f"|d_next {n}|" for n in FEATURE_COLUMNS]
for label, cols in (("causal inputs only [x, |d_prev|]", np.hstack([X, P])), ("with the next row too [x, |d_prev|, |d_next|]", np.hstack([X, P, N]))):
    tcols = np.hstack([Xt, Pt]) if cols.shape[1] == 224 else np.hstack([Xt, Pt, Nt])
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2000, random_state=0).fit(cols, Y)
    print(f"\nDepth-4 tree, {label}: held-out AUC {roc_auc_score(Yt, tree.predict_proba(tcols)[:, 1]):.3f}")
    imp = sorted(zip(tree.feature_importances_, names[:cols.shape[1]]), reverse=True)[:5]
    print("  top splits:", ", ".join(f"{n} ({i:.2f})" for i, n in imp if i > 0))
