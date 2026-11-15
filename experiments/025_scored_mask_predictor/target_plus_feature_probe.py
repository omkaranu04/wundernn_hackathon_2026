"""Do target-derived statistics add anything to the FEATURES for predicting
is_scored? Same boosted-tree model, one row at a time, with and without the
target statistics. (Offline-only question: this would only ever be used to
make loss weights for training rows.)"""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "shared")); sys.path.insert(0, str(HERE.parent.parent / "wnn_connectome_starterpack"))
from data import file_columns, read_sequence
from target_mask_probe import feats
pf = pq.ParquetFile(sys.argv[1]); hs, _ = file_columns(pf)
tr, te = list(range(2, 1873, 31))[:60], list(range(17, 1873, 83))[:22]
rng = np.random.default_rng(0)
def build(ids, sub=None):
    A, B, Y = [], [], []
    for g in ids:
        s = read_sequence(pf, g, has_is_scored=hs, has_targets=True); x = s.features; m = s.need_prediction
        d = np.vstack([np.zeros((1, 112), np.float32), np.diff(x, axis=0)])
        f = np.hstack([x, d])[m]; t = feats(s.targets)[0][m]; y = s.is_scored[m]
        if sub:
            k = rng.choice(len(f), sub, replace=False); f, t, y = f[k], t[k], y[k]
        A.append(f); B.append(np.hstack([f, t])); Y.append(y)
    return np.vstack(A), np.vstack(B), np.concatenate(Y).astype(np.int8)
A, B, Y = build(tr, sub=6000); At, Bt, Yt = build(te)
for label, X, Xt in (("features only (112 + step-diffs)", A, At), ("features + target statistics", B, Bt)):
    p = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(X, Y).predict_proba(Xt)[:, 1]
    print(f"{label:<36} held-out AUC {roc_auc_score(Yt, p):.4f}  AP {average_precision_score(Yt, p):.4f}")
