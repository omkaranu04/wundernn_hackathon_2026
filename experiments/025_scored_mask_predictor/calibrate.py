"""Fit a 2-parameter (Platt) recalibration p' = sigmoid(a*logit(p)+b) on the
HELD-OUT valid sequences (every 4th, never seen by the mask predictor) and
save it; label_train.py applies it. Ranking (AUC/AP) is unchanged by an
increasing monotone map; only the probability scale is corrected."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "shared"))
from data import file_columns, read_sequence  # noqa: E402
from mask_model import MaskPredictor, predict_sequence  # noqa: E402
a = argparse.ArgumentParser(); a.add_argument("--valid", required=True); a.add_argument("--model-dir", required=True)
a.add_argument("--hidden", type=int, default=96); a.add_argument("--layers", type=int, default=2); a.add_argument("--device", default="cuda"); args = a.parse_args()
device = torch.device(args.device); md = Path(args.model_dir)
m = MaskPredictor(hidden=args.hidden, layers=args.layers).to(device); m.load_state_dict(torch.load(md / "mask_model.pt", map_location=device)); m.eval()
pf = pq.ParquetFile(args.valid); hs, _ = file_columns(pf)
L, Y = [], []
for g in [i for i in range(pf.num_row_groups) if i % 4 == 0]:
    s = read_sequence(pf, g, has_is_scored=hs, has_targets=True)
    lg = predict_sequence(m, torch.from_numpy(s.features).to(device)).cpu().numpy()
    L.append(lg[s.need_prediction]); Y.append(s.is_scored[s.need_prediction])
L = np.concatenate(L); Y = np.concatenate(Y).astype(np.int8)
sub = np.random.default_rng(0).choice(len(L), 2_000_000, replace=False)
lr = LogisticRegression(C=1e6, max_iter=200).fit(L[sub, None], Y[sub])
A, B = float(lr.coef_[0, 0]), float(lr.intercept_[0])
sig = lambda z: 1 / (1 + np.exp(-z))
p0, p1 = sig(L), sig(A * L + B)
json.dump({"a": A, "b": B}, open(md / "calibration.json", "w"))
print(f"Platt: a={A:.4f} b={B:.4f}")
print(f"held-out rows {len(Y):,}: base rate {Y.mean():.4f} | mean pred before {p0.mean():.4f} -> after {p1.mean():.4f} | AUC {roc_auc_score(Y,p1):.4f} AP {average_precision_score(Y,p1):.4f}")
print("reliability after (bin | rows | mean predicted | actual):")
for lo, hi in zip([0,.02,.05,.1,.2,.3,.5,.7],[.02,.05,.1,.2,.3,.5,.7,1.01]):
    k = (p1 >= lo) & (p1 < hi)
    if k.sum(): print(f"  [{lo:.2f},{min(hi,1):.2f}) {int(k.sum()):>10,}  {p1[k].mean():.4f}  {Y[k].mean():.4f}")
