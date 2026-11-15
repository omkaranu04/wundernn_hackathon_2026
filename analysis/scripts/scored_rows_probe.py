"""Probe of the hidden `is_scored` mask (2026-10-09): how predictable is it from the features alone (gradient-boosting AUC/AP, no targets as inputs), and does experiment 012 correlate better on scored rows than on unscored ones? Run from the repo root. Results are recorded in EXPERIMENTS.md ("Scored-row structure")."""
import sys, numpy as np, pyarrow.parquet as pq, torch, importlib.util, json
sys.path.insert(0,'shared'); sys.path.insert(0,'wnn_connectome_starterpack'); sys.path.insert(0,'experiments/019_loss_optimizer_recipe')
from data import FEATURE_COLUMNS
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
pf=pq.ParquetFile('wnn_connectome_starterpack/datasets/valid.parquet')
def load(g):
    t=pf.read_row_group(g)
    X=np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    return X, t['need_prediction'].to_numpy().astype(bool), t['is_scored'].to_numpy().astype(bool), np.column_stack([t['t0'].to_numpy(),t['t1'].to_numpy()]).astype(np.float32)
tr_ids=list(range(0,1873,30))[:50]; te_ids=list(range(7,1873,90))[:20]
def build(ids, sub=None, rng=np.random.default_rng(0)):
    Xs=[];ys=[];As=[]
    for g in ids:
        X,need,sc,y=load(g); m=need
        # row features + simple temporal context (diff to previous step) -> mask predictor inputs, NO targets
        D=np.vstack([np.zeros((1,112),np.float32), np.diff(X,axis=0)])
        F=np.hstack([X,D])[m]; s=sc[m]; a=np.abs(y[m]).sum(1)
        if sub: k=rng.choice(len(F), sub, replace=False); F,s,a=F[k],s[k],a[k]
        Xs.append(F); ys.append(s); As.append(a)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(As)
Xtr,ytr,atr=build(tr_ids, sub=6000); Xte,yte,ate=build(te_ids)
print(f"mask predictor: train rows {len(Xtr):,} ({ytr.mean():.3f} scored), held-out rows {len(Xte):,} ({yte.mean():.3f} scored)")
clf=HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xtr,ytr)
p=clf.predict_proba(Xte)[:,1]
print(f"  features-only (112 + 112 step-diffs, no targets):  AUC={roc_auc_score(yte,p):.4f}  AP={average_precision_score(yte,p):.4f}  (base rate {yte.mean():.3f})")
# diagnostic only: how much of selection tracks target magnitude
print(f"  diagnostic: AUC of |t0|+|t1| alone vs is_scored = {roc_auc_score(yte,ate):.4f}")

# does 012 correlate better on scored rows?
spec=importlib.util.spec_from_file_location("m012","experiments/012_nonlinear_encoder/model.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg=json.load(open('experiments/012_nonlinear_encoder/checkpoints/config.json'))
m=mod.NonlinearGroupedEncoderLSTMModel(**cfg,dropout=0.0); m.load_state_dict(torch.load('experiments/012_nonlinear_encoder/checkpoints/best.pt',map_location='cpu')); m.eval()
torch.set_num_threads(8)
from eval_wp import GlobalWPAccumulator
acc={k:GlobalWPAccumulator(torch.device('cpu')) for k in ('scored','unscored','all_need')}
ids=list(range(3,1873,78))[:24]
with torch.no_grad():
    for i in range(0,len(ids),4):
        B=[load(g) for g in ids[i:i+4]]
        X=torch.from_numpy(np.stack([b[0] for b in B])); y=torch.from_numpy(np.stack([b[3] for b in B]))
        need=torch.from_numpy(np.stack([b[1] for b in B])); sc=torch.from_numpy(np.stack([b[2] for b in B]))
        pred=m(X)[0]
        acc['scored'].add(y,pred,need&sc); acc['unscored'].add(y,pred,need&~sc); acc['all_need'].add(y,pred,need)
print(f"\n012 corrected-WP by row subset, {len(ids)} valid sequences:")
for k,a in acc.items(): r=a.result(); print(f"  {k:<9} wp={r['wp']:.4f} (t0 {r['t0']:.4f}, t1 {r['t1']:.4f})")
