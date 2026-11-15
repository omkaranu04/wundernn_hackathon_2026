"""043: WP on the valid sequences with index % N == FOLD (held out from the fine-tunes), for the base model, each
fine-tune's EMA, and weight-space interpolations base + a (last - base) (WiSE-FT style), a in ALPHAS.
Usage: ft_eval.py N FOLD BASE RUN[,RUN...]"""
import sys, copy, numpy as np, torch
from teacher_preds import load_model
from dataset import ConnectomeSequenceDataset, collate
N, F, BASE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; runs = sys.argv[4].split(','); dev = torch.device('cuda')
ALPHAS = [0.25, 0.5, 0.75, 1.0]
ds = ConnectomeSequenceDataset('../../wnn_connectome_starterpack/datasets/valid.parquet')
idx = [i for i in range(len(ds)) if i % N == F]
R = 'results/checkpoints_'
base = load_model(R + BASE + ':ema.pt', dev); models = {'base ' + BASE: base}
for r in runs:
    models[r + ' ema'] = load_model(R + r + ':ema.pt', dev)
    last = torch.load(R + r + '/last.pt', map_location=dev); b = base.state_dict()
    for a in ALPHAS:
        m = copy.deepcopy(base); m.load_state_dict({k: b[k] + a * (last[k] - b[k]) if b[k].is_floating_point() else last[k] for k in b})
        models[f'{r} a={a}'] = m.eval()
P = {n: [] for n in models}; Y = []
with torch.no_grad():
    for i in range(0, len(idx), 8):
        batch = collate([ds[j] for j in idx[i:i + 8]]); msk = batch['need_prediction'] & batch['is_scored']
        Y.append(batch['targets'][msk].numpy()); x = batch['features'].to(dev)
        for n, mod in models.items():
            o = mod(x); o = o[0] if isinstance(o, tuple) else o; P[n].append(o.cpu()[msk].float().numpy())
Y = np.concatenate(Y).astype(np.float64)
def wp(p):
    r = []
    for t in range(2):
        y = np.clip(Y[:, t], -2, 2); q = np.clip(p[:, t], -2, 2); w = np.abs(y); W = w.sum()
        ym = (w*y).sum()/W; qm = (w*q).sum()/W
        r.append((w*(y-ym)*(q-qm)).sum() / np.sqrt((w*(y-ym)**2).sum() * (w*(q-qm)**2).sum()))
    return np.mean(r)
b0 = wp(np.concatenate(P['base ' + BASE]).astype(np.float64))
for n in models:
    v = wp(np.concatenate(P[n]).astype(np.float64)); print(f"fold {F}/{N} {n:48s} {v:.5f} {v - b0:+.5f}", flush=True)
