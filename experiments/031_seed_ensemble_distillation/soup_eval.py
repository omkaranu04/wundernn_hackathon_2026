"""043: offline model-soup test. Per fold: base, each fine-tune's last.pt, uniform soups of their last.pt, and
WiSE-style a=0.5 versions; scored on the held-out half of valid. Usage: soup_eval.py BASE"""
import sys, copy, itertools, numpy as np, torch
from teacher_preds import load_model
from dataset import ConnectomeSequenceDataset, collate
BASE = sys.argv[1]; dev = torch.device('cuda'); R = 'results/checkpoints_'
ds = ConnectomeSequenceDataset('../../wnn_connectome_starterpack/datasets/valid.parquet')
base = load_model(R + BASE + ':ema.pt', dev); b = base.state_dict()
def mk(sd):
    m = copy.deepcopy(base); m.load_state_dict(sd); return m.eval()
def mix(sds, a):
    return {k: b[k] + a * (sum(s[k] for s in sds) / len(sds) - b[k]) if b[k].is_floating_point() else sds[0][k] for k in b}
for F in (0, 1):
    tags = ['r0', 'lr3', 'ep12', 'd05']
    L = {t: torch.load(f'{R}S43_{BASE}_{t}_f{F}/last.pt', map_location=dev) for t in tags}
    models = {'base': base}
    for combo in [('lr3',), ('r0', 'lr3'), ('r0', 'lr3', 'ep12'), ('lr3', 'ep12'), ('r0', 'lr3', 'ep12', 'd05')]:
        for a in (0.5, 0.75, 1.0):
            models[f"soup[{'+'.join(combo)}] a={a}"] = mk(mix([L[t] for t in combo], a))
    idx = [i for i in range(len(ds)) if i % 2 == F]
    P = {n: [] for n in models}; Y = []
    with torch.no_grad():
        for i in range(0, len(idx), 8):
            bt = collate([ds[j] for j in idx[i:i + 8]]); msk = bt['need_prediction'] & bt['is_scored']
            Y.append(bt['targets'][msk].numpy()); x = bt['features'].to(dev)
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
    b0 = wp(np.concatenate(P['base']).astype(np.float64))
    for n in models:
        v = wp(np.concatenate(P[n]).astype(np.float64)); print(f"fold {F} {n:40s} {v:.5f} {v - b0:+.5f}", flush=True)
