"""035 (2026-11-06): look-ahead CLASSIFICATION teacher for the 0.6569 tick unit: P(|y| > eps) per row and target.

Proposal (user, "zero-inflated tail classification"): train a look-ahead teacher (the la12 architecture: lean LSTM +
gated forward convs over rows t..t+12) ONLY on sequences of the 0.6569 tick unit, with binary targets
y_bin = |y| > eps (eps 0.1), loss = focal BCE ("FZLPR": alpha 0.25 on the tick class, gamma 2) + AUCM (LibAUC's
min-max square-margin AUC loss; a, b minimised, alpha maximised and kept >= 0), AdamW. Its probabilities later gate the
regression soft targets of those sequences (gate_soft.py).

Measured class balance (tick_units.py): in the 0.6569 unit ticks are the MAJORITY (70% of train rows, 65% of valid);
alpha 0.25 on the tick class therefore up-weights the minority flat class, as focal alpha is meant to.

Data: train + valid folds 1-4 (0.6569 sequences only); valid fold 0's 0.6569 sequences are held out and scored every
epoch (AUC on scored rows) - the same split as the la12 regression teacher, so fold 0 stays honest for both.
Row weights 0.1 + 0.9 x p_scored (train: mask predictor; valid: true is_scored), like every other run.
Writes checkpoint_dir/{config.json, ema.pt, model.pt}; config.json is teacher_preds.py-compatible (outputs = LOGITS).

    python train_tick_cls.py --device cuda --checkpoint-dir checkpoints_TICKCLS_la12
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dataset import ConnectomeSequenceDataset, collate
from model import build_model

D = "../../wnn_connectome_starterpack/datasets"


class FZLPRLoss(nn.Module):
    """Focal BCE as in the proposal's skeleton (alpha on the tick class, gamma focusing), row-weighted mean."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y_bin, w):
        bce = F.binary_cross_entropy_with_logits(logits, y_bin, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = y_bin * self.alpha + (1 - y_bin) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce * w).sum() / w.sum().clamp_min(1e-8)


class AUCMLoss(nn.Module):
    """LibAUC's AUC-margin (AUCM) min-max loss on scores in [0, 1], with row weights.
    min over (model, a, b), max over alpha >= 0 (alpha gets its own maximising optimiser)."""

    def __init__(self, imratio: float, margin: float = 1.0):
        super().__init__()
        self.p, self.m = imratio, margin
        self.a = nn.Parameter(torch.zeros(1)); self.b = nn.Parameter(torch.zeros(1)); self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, s, y, w):
        p, m = self.p, self.m
        W = w.sum().clamp_min(1e-8)
        pos, neg = y * w, (1 - y) * w
        return ((1 - p) * ((s - self.a) ** 2 * pos).sum() / W + p * ((s - self.b) ** 2 * neg).sum() / W
                + 2 * self.alpha * (p * (1 - p) * m + (p * s * neg).sum() / W - ((1 - p) * s * pos).sum() / W)
                - p * (1 - p) * self.alpha ** 2)


def auc(scores, labels):
    """Rank AUC (ties averaged)."""
    from scipy.stats import rankdata
    labels = labels.astype(bool); n1 = labels.sum(); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="la12")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-milestones", type=int, nargs="*", default=[9])
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--aucm-weight", type=float, default=1.0)
    ap.add_argument("--weight-floor", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.9998)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--unit", type=int, default=3, help="tick unit index (3 = 0.6569)")
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: stop each epoch / evaluation after this many batches")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)

    ut, uv = np.load("soft/units_train.npy"), np.load("soft/units_valid.npy")
    tr = ConnectomeSequenceDataset(f"{D}/train.parquet", mask_probs_path="../025_scored_mask_predictor/mask_out/mask_probs_train.npy")
    va_all = ConnectomeSequenceDataset(f"{D}/valid.parquet", true_mask_as_weight=True)
    tr_idx = [i for i in range(len(tr)) if ut[i] == a.unit]
    va_tr_idx = [i for i in range(len(va_all)) if uv[i] == a.unit and i % 5 != 0]
    va_ho_idx = [i for i in range(len(va_all)) if uv[i] == a.unit and i % 5 == 0]
    train_ds = ConcatDataset([Subset(tr, tr_idx), Subset(va_all, va_tr_idx)])
    hold_ds = Subset(ConnectomeSequenceDataset(f"{D}/valid.parquet"), va_ho_idx)
    print(f"unit {a.unit}: train {len(tr_idx)} + valid folds 1-4 {len(va_tr_idx)} sequences; held out (valid fold 0) {len(hold_ds)}", flush=True)
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    hold_loader = DataLoader(hold_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = build_model(a.arch, hidden_size=128, num_layers=2, price_dim=24, volume_dim=24, trade_dim=12, head_hidden=64,
                        dropout=0.1, aux_scored=False).to(dev)
    focal = FZLPRLoss(a.focal_alpha, a.focal_gamma)
    aucm = AUCMLoss(imratio=0.70).to(dev)  # tick share of the unit's train rows (tick_units.py)
    opt = torch.optim.AdamW(list(model.parameters()) + [aucm.a, aucm.b], lr=a.lr, weight_decay=a.weight_decay)
    opt_alpha = torch.optim.SGD([aucm.alpha], lr=a.lr, maximize=True)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, a.lr_milestones, 0.5)
    ema = copy.deepcopy(model).eval(); n_ema = 0
    ck = Path(a.checkpoint_dir); ck.mkdir(parents=True, exist_ok=True)
    json.dump({"arch": a.arch, "hidden_size": 128, "num_layers": 2, "price_dim": 24, "volume_dim": 24, "trade_dim": 12,
               "head_hidden": 64, "aux_scored": False, "bidirectional": False, "outputs": "tick logits (t0, t1)",
               "unit": a.unit, "eps": a.eps, "args": vars(a)}, open(ck / "config.json", "w"), indent=1)

    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = n = 0
        for b in loader:
            x = b["features"].to(dev); y = b["targets"].to(dev); npred = b["need_prediction"].to(dev).float()
            w = (a.weight_floor + (1 - a.weight_floor) * b["row_weight"].to(dev)) * npred
            w2 = w.unsqueeze(-1).expand_as(y)
            ybin = (y.abs() > a.eps).float()
            logits, _ = model(x)
            loss = focal(logits, ybin, w2) + a.aucm_weight * aucm(torch.sigmoid(logits), ybin, w2)
            opt.zero_grad(); opt_alpha.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt_alpha.step()
            with torch.no_grad():
                aucm.alpha.clamp_(min=0)
                n_ema += 1; d = min(a.ema_decay, (1 + n_ema) / (10 + n_ema))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            tot += loss.item(); n += 1
            if a.max_batches and n >= a.max_batches:
                break
        sched.step()
        # held-out AUC on SCORED rows of valid fold 0 (0.6569 sequences), raw and EMA weights
        res = {}
        for tag, m in (("raw", model), ("ema", ema)):
            m.eval(); S, Y = [], []
            with torch.no_grad():
                for k, b in enumerate(hold_loader):
                    if a.max_batches and k >= a.max_batches:
                        break
                    lg, _ = m(b["features"].to(dev))
                    sel = (b["need_prediction"] & b["is_scored"].bool()).numpy()
                    S.append(lg.cpu().numpy()[sel]); Y.append((b["targets"].abs() > a.eps).numpy()[sel])
            S, Y = np.concatenate(S), np.concatenate(Y)
            res[tag] = (auc(S[:, 0], Y[:, 0]), auc(S[:, 1], Y[:, 1]))
        print(f"epoch {ep}: loss {tot / n:.4f} lr {opt.param_groups[0]['lr']:.2e} alpha {aucm.alpha.item():.3f} | held-out AUC "
              f"raw t0 {res['raw'][0]:.4f} t1 {res['raw'][1]:.4f} | ema t0 {res['ema'][0]:.4f} t1 {res['ema'][1]:.4f} | {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), ck / "model.pt"); torch.save(ema.state_dict(), ck / "ema.pt")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
