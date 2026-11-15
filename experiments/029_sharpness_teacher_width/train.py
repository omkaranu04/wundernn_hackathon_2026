"""Training script for experiment 029 (028's script + `--weight-power`,
`--bidirectional`; `--hidden-size` was already there).

028's docstring follows.

Training script for experiment 028: the lean cos-expanded model (027) with
any combination of four things, each behind its own switch:

  --mask-probs PATH   weight every training row's loss by its predicted
                      scored-probability (026's idea), in BOTH loss terms.
  --ema-decay D       keep an exponential moving average of the weights and
                      score it on valid every epoch next to the raw weights
                      (0 = off). Costs nothing at inference: still one model.
  --aux-weight L      add L * BCE(aux_logit, scored-probability) from an
                      auxiliary head (training only; never exported).
  --extra-train PATH  append a labelled file (valid.parquet) to the training
                      set, weighted by its TRUE is_scored flag. Requires
                      --no-validation: valid can no longer measure anything.
  --no-validation     fixed schedule: no valid pass, LR halves at
                      --lr-milestones, runs exactly --epochs, saves last.pt
                      and ema.pt.

Loss form, optimiser, warmup, plateau decay, early stopping and WP-based
selection are otherwise exactly experiment 026/027's:
    0.5 * weighted MSE + 0.5 * (-per-sequence weighted Pearson),
    weights = row_weight * |clip(target, +-2)|.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

_NUM_THREADS = os.environ.get("TRAIN_NUM_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _NUM_THREADS)

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

torch.set_num_threads(int(_NUM_THREADS))

from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import LeanCosGroupedEncoderLSTMModel
from dataset import ConnectomeSequenceDataset, collate
from eval_wp import GlobalWPAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from losses import weighted_pearson_loss, METRIC_CLIP  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mse-weight", type=float, default=0.5)
    p.add_argument("--mask-probs", default=None)
    p.add_argument("--weight-floor", type=float, default=0.0)
    p.add_argument("--weight-power", type=float, default=1.0, help="row weight = floor + (1-floor) * p**power; >1 sharpens toward confident rows")
    p.add_argument("--bidirectional", action="store_true", help="hindsight TEACHER: LSTM sees the future; cannot be shipped")
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--aux-weight", type=float, default=0.0)
    p.add_argument("--extra-train", default=None)
    p.add_argument("--no-validation", action="store_true")
    p.add_argument("--lr-milestones", type=int, nargs="*", default=[], help="epochs at which LR is multiplied by --lr-factor (fixed-schedule mode)")
    p.add_argument("--save-epochs", action="store_true", help="also save epoch_NN.pt every epoch (for offline checkpoint averaging)")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--warmup-start-factor", type=float, default=0.1)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=2)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--price-dim", type=int, default=24)
    p.add_argument("--volume-dim", type=int, default=24)
    p.add_argument("--trade-dim", type=int, default=12)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-valid-batches", type=int, default=0)
    a = p.parse_args()
    if a.extra_train and not a.no_validation:
        p.error("--extra-train puts validation sequences in the training set; it requires --no-validation")
    if a.aux_weight > 0 and not a.mask_probs:
        p.error("--aux-weight needs --mask-probs (the auxiliary target is the scored-probability)")
    return a


def compute_loss(args, predictions, targets, need_prediction, row_weight=None) -> torch.Tensor:
    """026's loss: 0.5*MSE + 0.5*(-per-sequence Pearson), every row multiplied
    by row_weight in BOTH terms on top of the |clipped target| weights.
    row_weight=None reproduces the unweighted loss of earlier experiments."""
    rw = need_prediction.float() if row_weight is None else need_prediction.float() * row_weight
    y = targets.clamp(-METRIC_CLIP, METRIC_CLIP); p = predictions.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs() * rw.unsqueeze(-1)
    mse = (w * (y - p) ** 2).sum() / w.sum().clamp_min(1e-8)
    if args.mse_weight >= 1.0:
        return mse
    per_seq = []
    for b in range(predictions.shape[0]):
        m = need_prediction[b] & (rw[b] > 0)
        if m.sum() < 2:
            continue
        tgt = targets[b, m].T
        wb = tgt.clamp(-METRIC_CLIP, METRIC_CLIP).abs() * rw[b, m].unsqueeze(0)
        if wb.sum() <= 1e-6:
            continue
        per_seq.append(weighted_pearson_loss(tgt, predictions[b, m].T, weight=wb))
    pearson = torch.stack(per_seq).mean()
    return args.mse_weight * mse + (1.0 - args.mse_weight) * pearson


class EMA:
    """Exponential moving average of all floating-point entries of a state_dict,
    with the usual warm-up of the decay so early steps are not dominated by
    the random initialisation."""

    def __init__(self, model, decay: float):
        self.decay, self.n = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def copy_to(self, model):
        sd = model.state_dict()
        sd.update({k: v.clone() for k, v in self.shadow.items()})
        model.load_state_dict(sd)


def train_one_epoch(args, model, loader, optimizer, device, epoch, ema):
    model.train()
    tot, tot_aux, tot_g, n = 0.0, 0.0, 0.0, 0
    progress = tqdm(loader, desc=f"epoch {epoch} [train]", unit="batch", leave=False)
    for i, batch in enumerate(progress):
        if args.max_train_batches and i >= args.max_train_batches:
            break
        features = batch["features"].to(device); targets = batch["targets"].to(device)
        need = batch["need_prediction"].to(device); p_scored = batch["row_weight"].to(device)
        row_weight = args.weight_floor + (1.0 - args.weight_floor) * p_scored.clamp(0, 1) ** args.weight_power
        optimizer.zero_grad()
        predictions, _, aux = model(features, return_aux=True)
        loss = compute_loss(args, predictions, targets, need, row_weight)
        if args.aux_weight > 0:
            aux_loss = F.binary_cross_entropy_with_logits(aux[need], p_scored[need].clamp(0, 1))
            loss = loss + args.aux_weight * aux_loss
            tot_aux += aux_loss.item()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()
        if ema is not None:
            ema.update(model)
        tot += loss.item(); tot_g += float(gnorm); n += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    n = max(n, 1)
    return tot / n, tot_g / n, tot_aux / n


@torch.no_grad()
def evaluate(args, model, loader, device, epoch, tag="valid", aux_batches=80):
    """Unweighted proxy loss + corrected global WP (true scored mask); if the
    model has an auxiliary head, also its AUC/AP against the TRUE is_scored
    flag on the first `aux_batches` batches (a causal mask predictor's quality)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    tot, n = 0.0, 0
    acc = GlobalWPAccumulator(device)
    al, ay = [], []
    for i, batch in enumerate(tqdm(loader, desc=f"epoch {epoch} [{tag}]", unit="batch", leave=False)):
        if args.max_valid_batches and i >= args.max_valid_batches:
            break
        features = batch["features"].to(device); targets = batch["targets"].to(device)
        need = batch["need_prediction"].to(device)
        scored = (batch["need_prediction"] & batch["is_scored"]).to(device)
        predictions, _, aux = model(features, return_aux=True)
        tot += compute_loss(args, predictions, targets, need).item(); n += 1
        acc.add(targets, predictions, scored)
        if aux is not None and i < aux_batches:
            al.append(aux[need].float().cpu().numpy()); ay.append(batch["is_scored"][batch["need_prediction"]].numpy())
    r = acc.result()
    if al:
        l, y = np.concatenate(al), np.concatenate(ay).astype(np.int8)
        r["aux_auc"], r["aux_ap"] = float(roc_auc_score(y, l)), float(average_precision_score(y, l))
    return tot / max(n, 1), r


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_ds = ConnectomeSequenceDataset(args.train, mask_probs_path=args.mask_probs)
    if args.extra_train:
        train_ds = ConcatDataset([train_ds, ConnectomeSequenceDataset(args.extra_train, true_mask_as_weight=True)])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = None if args.no_validation else DataLoader(ConnectomeSequenceDataset(args.valid), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    aux_on = args.aux_weight > 0
    model = LeanCosGroupedEncoderLSTMModel(
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        price_dim=args.price_dim, volume_dim=args.volume_dim, trade_dim=args.trade_dim,
        head_hidden=args.head_hidden, dropout=args.dropout, aux_scored=aux_on, bidirectional=args.bidirectional,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    ema_model = copy.deepcopy(model) if ema is not None else None

    ckpt = Path(args.checkpoint_dir); ckpt.mkdir(parents=True, exist_ok=True)
    with open(ckpt / "config.json", "w") as f:
        json.dump({"hidden_size": args.hidden_size, "num_layers": args.num_layers, "price_dim": args.price_dim,
                   "volume_dim": args.volume_dim, "trade_dim": args.trade_dim, "head_hidden": args.head_hidden,
                   "aux_scored": aux_on, "bidirectional": args.bidirectional,
                   "recipe": {k: v for k, v in vars(args).items() if k not in ("train", "valid")}}, f, indent=2)
    print(f"train sequences: {len(train_ds)} | weights: {'scored-prob' if args.mask_probs else 'none'} (power {args.weight_power}, floor {args.weight_floor}) | ema: {args.ema_decay} | aux: {args.aux_weight} | hidden: {args.hidden_size} | bidirectional: {args.bidirectional} | validation: {not args.no_validation}", flush=True)

    warm = max(0, args.warmup_epochs); warm_lr0 = args.lr * args.warmup_start_factor
    best, best_ema, stale, stale_ema = float("-inf"), float("-inf"), 0, 0
    for epoch in tqdm(range(args.epochs), desc="epochs", unit="epoch"):
        if epoch < warm:
            for g in optimizer.param_groups:
                g["lr"] = warm_lr0 + (epoch + 1) / warm * (args.lr - warm_lr0)
        elif args.no_validation and epoch in args.lr_milestones:
            for g in optimizer.param_groups:
                g["lr"] = max(g["lr"] * args.lr_factor, args.min_lr)

        train_loss, gnorm, aux_loss = train_one_epoch(args, model, train_loader, optimizer, device, epoch, ema)
        lr = optimizer.param_groups[0]["lr"]
        tag = " [warmup]" if epoch < warm else ""
        torch.save(model.state_dict(), ckpt / "last.pt")
        if args.save_epochs:
            torch.save(model.state_dict(), ckpt / f"epoch_{epoch:02d}.pt")
        if ema is not None:
            ema.copy_to(ema_model)
            torch.save(ema_model.state_dict(), ckpt / "ema.pt")

        if args.no_validation:
            tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f} lr={lr:.2e} gnorm={gnorm:.3f} aux_loss={aux_loss:.4f}{tag} [no validation]")
            continue

        valid_loss, wp = evaluate(args, model, valid_loader, device, epoch)
        extra = ""
        if "aux_auc" in wp:
            extra += f" aux_auc={wp['aux_auc']:.4f} aux_ap={wp['aux_ap']:.4f}"
        ema_wp = None
        if ema is not None:
            _, r = evaluate(args, ema_model, valid_loader, device, epoch, tag="valid-ema", aux_batches=0)
            ema_wp = r["wp"]; extra += f" ema_wp={ema_wp:.6f}"
        if epoch >= warm:
            scheduler.step(wp["wp"])
            lr = optimizer.param_groups[0]["lr"]
        # keep the "epoch N: train_loss= valid_loss= lr=" prefix (shared/plot_loss_curves.py regex)
        tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} lr={lr:.2e} "
                   f"valid_wp={wp['wp']:.6f} t0={wp['t0']:.6f} t1={wp['t1']:.6f} gnorm={gnorm:.3f}{extra}{tag}")

        if wp["wp"] > best + 1e-6:
            best, stale = wp["wp"], 0
            torch.save(model.state_dict(), ckpt / "best.pt")
        else:
            stale += 1
        if ema_wp is not None:
            if ema_wp > best_ema + 1e-6:
                best_ema, stale_ema = ema_wp, 0
                torch.save(ema_model.state_dict(), ckpt / "best_ema.pt")
            else:
                stale_ema += 1
        # stop only when BOTH the raw and the averaged weights have stopped improving
        if stale >= args.early_stop_patience and (ema is None or stale_ema >= args.early_stop_patience):
            tqdm.write(f"early stopping at epoch {epoch}: best raw wp={best:.6f}" + (f", best ema wp={best_ema:.6f}" if ema is not None else ""))
            break

    print(f"done. best raw wp={best:.6f}" + (f" best ema wp={best_ema:.6f}" if ema is not None else ""))


if __name__ == "__main__":
    main()
