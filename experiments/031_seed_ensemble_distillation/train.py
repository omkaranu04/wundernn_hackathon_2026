"""Training script for experiment 031: 030/029's script plus DISTILLATION from a
teacher ensemble.

  --soft-targets PATH        mean teacher prediction for every row of --train
                             (float16 (n_seq, 20000, 2), from teacher_preds.py)
  --extra-soft-targets PATH  the same for --extra-train
  --distill-weight A         loss = (1-A) * L(prediction, y) + A * L(prediction, teacher mean)
                             L is the usual 0.5 MSE + 0.5 per-sequence Pearson in both terms; the row
                             weights (scored-probability x |clip(y)|) ALWAYS come from the true y, so
                             the emphasis still matches the metric. A=0 is exactly 030's training.
  --distill-anneal-start E --distill-final A_f
                             teacher annealing (BAM!): from epoch E the distill weight moves linearly
                             from A to A_f, reached in the last epoch, so the student ends on true labels.

Valid-only fine-tune stage (2026-10-30, a rival's +0.007 public step):
  --init-from CKPT_DIR:WEIGHTS  start from a trained model (its config.json sets the widths)
  --train-true-mask            weight --train rows by their TRUE is_scored (with --weight-floor 0.1:
                               scored 1, unscored 0.1); --train is then valid.parquet
  --blend-targets              distil by TARGET mixing: one loss on (1-A)*y + A*teacher mean
                               (instead of mixing two losses)
  --valid-fold F               score only the --valid sequences with index % --n-folds == F
                               (with --exclude-fold F on the same file: an honest held-out fold)
  --extra-exclude-fold F       drop that fold of --extra-train too; then --extra-train may be combined with
                               validation on --valid-fold F (the held-out fold stays unseen)
  --weight-decay W / --clip-grad C   AdamW / gradient-norm clipping

Weight soups (2026-10-31): branch one run into several and average their weights.
  --save-state                 also write state.pt after every epoch: weights, optimizer, EMA, epoch, RNG
  --resume-state PATH          continue from such a state.pt at the next epoch (same schedule; --seed then
                               changes only the data order / dropout of THIS branch). soup.py averages branches.

030's docstring (029's script, which is 028's + `--weight-power`, `--bidirectional`):

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
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm.auto import tqdm

torch.set_num_threads(int(_NUM_THREADS))

from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import LeanCosGroupedEncoderLSTMModel, build_model
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
    p.add_argument("--soft-targets", default=None, help="teacher-ensemble mean predictions for --train (float16 (n_seq, T, 2))")
    p.add_argument("--extra-soft-targets", default=None, help="the same for --extra-train")
    p.add_argument("--distill-weight", type=float, default=0.0, help="A: loss = (1-A)*L(y) + A*L(teacher mean); 0 = off")
    p.add_argument("--distill-anneal-start", type=int, default=None,
                   help="teacher annealing: from this epoch on, move the distill weight linearly from --distill-weight "
                        "to --distill-final, reaching it in the last epoch (the student ends on the true labels)")
    p.add_argument("--distill-final", type=float, default=0.0)
    p.add_argument("--pearson-seq-weight", choices=["equal", "mass"], default="equal",
                   help="how the per-sequence Pearson terms are averaged: equally (as always), or by each sequence's weight mass "
                        "sum(row_weight * |clip y|), which is how much the sequence counts in the pooled metric")
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
    p.add_argument("--arch", choices=["lean", "lean_gru", "joint_pair", "lean_dist", "fusion_gru", "fusion_gru_sin", "full_lstm", "full_gru", "leanx_i", "leanx_r", "leanx_p", "leanx_irp", "nl_lstm", "nl_gru", "la6", "la12", "la25", "la50", "fla12", "fla25", "fla50", "ela4", "ela6", "ela8"], default="lean",
                   help="lean = the exportable lean model (default); nl_lstm / nl_gru = experiment 012's 5-way nonlinear architecture with an LSTM / GRU core (round-4 teachers only)")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--price-dim", type=int, default=24)
    p.add_argument("--volume-dim", type=int, default=24)
    p.add_argument("--trade-dim", type=int, default=12)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-fold", type=int, default=None,
                   help="out-of-fold teachers: leave out the --train sequences with index %% --n-folds == this fold")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--init-from", default=None, help="CKPT_DIR:WEIGHTS of a trained model to fine-tune")
    p.add_argument("--train-true-mask", action="store_true", help="weight --train rows by its TRUE is_scored flag")
    p.add_argument("--blend-targets", action="store_true", help="distil on the target (1-A)*y + A*teacher instead of mixing losses")
    p.add_argument("--dist-ce", type=float, default=0.0, help="--arch lean_dist: weight of the cross-entropy of the head's distribution on the TRUE labels (HL-Gauss)")
    p.add_argument("--dist-sigma", type=float, default=0.05, help="--arch lean_dist: HL-Gauss smoothing (in target units)")
    p.add_argument("--joint-lambda", type=float, default=0.5, help="--arch joint_pair: weight of the loss on the members' MEAN")
    p.add_argument("--soft-targets-b", default=None, help="--arch joint_pair: member b's soft targets for --train")
    p.add_argument("--extra-soft-targets-b", default=None, help="--arch joint_pair: member b's soft targets for --extra-train")
    p.add_argument("--extra-repeat", type=int, default=1, help="show --extra-train K times per epoch (oversample the valid regime)")
    p.add_argument("--extra-exclude-fold", type=int, default=None, help="leave out the --extra-train sequences with index %% --n-folds == this fold")
    p.add_argument("--valid-fold", type=int, default=None, help="score only --valid sequences with index %% --n-folds == this fold")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad", type=float, default=0.0)
    # 036 (2026-11-06, user proposal "orthogonal representation subspaces"): penalise redundancy between the group
    # encoders' outputs (training only - forward hooks on each GroupEncoder's ReLU; the exported graph is unchanged)
    p.add_argument("--ortho-weight", type=float, default=0.0, help="weight of the group-redundancy penalty (0 = off)")
    p.add_argument("--ortho-mode", choices=["decor", "cosine"], default="decor",
                   help="decor = mean squared cross-CORRELATION between every pair of groups over rows (permutation-invariant, any dims); "
                        "cosine = the proposal's literal |cos| of sequence-mean-pooled embeddings (equal-dim pairs only)")
    p.add_argument("--ortho-rows", type=int, default=8192, help="decor: rows sampled per batch for the correlation estimate")
    p.add_argument("--cosine-restarts", type=int, default=0, help="T0 > 0: after warmup, cosine annealing with warm restarts every T0 epochs (per batch), lr -> --min-lr")
    p.add_argument("--save-state", action="store_true", help="write state.pt (weights, optimizer, EMA, epoch) after every epoch")
    p.add_argument("--resume-state", default=None, help="continue from a state.pt at its next epoch (a soup branch)")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-valid-batches", type=int, default=0)
    a = p.parse_args()
    if a.extra_train and not a.no_validation and not (a.valid_fold is not None and a.extra_exclude_fold == a.valid_fold):
        p.error("--extra-train puts validation sequences in the training set; it requires --no-validation")
    if a.distill_weight > 0 and not a.soft_targets:
        p.error("--distill-weight needs --soft-targets")
    if a.distill_weight > 0 and a.extra_train and not a.extra_soft_targets:
        p.error("--distill-weight with --extra-train needs --extra-soft-targets")
    if not 0.0 <= a.distill_weight <= 1.0:
        p.error("--distill-weight must be in [0, 1]")
    if a.aux_weight > 0 and not (a.mask_probs or a.train_true_mask):
        p.error("--aux-weight needs --mask-probs (the auxiliary target is the scored-probability)")
    if a.mask_probs and a.train_true_mask:
        p.error("--mask-probs and --train-true-mask are alternatives")
    return a


def compute_loss(args, predictions, targets, need_prediction, row_weight=None, weight_targets=None) -> torch.Tensor:
    """026's loss: 0.5*MSE + 0.5*(-per-sequence Pearson), every row multiplied
    by row_weight in BOTH terms on top of the |clipped target| weights.
    row_weight=None reproduces the unweighted loss of earlier experiments.
    weight_targets: take the |clipped target| weights from THIS tensor instead of
    `targets` (distillation: `targets` is the teacher mean, weights stay the true y's)."""
    wt = targets if weight_targets is None else weight_targets
    rw = need_prediction.float() if row_weight is None else need_prediction.float() * row_weight
    y = targets.clamp(-METRIC_CLIP, METRIC_CLIP); p = predictions.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = wt.clamp(-METRIC_CLIP, METRIC_CLIP).abs() * rw.unsqueeze(-1)
    mse = (w * (y - p) ** 2).sum() / w.sum().clamp_min(1e-8)
    if args.mse_weight >= 1.0:
        return mse
    per_seq, mass = [], []
    for b in range(predictions.shape[0]):
        m = need_prediction[b] & (rw[b] > 0)
        if m.sum() < 2:
            continue
        tgt = targets[b, m].T
        wb = wt[b, m].T.clamp(-METRIC_CLIP, METRIC_CLIP).abs() * rw[b, m].unsqueeze(0)
        if wb.sum() <= 1e-6:
            continue
        per_seq.append(weighted_pearson_loss(tgt, predictions[b, m].T, weight=wb)); mass.append(wb.sum().detach())
    if getattr(args, "pearson_seq_weight", "equal") == "mass":
        mw = torch.stack(mass); pearson = (torch.stack(per_seq) * mw).sum() / mw.sum().clamp_min(1e-8)
    else:
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


GROUP_ENCODERS = ("price_encoder", "volume_encoder", "i1_price_encoder", "i1_volume_encoder", "trade_encoder")


def attach_group_hooks(model):
    """036: capture every GroupEncoder's post-ReLU output (B, T, d) in model._group_out during the forward pass."""
    model._group_out = {}
    names = [n for n in GROUP_ENCODERS if hasattr(model, n)]
    for n in names:
        getattr(model, n).net[1].register_forward_hook(lambda m, i, o, n=n: model._group_out.__setitem__(n, o))
    return names


def ortho_penalty(args, groups, need):
    """decor: rows = need_prediction rows (subsampled); per group standardise each unit, C_ab = Z_a^T Z_b / N,
    penalty = mean over group pairs of mean(C_ab^2). cosine: the proposal's literal version."""
    pairs, out = 0, 0.0
    if args.ortho_mode == "decor":
        idx = need.reshape(-1).nonzero().squeeze(1)
        if len(idx) > args.ortho_rows:
            idx = idx[torch.randint(len(idx), (args.ortho_rows,), device=idx.device)]
        Z = []
        for g in groups:
            z = g.reshape(-1, g.shape[-1])[idx].float()
            Z.append((z - z.mean(0)) / (z.std(0) + 1e-4))
        for i in range(len(Z)):
            for j in range(i + 1, len(Z)):
                out = out + ((Z[i].T @ Z[j]) / len(idx)).pow(2).mean(); pairs += 1
    else:
        P = [F.normalize(g.mean(1), dim=1) for g in groups]
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                if P[i].shape[-1] == P[j].shape[-1]:
                    out = out + (P[i] * P[j]).sum(1).abs().mean(); pairs += 1
    return out / max(pairs, 1)


def train_one_epoch(args, model, loader, optimizer, device, epoch, ema):
    model.train()
    tot, tot_aux, tot_g, n = 0.0, 0.0, 0.0, 0
    tot_ortho, n_ortho = 0.0, 0
    progress = tqdm(loader, desc=f"epoch {epoch} [train]", unit="batch", leave=False)
    for i, batch in enumerate(progress):
        if args.max_train_batches and i >= args.max_train_batches:
            break
        features = batch["features"].to(device); targets = batch["targets"].to(device)
        need = batch["need_prediction"].to(device); p_scored = batch["row_weight"].to(device)
        row_weight = args.weight_floor + (1.0 - args.weight_floor) * p_scored.clamp(0, 1) ** args.weight_power
        if args.cosine_restarts > 0 and epoch >= args.warmup_epochs:  # 036: per-batch cosine annealing with warm restarts
            t = ((epoch - args.warmup_epochs) + i / len(loader)) % args.cosine_restarts / args.cosine_restarts
            for g in optimizer.param_groups:
                g["lr"] = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))
        optimizer.zero_grad()
        if args.arch == "joint_pair":  # 032: members on their own targets + their MEAN on the mean target
            ta, tb = batch["soft_targets"].to(device), batch["soft_targets_b"].to(device)
            pa, pb, xa, xb = model.members(features)
            lam = args.joint_lambda
            loss = lam * compute_loss(args, 0.5 * (pa + pb), 0.5 * (ta + tb), need, row_weight, weight_targets=targets) \
                + (1 - lam) * 0.5 * (compute_loss(args, pa, ta, need, row_weight, weight_targets=targets) + compute_loss(args, pb, tb, need, row_weight, weight_targets=targets))
            if args.aux_weight > 0:
                aux_loss = 0.5 * (F.binary_cross_entropy_with_logits(xa[need], p_scored[need].clamp(0, 1)) + F.binary_cross_entropy_with_logits(xb[need], p_scored[need].clamp(0, 1)))
                loss = loss + args.aux_weight * aux_loss; tot_aux += aux_loss.item()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad if args.clip_grad > 0 else float("inf"))
            optimizer.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item(); tot_g += float(gnorm); n += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            continue
        predictions, _, aux = model(features, return_aux=True)
        if args.distill_weight > 0 and args.blend_targets:
            mixed = (1.0 - args.distill_weight) * targets + args.distill_weight * batch["soft_targets"].to(device).float()
            loss = compute_loss(args, predictions, mixed, need, row_weight, weight_targets=targets)
        else:
            loss = compute_loss(args, predictions, targets, need, row_weight)
        if args.distill_weight > 0 and not args.blend_targets:
            soft = batch["soft_targets"].to(device)
            loss = (1.0 - args.distill_weight) * loss + args.distill_weight * compute_loss(args, predictions, soft, need, row_weight, weight_targets=targets)
        if args.dist_ce > 0:  # 032 lean_dist: the distribution learns the TRUE label distribution
            from model import hl_gauss
            logits = model.last_logits[need]                      # (rows, 2, n_bins)
            tgt = hl_gauss(targets[need], model.centers, args.dist_sigma)
            ce = -(tgt * torch.log_softmax(logits, dim=-1)).sum(-1).mean(-1)   # per row, mean of the two targets
            rw = row_weight[need]
            loss = loss + args.dist_ce * (ce * rw).sum() / rw.sum().clamp_min(1e-8)
        if args.aux_weight > 0:
            aux_loss = F.binary_cross_entropy_with_logits(aux[need], p_scored[need].clamp(0, 1))
            loss = loss + args.aux_weight * aux_loss
            tot_aux += aux_loss.item()
        if args.ortho_weight > 0:  # 036
            pen = ortho_penalty(args, [model._group_out[n] for n in model._group_names], need)
            loss = loss + args.ortho_weight * pen
            tot_ortho = tot_ortho + pen.item(); n_ortho = n_ortho + 1
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad if args.clip_grad > 0 else float("inf"))
        optimizer.step()
        if ema is not None:
            ema.update(model)
        tot += loss.item(); tot_g += float(gnorm); n += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    if n_ortho:
        tqdm.write(f"epoch {epoch}: ortho penalty ({args.ortho_mode}) mean {tot_ortho / n_ortho:.5f} lr_end {optimizer.param_groups[0]['lr']:.2e}")
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

    soft = args.soft_targets if args.distill_weight > 0 else None
    train_ds = ConnectomeSequenceDataset(args.train, mask_probs_path=args.mask_probs, true_mask_as_weight=args.train_true_mask,
                                         soft_targets_path=soft, soft_targets_b_path=args.soft_targets_b)
    if args.exclude_fold is not None:
        keep = [i for i in range(len(train_ds)) if i % args.n_folds != args.exclude_fold]
        print(f"out-of-fold: excluding fold {args.exclude_fold} of {args.n_folds} -> {len(keep)} of {len(train_ds)} train sequences", flush=True)
        train_ds = Subset(train_ds, keep)
    if args.extra_train:
        extra_ds = ConnectomeSequenceDataset(args.extra_train, true_mask_as_weight=True,
                                             soft_targets_path=args.extra_soft_targets if args.distill_weight > 0 else None,
                                             soft_targets_b_path=args.extra_soft_targets_b)
        if args.extra_exclude_fold is not None:
            extra_ds = Subset(extra_ds, [i for i in range(len(extra_ds)) if i % args.n_folds != args.extra_exclude_fold])
            print(f"extra-train: excluding fold {args.extra_exclude_fold} of {args.n_folds} -> {len(extra_ds)} sequences", flush=True)
        train_ds = ConcatDataset([train_ds] + [extra_ds] * args.extra_repeat)
        if args.extra_repeat > 1:
            print(f"extra-train repeated {args.extra_repeat}x per epoch", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = None
    if not args.no_validation:
        valid_ds = ConnectomeSequenceDataset(args.valid)
        if args.valid_fold is not None:
            valid_ds = Subset(valid_ds, [i for i in range(len(valid_ds)) if i % args.n_folds == args.valid_fold])
            print(f"scoring only fold {args.valid_fold} of {args.n_folds}: {len(valid_ds)} valid sequences", flush=True)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    aux_on = args.aux_weight > 0
    init_dir = init_weights = None
    if args.init_from:
        init_dir, init_weights = args.init_from.rsplit(":", 1)
        with open(Path(init_dir) / "config.json") as f:
            icfg = json.load(f)
        args.arch = icfg.get("arch", "lean")
        for k in ("hidden_size", "num_layers", "price_dim", "volume_dim", "trade_dim", "head_hidden"):
            setattr(args, k, icfg[k])
        aux_on = icfg.get("aux_scored", False)  # keep the checkpoint's heads (an unused aux head is harmless)
    model = build_model(
        args.arch, hidden_size=args.hidden_size, num_layers=args.num_layers,
        price_dim=args.price_dim, volume_dim=args.volume_dim, trade_dim=args.trade_dim,
        head_hidden=args.head_hidden, dropout=args.dropout, aux_scored=aux_on, bidirectional=args.bidirectional,
    ).to(device)
    if init_dir:
        model.load_state_dict(torch.load(Path(init_dir) / init_weights, map_location=device))
        print(f"initialised from {args.init_from}", flush=True)
    if args.ortho_weight > 0:  # 036
        model._group_names = attach_group_hooks(model)
        print(f"ortho penalty: {args.ortho_mode}, weight {args.ortho_weight}, groups {model._group_names}", flush=True)
        if len(model._group_names) < 2:
            raise SystemExit("--ortho-weight needs a model with >= 2 group encoders")
    if args.cosine_restarts > 0:
        print(f"LR: cosine annealing with warm restarts every {args.cosine_restarts} epochs after {args.warmup_epochs} warmup epochs, min {args.min_lr}", flush=True)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.weight_decay > 0
                 else torch.optim.Adam(model.parameters(), lr=args.lr))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    ema_model = copy.deepcopy(model) if ema is not None else None
    start_epoch = 0
    if args.resume_state:
        st = torch.load(args.resume_state, map_location=device)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        if ema is not None:
            ema.shadow = {k: v.to(device) for k, v in st["ema_shadow"].items()}; ema.n = st["ema_n"]
        start_epoch = st["epoch"] + 1
        torch.manual_seed(args.seed)  # the branch's own data order / dropout from here on
        print(f"resumed {args.resume_state} at epoch {start_epoch} (lr {optimizer.param_groups[0]['lr']:.2e}), branch seed {args.seed}", flush=True)

    ckpt = Path(args.checkpoint_dir); ckpt.mkdir(parents=True, exist_ok=True)
    with open(ckpt / "config.json", "w") as f:
        json.dump({"arch": args.arch, "hidden_size": args.hidden_size, "num_layers": args.num_layers, "price_dim": args.price_dim,
                   "volume_dim": args.volume_dim, "trade_dim": args.trade_dim, "head_hidden": args.head_hidden,
                   "aux_scored": aux_on, "bidirectional": args.bidirectional,
                   "recipe": {k: v for k, v in vars(args).items() if k not in ("train", "valid")}}, f, indent=2)
    print(f"train sequences: {len(train_ds)} | weights: {'scored-prob' if args.mask_probs else 'none'} (power {args.weight_power}, floor {args.weight_floor}) | ema: {args.ema_decay} | aux: {args.aux_weight} | distill: {args.distill_weight} | hidden: {args.hidden_size} | bidirectional: {args.bidirectional} | validation: {not args.no_validation}", flush=True)

    if init_dir and valid_loader is not None:  # the starting point's score on the same sequences
        _, wp0 = evaluate(args, model, valid_loader, device, -1, tag="initial")
        tqdm.write(f"initial (before fine-tuning): valid_wp={wp0['wp']:.6f} t0={wp0['t0']:.6f} t1={wp0['t1']:.6f}")

    warm = max(0, args.warmup_epochs); warm_lr0 = args.lr * args.warmup_start_factor
    distill0 = args.distill_weight
    best, best_ema, stale, stale_ema = float("-inf"), float("-inf"), 0, 0
    for epoch in tqdm(range(start_epoch, args.epochs), desc="epochs", unit="epoch"):
        if epoch < warm:
            for g in optimizer.param_groups:
                g["lr"] = warm_lr0 + (epoch + 1) / warm * (args.lr - warm_lr0)
        elif args.cosine_restarts <= 0 and (args.no_validation or args.lr_milestones) and epoch in args.lr_milestones:  # milestones = fixed schedule, also with validation
            for g in optimizer.param_groups:
                g["lr"] = max(g["lr"] * args.lr_factor, args.min_lr)

        if args.distill_anneal_start is not None and epoch >= args.distill_anneal_start:
            span = max(1, args.epochs - 1 - args.distill_anneal_start)
            frac = min(1.0, (epoch - args.distill_anneal_start + 1) / (span + 1))
            args.distill_weight = distill0 + (args.distill_final - distill0) * frac
        train_loss, gnorm, aux_loss = train_one_epoch(args, model, train_loader, optimizer, device, epoch, ema)
        lr = optimizer.param_groups[0]["lr"]
        tag = " [warmup]" if epoch < warm else ""
        torch.save(model.state_dict(), ckpt / "last.pt")
        if args.save_state:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "ema_shadow": ema.shadow if ema is not None else None, "ema_n": ema.n if ema is not None else 0},
                       ckpt / "state.pt")
        if args.save_epochs:
            torch.save(model.state_dict(), ckpt / f"epoch_{epoch:02d}.pt")
        if ema is not None:
            ema.copy_to(ema_model)
            torch.save(ema_model.state_dict(), ckpt / "ema.pt")

        if args.no_validation:
            tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f} lr={lr:.2e} gnorm={gnorm:.3f} aux_loss={aux_loss:.4f} distill_w={args.distill_weight:.3f}{tag} [no validation]")
            continue

        valid_loss, wp = evaluate(args, model, valid_loader, device, epoch)
        extra = ""
        if "aux_auc" in wp:
            extra += f" aux_auc={wp['aux_auc']:.4f} aux_ap={wp['aux_ap']:.4f}"
        ema_wp = None
        if ema is not None:
            _, r = evaluate(args, ema_model, valid_loader, device, epoch, tag="valid-ema", aux_batches=0)
            ema_wp = r["wp"]; extra += f" ema_wp={ema_wp:.6f}"
        if epoch >= warm and not args.lr_milestones:  # plateau decay only without a fixed schedule
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
