"""Train the mask predictor on a labelled split of valid.parquet, report
held-out AUC / average precision / calibration. Targets t0,t1 are never read
into the model. Split by sequence: held-out = every 4th valid sequence.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "shared"))
from data import file_columns, read_sequence  # noqa: E402
from mask_model import MaskPredictor, predict_sequence, SEQ_LEN  # noqa: E402


def calibration(p, y, bins=(0, .02, .05, .1, .2, .3, .5, .7, 1.01)):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append((f"[{lo:.2f},{min(hi,1):.2f})", int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def evaluate(model, pf, ids, hs, device, cap_rows=None):
    model.eval()
    ps, ys = [], []
    for g in ids:
        s = read_sequence(pf, g, has_is_scored=hs, has_targets=True)
        X = torch.from_numpy(s.features).to(device)
        p = torch.sigmoid(predict_sequence(model, X)).cpu().numpy()
        m = s.need_prediction
        ps.append(p[m]); ys.append(s.is_scored[m])
    p, y = np.concatenate(ps), np.concatenate(ys).astype(np.int8)
    return p, y, {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "base_rate": float(y.mean()), "mean_pred": float(p.mean())}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--valid", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--epochs", type=int, default=15)
    a.add_argument("--crops-per-seq", type=int, default=4)
    a.add_argument("--crop", type=int, default=2000)
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--lr", type=float, default=2e-3)
    a.add_argument("--hidden", type=int, default=96)
    a.add_argument("--layers", type=int, default=2)
    a.add_argument("--cosine", action="store_true", help="cosine-decay the LR to 5%% of --lr over the run")
    a.add_argument("--max-train-seqs", type=int, default=0)
    a.add_argument("--eval-seqs", type=int, default=60, help="held-out seqs scored each epoch (final eval uses all)")
    a.add_argument("--causal", action="store_true", help="unidirectional LSTM: only past and current rows (what a shipped model could know)")
    a.add_argument("--device", default="cuda")
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = torch.device(args.device); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.valid); hs, _ = file_columns(pf)
    assert hs, "valid file must carry is_scored"
    n = pf.num_row_groups
    held = [i for i in range(n) if i % 4 == 0]; tr = [i for i in range(n) if i % 4 != 0]
    if args.max_train_seqs: tr = tr[:args.max_train_seqs]; held = held[:max(8, args.max_train_seqs // 3)]
    ev_ids = held[::max(1, len(held) // args.eval_seqs)][:args.eval_seqs]
    print(f"train seqs {len(tr)}, held-out seqs {len(held)} (per-epoch eval on {len(ev_ids)})", flush=True)

    model = MaskPredictor(hidden=args.hidden, layers=args.layers, bidirectional=not args.causal).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ap, history = -1.0, []
    import math
    for epoch in range(args.epochs):
        if args.cosine:
            for g in opt.param_groups:
                g["lr"] = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * epoch / max(1, args.epochs - 1))))
        model.train(); order = rng.permutation(tr); xs, ys, ms, ps = [], [], [], []
        losses = []; t0 = time.time()
        def flush():
            x = torch.stack(xs).to(device); y = torch.stack(ys).to(device); m = torch.stack(ms).to(device)
            pos = torch.stack(ps).to(device).unsqueeze(-1)
            logit = model(x, pos)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[m], y[m].float())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item()); xs.clear(); ys.clear(); ms.clear(); ps.clear()
        for g in tqdm(order, desc=f"epoch {epoch}", leave=False):
            s = read_sequence(pf, int(g), has_is_scored=hs, has_targets=True)
            for _ in range(args.crops_per_seq):
                st = int(rng.integers(0, SEQ_LEN - args.crop + 1)); sl = slice(st, st + args.crop)
                if not s.need_prediction[sl].any():
                    continue
                xs.append(torch.from_numpy(s.features[sl])); ys.append(torch.from_numpy(s.is_scored[sl]))
                ms.append(torch.from_numpy(s.need_prediction[sl]))
                ps.append((torch.arange(st, st + args.crop, dtype=torch.float32) / SEQ_LEN))
                if len(xs) == args.batch: flush()
        if xs: flush()
        _, _, r = evaluate(model, pf, ev_ids, hs, device)
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), **r})
        print(f"epoch {epoch}: bce={np.mean(losses):.4f}  held-out AUC={r['auc']:.4f} AP={r['ap']:.4f} (base {r['base_rate']:.3f}, mean pred {r['mean_pred']:.3f})  [{time.time()-t0:.0f}s]", flush=True)
        if r["ap"] > best_ap:
            best_ap = r["ap"]; torch.save(model.state_dict(), out / "mask_model.pt")
    model.load_state_dict(torch.load(out / "mask_model.pt", map_location=device))
    p, y, r = evaluate(model, pf, held, hs, device)
    cal = calibration(p, y)
    res = {"final_heldout": r, "n_heldout_seqs": len(held), "n_train_seqs": len(tr), "calibration": cal, "history": history, "hidden": args.hidden, "layers": args.layers, "epochs": args.epochs, "cosine": args.cosine, "causal": args.causal}
    (out / "mask_train_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nFINAL held-out ({len(held)} seqs, {len(y):,} rows): AUC={r['auc']:.4f}  AP={r['ap']:.4f}  base rate={r['base_rate']:.4f}  mean predicted={r['mean_pred']:.4f}")
    print("calibration (predicted bin | rows | mean predicted | actual scored rate):")
    for b, nrow, mp, ar in cal: print(f"  {b:<12} {nrow:>10,}  {mp:.4f}  {ar:.4f}")


if __name__ == "__main__":
    main()
