"""Model-based permutation feature importance on ALREADY-TRAINED models (no
training). For each of the 112 raw input features (and for feature groups),
replace that feature's whole 20,000-step trace in each sequence with the
trace from a DIFFERENT sequence in the same batch (torch.roll over the batch
dim). This keeps the feature's marginal distribution and its temporal
structure but breaks its alignment with the targets. The drop in corrected
global WP (019's validated `GlobalWPAccumulator`) = how much the model relies
on it. Done with two different roll offsets per condition to expose
permutation noise.

Caveats (also in the README): permutation importance under-credits features
that are redundant with correlated ones (the model re-routes), and says
nothing about a feature's value to a model RETRAINED without it. It ranks
reliance, it doesn't prove dispensability.

Usage:
    python importance.py --model-dir ../004_grouped_encoder_lstm --class-name GroupedEncoderLSTMModel \
        --valid ../../wnn_connectome_starterpack/datasets/valid.parquet --device cuda --out results_004.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "019_loss_optimizer_recipe"))
sys.path.insert(0, str(HERE.parent.parent / "shared"))
from dataset import ConnectomeSequenceDataset  # noqa: E402
from eval_wp import GlobalWPAccumulator  # noqa: E402
from data import FEATURE_COLUMNS  # noqa: E402

# column ranges, identical to every grouped-encoder model's slice constants
ATOMIC_GROUPS = {
    "i0_price": (0, 22), "i0_volume": (22, 44), "i0_dp": (44, 48), "i0_dv": (48, 52),
    "i1_price": (52, 74), "i1_volume": (74, 96), "i1_dp": (96, 100), "i1_dv": (100, 104),
    "additional": (104, 112),
}
COMBINED_GROUPS = {
    "ALL price (i0+i1)": ["i0_price", "i1_price"],
    "ALL volume (i0+i1)": ["i0_volume", "i1_volume"],
    "ALL trade dp+dv (i0+i1)": ["i0_dp", "i0_dv", "i1_dp", "i1_dv"],
    "ALL instrument 0": ["i0_price", "i0_volume", "i0_dp", "i0_dv"],
    "ALL instrument 1": ["i1_price", "i1_volume", "i1_dp", "i1_dv"],
}


def load_model(model_dir: Path, class_name: str, device):
    spec = importlib.util.spec_from_file_location("imp_model_mod", model_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.load(open(model_dir / "checkpoints" / "config.json"))
    keys = ("hidden_size", "num_layers", "price_dim", "volume_dim", "trade_dim", "head_hidden")
    model = getattr(mod, class_name)(**{k: cfg[k] for k in keys}, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_dir / "checkpoints" / "best.pt", map_location=device))
    return model.eval()


@torch.no_grad()
def score(model, feats, targs, masks, cols, roll, device, bs):
    acc = GlobalWPAccumulator(device)
    for i in range(0, feats.shape[0], bs):
        x = feats[i:i + bs].to(device, non_blocking=True)
        if cols is not None:
            x = x.clone()
            x[:, :, cols] = x.roll(roll, dims=0)[:, :, cols]
        preds, _ = model(x)
        acc.add(targs[i:i + bs].to(device), preds, masks[i:i + bs].to(device))
    return acc.result()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--class-name", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-seqs", type=int, default=240, help="evenly spaced valid sequences used")
    p.add_argument("--batch-size", type=int, default=16, help="also the pool permuted-in sequences come from")
    p.add_argument("--rolls", type=int, nargs="+", default=[1, 7])
    p.add_argument("--only-groups", action="store_true", help="skip per-feature pass (smoke tests)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.model_dir), args.class_name, device)
    ds = ConnectomeSequenceDataset(args.valid)
    idx = np.unique(np.linspace(0, len(ds) - 1, args.n_seqs).astype(int))
    n = (len(idx) // args.batch_size) * args.batch_size  # whole batches only (roll needs full pools)
    idx = idx[:n]
    print(f"loading {n} of {len(ds)} valid sequences (every ~{len(ds)/n:.1f}th) into memory ...", flush=True)
    items = [ds[int(i)] for i in tqdm(idx, desc="load", unit="seq")]
    feats = torch.stack([it["features"] for it in items])
    targs = torch.stack([it["targets"] for it in items])
    masks = torch.stack([it["need_prediction"] & it["is_scored"] for it in items])

    t = time.time()
    base = score(model, feats, targs, masks, None, 0, device, args.batch_size)
    print(f"baseline WP on subset: {base['wp']:.6f} (t0={base['t0']:.6f}, t1={base['t1']:.6f})  [{time.time()-t:.0f}s/pass]", flush=True)

    def drop(cols):
        runs = [score(model, feats, targs, masks, cols, r, device, args.batch_size) for r in args.rolls]
        d = {k: float(np.mean([base[k] - r[k] for r in runs])) for k in ("wp", "t0", "t1")}
        d["wp_runs"] = [base["wp"] - r["wp"] for r in runs]
        return d

    res = {"baseline": base, "n_seqs": int(n), "rolls": args.rolls, "groups": {}, "combined": {}, "features": {}}
    for name, (a, b) in tqdm(ATOMIC_GROUPS.items(), desc="groups", unit="grp"):
        res["groups"][name] = {**drop(list(range(a, b))), "n_features": b - a}
    for name, parts in tqdm(COMBINED_GROUPS.items(), desc="combined", unit="grp"):
        cols = [c for part in parts for c in range(*ATOMIC_GROUPS[part])]
        res["combined"][name] = {**drop(cols), "n_features": len(cols)}
    if not args.only_groups:
        for j, name in enumerate(tqdm(FEATURE_COLUMNS, desc="features", unit="feat")):
            res["features"][name] = drop([j])
    Path(args.out).write_text(json.dumps(res, indent=2))

    print("\nGROUP importance (drop in WP when permuted; larger = more relied upon):")
    for k, v in sorted({**res["groups"], **res["combined"]}.items(), key=lambda kv: -kv[1]["wp"]):
        print(f"  {k:<26} n={v['n_features']:>3}  dWP={v['wp']:+.5f}  (t0 {v['t0']:+.5f}, t1 {v['t1']:+.5f})  runs={[round(x,5) for x in v['wp_runs']]}")
    if res["features"]:
        ranked = sorted(res["features"].items(), key=lambda kv: -kv[1]["wp"])
        print("\nTOP 15 features:")
        for k, v in ranked[:15]:
            print(f"  {k:<8} dWP={v['wp']:+.5f}  runs={[round(x,5) for x in v['wp_runs']]}")
        print("\nBOTTOM 15 features (least relied upon):")
        for k, v in ranked[-15:]:
            print(f"  {k:<8} dWP={v['wp']:+.5f}  runs={[round(x,5) for x in v['wp_runs']]}")


if __name__ == "__main__":
    main()
