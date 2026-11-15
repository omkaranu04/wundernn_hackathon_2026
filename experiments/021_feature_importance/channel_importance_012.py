"""Experiment 012-specific: which of its 5 expanded channels
[x, asinh(x), sin(x), cos(x), tanh(x)] does the trained model rely on?

012's `encode()` calls the module-level `expand()` four times per forward
(price, volume, trade, additional - in that order). We replace `expand` with
a wrapper that, for chosen (group, channel) pairs, swaps that channel's
values with another sequence's (batch roll) AFTER the expansion, leaving the
raw input and the other channels intact. Drop in corrected global WP = reliance.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importance import load_model, GlobalWPAccumulator, ConnectomeSequenceDataset  # noqa: E402

CHANNELS = ["x", "asinh", "sin", "cos", "tanh"]
GROUPS = ["price", "volume", "trade", "additional"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-seqs", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rolls", type=int, nargs="+", default=[1, 7])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device(a.device)

    model = load_model(Path(a.model_dir), "NonlinearGroupedEncoderLSTMModel", device)
    # load_model() doesn't register its module in sys.modules; reach its namespace via encode()'s globals
    ns = type(model).encode.__globals__
    orig_expand = ns["expand"]

    ds = ConnectomeSequenceDataset(a.valid)
    idx = np.unique(np.linspace(0, len(ds) - 1, a.n_seqs).astype(int))
    n = (len(idx) // a.batch_size) * a.batch_size
    items = [ds[int(i)] for i in idx[:n]]
    feats = torch.stack([it["features"] for it in items])
    targs = torch.stack([it["targets"] for it in items])
    masks = torch.stack([it["need_prediction"] & it["is_scored"] for it in items])

    state = {"calls": 0, "targets": set(), "roll": 1}

    def patched(raw):
        out = orig_expand(raw)
        g = state["calls"] % 4
        state["calls"] += 1
        d = raw.shape[-1]
        for (gg, c) in state["targets"]:
            if gg == g:
                sl = slice(c * d, (c + 1) * d)
                out = out.clone()
                out[..., sl] = out.roll(state["roll"], dims=0)[..., sl]
        return out

    ns["expand"] = patched

    @torch.no_grad()
    def run(targets, roll):
        state.update(calls=0, targets=set(targets), roll=roll)
        acc = GlobalWPAccumulator(device)
        for i in range(0, feats.shape[0], a.batch_size):
            x = feats[i:i + a.batch_size].to(device)
            preds, _ = model(x)
            acc.add(targs[i:i + a.batch_size].to(device), preds, masks[i:i + a.batch_size].to(device))
        return acc.result()

    base = run([], 0)
    print(f"baseline WP on subset: {base['wp']:.6f}", flush=True)

    def drop(targets):
        runs = [run(targets, r) for r in a.rolls]
        return {"wp": float(np.mean([base["wp"] - r["wp"] for r in runs])),
                "runs": [base["wp"] - r["wp"] for r in runs]}

    res = {"baseline": base, "per_channel_all_groups": {}, "per_group_channel": {}}
    for c, cn in enumerate(CHANNELS):
        res["per_channel_all_groups"][cn] = drop([(g, c) for g in range(4)])
    for g, gn in enumerate(GROUPS):
        for c, cn in enumerate(CHANNELS):
            res["per_group_channel"][f"{gn}/{cn}"] = drop([(g, c)])
    Path(a.out).write_text(json.dumps(res, indent=2))

    print("\nPer channel type, permuted in ALL four groups at once:")
    for k, v in res["per_channel_all_groups"].items():
        print(f"  {k:<6} dWP={v['wp']:+.5f}  runs={[round(x,5) for x in v['runs']]}")
    print("\nPer group x channel (dWP, rows=group, cols=" + "/".join(CHANNELS) + "):")
    for gn in GROUPS:
        print(f"  {gn:<11} " + "  ".join(f"{res['per_group_channel'][f'{gn}/{cn}']['wp']:+.5f}" for cn in CHANNELS))


if __name__ == "__main__":
    main()
