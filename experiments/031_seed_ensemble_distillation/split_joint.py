"""Split a --arch joint_pair checkpoint into its two members (2026-11-03, experiment 032): writes OUT_a (lean LSTM)
and OUT_b (lean_gru) checkpoint dirs with config.json + ema.pt, ready for export_slim.py --multi OUT_a:ema.pt OUT_b:ema.pt.
    python split_joint.py --checkpoint-dir results/checkpoints_R17_JOINT --weights ema.pt"""
import argparse
import json
from pathlib import Path

import torch

ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint-dir", required=True); ap.add_argument("--weights", default="ema.pt")
a = ap.parse_args()
d = Path(a.checkpoint_dir); cfg = json.load(open(d / "config.json")); sd = torch.load(d / a.weights, map_location="cpu")
for m, arch in (("a", "lean"), ("b", "lean_gru")):
    out = Path(f"{d}_{m}"); out.mkdir(parents=True, exist_ok=True)
    torch.save({k[2:]: v for k, v in sd.items() if k.startswith(f"{m}.")}, out / "ema.pt")
    c = {k: cfg[k] for k in ("hidden_size", "num_layers", "price_dim", "volume_dim", "trade_dim", "head_hidden", "aux_scored", "bidirectional")}
    c["arch"] = arch; c["split_from"] = str(d)
    json.dump(c, open(out / "config.json", "w"), indent=2)
    print(f"wrote {out}")
