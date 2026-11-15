"""Offline uniform averaging of saved per-epoch checkpoints (SWA-style) for an
028 run: score several epoch windows with the corrected global WP on valid
(true scored mask), next to the run's best raw epoch and best EMA. No
training. Windows are inclusive epoch ranges."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import ConnectomeSequenceDataset, collate  # noqa: E402
from eval_wp import evaluate_wp  # noqa: E402
from model import LeanCosGroupedEncoderLSTMModel  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--run-dir", required=True); a.add_argument("--valid", required=True)
a.add_argument("--windows", nargs="+", default=["4-8", "6-10", "6-12", "8-12", "7-13", "8-14", "9-13", "10-12", "5-15", "3-16"])
a.add_argument("--also", nargs="*", default=["best.pt", "best_ema.pt"])
a.add_argument("--device", default="cuda"); a.add_argument("--out", default=None)
args = a.parse_args()
rd = Path(args.run_dir); device = torch.device(args.device)
cfg = json.load(open(rd / "config.json"))
kw = {k: cfg[k] for k in ("hidden_size", "num_layers", "price_dim", "volume_dim", "trade_dim", "head_hidden")}
loader = DataLoader(ConnectomeSequenceDataset(args.valid), batch_size=8, shuffle=False, collate_fn=collate)
def score(sd):
    m = LeanCosGroupedEncoderLSTMModel(**kw, dropout=0.0, aux_scored=cfg.get("aux_scored", False)).to(device)
    m.load_state_dict(sd); r = evaluate_wp(m, loader, device); return r["wp"]
res = {}
for f in args.also:
    if (rd / f).exists(): res[f] = score(torch.load(rd / f, map_location=device)); print(f"{rd.name}  {f:<14} {res[f]:.6f}", flush=True)
for w in args.windows:
    lo, hi = map(int, w.split("-")); files = [rd / f"epoch_{e:02d}.pt" for e in range(lo, hi + 1)]
    files = [f for f in files if f.exists()]
    if len(files) < 2: continue
    sds = [torch.load(f, map_location=device) for f in files]
    avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].dtype.is_floating_point else sds[0][k] for k in sds[0]}
    res[f"avg {w}"] = score(avg); print(f"{rd.name}  avg epochs {w:<6} ({len(files)} ckpts) {res[f'avg {w}']:.6f}", flush=True)
    if args.out and res[f"avg {w}"] >= max(v for k, v in res.items() if k.startswith("avg")):
        torch.save(avg, rd / "best_window_avg.pt"); (rd / "best_window_avg.txt").write_text(f"{w} {res[f'avg {w}']:.6f}\n")
print("DONE", rd.name, "best:", max(res, key=res.get), f"{max(res.values()):.6f}")
