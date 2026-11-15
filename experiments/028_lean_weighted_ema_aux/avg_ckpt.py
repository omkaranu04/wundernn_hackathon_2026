"""Quick check of weight averaging on finished runs: score each given
checkpoint and the uniform average of their weights with the corrected global
WP (true scored mask) on valid. No training. LSTMs have no batch-norm
statistics, so averaging state_dicts needs no re-calibration."""
from __future__ import annotations
import argparse, importlib.util, inspect, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import ConnectomeSequenceDataset, collate  # noqa: E402
from eval_wp import evaluate_wp  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--run-dir", required=True, help="dir with model.py, config.json and the checkpoints")
a.add_argument("--class-name", required=True)
a.add_argument("--ckpts", nargs="+", required=True)
a.add_argument("--valid", required=True); a.add_argument("--device", default="cuda")
args = a.parse_args()
rd = Path(args.run_dir); device = torch.device(args.device)
spec = importlib.util.spec_from_file_location("avg_model", rd / "model.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cls = getattr(mod, args.class_name); cfg = json.load(open(rd / "config.json"))
sig = inspect.signature(cls.__init__).parameters
kw = {k: v for k, v in cfg.items() if k in sig}
if "dropout" in sig: kw["dropout"] = 0.0
loader = DataLoader(ConnectomeSequenceDataset(args.valid), batch_size=8, shuffle=False, collate_fn=collate)
sds = [torch.load(rd / c, map_location=device) for c in args.ckpts]
def score(sd):
    m = cls(**kw).to(device); m.load_state_dict(sd); return evaluate_wp(m, loader, device)["wp"]
res = {c: score(sd) for c, sd in zip(args.ckpts, sds)}
avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].dtype.is_floating_point else sds[0][k] for k in sds[0]}
res["AVERAGE"] = score(avg)
print(rd.name, " | ".join(f"{k}={v:.6f}" for k, v in res.items()), f"| avg - best single = {res['AVERAGE']-max(v for k,v in res.items() if k!='AVERAGE'):+.6f}")
