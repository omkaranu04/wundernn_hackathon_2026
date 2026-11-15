"""Weight soup: average the weights of several branches of ONE run (train.py --resume-state from a shared
state.pt), and write a checkpoint dir (config.json of the first branch + ema.pt) that export_slim.py can read.
    python soup.py --out results/checkpoints_SOUP --weights ema.pt results/checkpoints_B0 results/checkpoints_B1 ..."""
import argparse
import json
import shutil
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--weights", default="ema.pt")
ap.add_argument("dirs", nargs="+")
a = ap.parse_args()
sds = [torch.load(Path(d) / a.weights, map_location="cpu") for d in a.dirs]
soup = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k]
        for k in sds[0]}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.save(soup, out / "ema.pt")
cfg = json.load(open(Path(a.dirs[0]) / "config.json")); cfg["soup_of"] = [str(d) for d in a.dirs]
json.dump(cfg, open(out / "config.json", "w"), indent=2)
print(f"soup of {len(sds)} branches -> {out / 'ema.pt'}")
