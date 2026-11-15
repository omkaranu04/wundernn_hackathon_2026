"""Run every trained model in `ens/<name>/{model.py,best.pt,config.json}` over
valid.parquet ONCE (data read once, all models per batch) and store each
model's predictions ONLY at the rows the corrected metric scores
(need_prediction & is_scored: ~3.5M of 37.5M rows), plus the matching
targets and sequence ids. Ensembles are then evaluated offline, instantly,
by ensemble_eval.py. Predictions are the raw model outputs (unclipped), in
(t0, t1) units, exactly what the exported ONNX would return.
"""
from __future__ import annotations

import argparse, importlib.util, inspect, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "019_loss_optimizer_recipe"))
from dataset import ConnectomeSequenceDataset, collate  # noqa: E402

# name -> class name in that model.py
REGISTRY = {
    "001_d32": "LSTMAttentionModel", "002": "StackedLSTMModel", "003": "StackedLSTMMLPModel",
    "004": "GroupedEncoderLSTMModel", "005": "StackedLSTMMLPModel", "006": "WideLSTMSinhModel",
    "008": "StackedLSTMModel", "009": "SplitGroupedEncoderLSTMModel",
    "010": "SumReparamGroupedEncoderLSTMModel", "012": "NonlinearGroupedEncoderLSTMModel",
    "013_priceonly": "RollingNormGroupedEncoderLSTMModel", "014": "SparseAttentionLSTMModel",
    "015": "NonlinearWindowedAttentionModel", "016": "NonlinearSparseAttentionModel",
    "017": "NonlinearSumReparamGroupedEncoderLSTMModel",
    "018": "NonlinearRollingNormGroupedEncoderLSTMModel",
    "019_control": "GroupedEncoderLSTMModel", "019_pooled50": "GroupedEncoderLSTMModel",
    "019_pooled60": "GroupedEncoderLSTMModel", "022_arm2": "GroupedEncoderLSTMModel",
}


def load(name, cls_name, ens_dir, device):
    d = ens_dir / name
    spec = importlib.util.spec_from_file_location(f"ens_{name}", d / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, cls_name)
    cfg = json.load(open(d / "config.json"))
    sig = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in cfg.items() if k in sig}
    if "dropout" in sig:
        kwargs["dropout"] = 0.0
    m = cls(**kwargs).to(device)
    m.load_state_dict(torch.load(d / "best.pt", map_location=device))
    return m.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ens-dir", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    device = torch.device(a.device)
    ens_dir, out = Path(a.ens_dir), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    names = [n for n in REGISTRY if (a.only is None or n in a.only) and (ens_dir / n / "best.pt").exists()]
    models = {}
    for n in names:
        try:
            models[n] = load(n, REGISTRY[n], ens_dir, device)
        except Exception as e:  # report and continue
            print(f"SKIP {n}: {type(e).__name__}: {e}", flush=True)
    print(f"loaded {len(models)} models: {list(models)}", flush=True)

    loader = DataLoader(ConnectomeSequenceDataset(a.valid), batch_size=a.batch_size, shuffle=False, collate_fn=collate)
    preds = {n: [] for n in models}; targets, seqids = [], []
    secs = {n: 0.0 for n in models}
    seq_base = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="valid", unit="batch")):
            if a.max_batches and i >= a.max_batches:
                break
            x = batch["features"].to(device)
            mask = (batch["need_prediction"] & batch["is_scored"])
            B = x.shape[0]
            targets.append(batch["targets"][mask].numpy())
            seqids.append((torch.arange(B).unsqueeze(1).expand_as(mask)[mask] + seq_base).numpy())
            seq_base += B
            mask_d = mask.to(device)
            for n, m in models.items():
                t = time.perf_counter()
                y = m(x)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                secs[n] += time.perf_counter() - t
                preds[n].append(y[mask_d].float().cpu().numpy())
    np.savez(out / "meta.npz", targets=np.concatenate(targets), seq=np.concatenate(seqids))
    for n in models:
        np.save(out / f"pred_{n}.npy", np.concatenate(preds[n]))
    print("forward seconds per model:", {n: round(s) for n, s in secs.items()})
    print(f"saved {len(models)} prediction files + meta to {out}")


if __name__ == "__main__":
    main()
