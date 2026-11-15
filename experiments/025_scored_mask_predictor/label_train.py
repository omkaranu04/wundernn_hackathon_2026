"""Label every row of train.parquet with the mask predictor's scored-probability.
Output: <out>/mask_probs_train.npy, float16, shape (n_sequences, 20000),
row-group order (= sequence order in train.parquet). Rows with
need_prediction == False are stored too (ignored downstream)."""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from tqdm.auto import tqdm
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "shared"))
from data import file_columns, read_sequence  # noqa: E402
from mask_model import MaskPredictor, SEQ_LEN, predict_sequence  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--train", required=True); a.add_argument("--model-dir", required=True)
a.add_argument("--hidden", type=int, default=96); a.add_argument("--layers", type=int, default=2); a.add_argument("--device", default="cuda")
a.add_argument("--max-seqs", type=int, default=0)
args = a.parse_args()
device = torch.device(args.device); md = Path(args.model_dir)
model = MaskPredictor(hidden=args.hidden, layers=args.layers).to(device)
model.load_state_dict(torch.load(md / "mask_model.pt", map_location=device)); model.eval()
cal = json.load(open(md / "calibration.json")) if (md / "calibration.json").exists() else {"a": 1.0, "b": 0.0}
print("calibration:", cal)
pf = pq.ParquetFile(args.train); hs, ht = file_columns(pf)
n = args.max_seqs or pf.num_row_groups
arr = np.lib.format.open_memmap(md / "mask_probs_train.npy", mode="w+", dtype=np.float16, shape=(n, SEQ_LEN))
sum_p = sum_rows = 0.0
for g in tqdm(range(n), desc="label train", unit="seq"):
    s = read_sequence(pf, g, has_is_scored=hs, has_targets=ht)
    p = torch.sigmoid(cal["a"] * predict_sequence(model, torch.from_numpy(s.features).to(device)) + cal["b"]).cpu().numpy()
    arr[g] = p.astype(np.float16)
    sum_p += float(p[s.need_prediction].sum()); sum_rows += float(s.need_prediction.sum())
arr.flush()
stats = {"sequences": int(n), "mean_prob_on_need_rows": sum_p / sum_rows}
(md / "mask_probs_train_stats.json").write_text(json.dumps(stats, indent=2)); print(stats)
