"""Package one slim ONNX graph as a submission zip and verify it.

    python make_zip.py --model results/slim/X_int8.onnx --name 031_something_slim_int8

Writes submissions/<name>_<today>.zip = solution.py (a copy of solution_slim.py, or of --solution)
+ checkpoints/model.onnx, never overwriting an existing zip. Then extracts it
into a fresh scratch folder, checks the bytes, imports the extracted
solution.py and runs PredictionModel.predict over 1,200 rows of a valid
sequence (None for warm-up rows, shape (2,), float32, finite).
"""
import argparse, datetime, hashlib, importlib.util, pathlib, sys, tempfile, types, zipfile
import numpy as np, onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared"))
import pyarrow.parquet as pq
from data import file_columns, read_sequence

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--solution", default="solution_slim.py", help="entrypoint copied as solution.py (e.g. solution_iob.py)")
a = ap.parse_args()
name = f"{a.name}_{datetime.date.today().isoformat()}"
zp = ROOT / "submissions" / f"{name}.zip"
if zp.exists():
    raise SystemExit(f"{zp} already exists - not overwriting")
src = {"solution.py": EXP / a.solution, "checkpoints/model.onnx": pathlib.Path(a.model)}
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, f in src.items():
        z.write(f, arc)
md5 = lambda b: hashlib.md5(b).hexdigest()
with tempfile.TemporaryDirectory(prefix="verify_zip_") as tmp:
    out = pathlib.Path(tmp)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None; z.extractall(out)
    for arc, f in src.items():
        assert md5((out / arc).read_bytes()) == md5(pathlib.Path(f).read_bytes()), arc
    m = onnx.load(str(out / "checkpoints/model.onnx"))
    spec = importlib.util.spec_from_file_location("sol_" + md5(name.encode())[:8], out / "solution.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); pm = mod.PredictionModel()
    pf = pq.ParquetFile(ROOT / "wnn_connectome_starterpack/datasets/valid.parquet"); hs, ht = file_columns(pf)
    seq = read_sequence(pf, 11, has_is_scored=hs, has_targets=ht); n = 0
    for r in range(1200):
        p = pm.predict(types.SimpleNamespace(seq_ix=11, step_in_seq=r, need_prediction=bool(seq.need_prediction[r]), state=seq.features[r]))
        assert (p is None) == (not seq.need_prediction[r])
        if p is not None:
            assert p.shape == (2,) and p.dtype == np.float32 and np.isfinite(p).all(); n += 1
    states = [tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in m.graph.input][1:]
print(f"ZIP OK {zp.name}: {zp.stat().st_size / 1e6:.2f} MB, {len(m.graph.node)} nodes, state {states[0]} x{len(states)}, {n} predictions verified")
