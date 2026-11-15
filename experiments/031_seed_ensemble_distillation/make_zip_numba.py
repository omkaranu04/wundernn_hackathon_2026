"""Package a numba-engine ensemble (+ an ONNX fallback graph) as a submission zip and verify it (039).

    python make_zip_numba.py --npz results/numba/X.npz --fallback results/slim/Y_int8.onnx --name 039_numba_something

Zip: solution.py (= solution_numba.py, or solution_numba3.py with --engine 3), numba_engine{2,3}.py, checkpoints/ens.npz, checkpoints/model.onnx. Verification
from a fresh extraction: (1) the numba path on 2 real valid sequences (a sequence switch included) against the
torch members named in the npz (weighted mean), (2) the fallback path (WNN_FORCE_ONNX=1) runs and is finite,
(3) contract: None on warm-up rows, (2,) float32 finite otherwise.
"""
import argparse, datetime, hashlib, importlib.util, json, os, pathlib, sys, tempfile, types, zipfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared")); sys.path.insert(0, str(EXP))
import pyarrow.parquet as pq
from data import file_columns, read_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--engine", type=int, default=2, choices=[2, 3, 4, 5, 7]); ap.add_argument("--npz", required=True); ap.add_argument("--fallback", required=True); ap.add_argument("--name", required=True)
a = ap.parse_args()
name = f"{a.name}_{datetime.date.today().isoformat()}"
zp = ROOT / "submissions" / f"{name}.zip"
if zp.exists():
    raise SystemExit(f"{zp} already exists - not overwriting")
eng = f"numba_engine{a.engine}.py"
src = {"solution.py": EXP / {2: "solution_numba.py", 3: "solution_numba3.py", 4: "solution_numba4.py", 5: "solution_numba5.py", 7: "solution_numba7.py"}[a.engine], eng: EXP / eng,
       "checkpoints/ens.npz": pathlib.Path(a.npz), "checkpoints/model.onnx": pathlib.Path(a.fallback)}
if a.engine in (5, 7):
    src["numba_engine4.py"] = EXP / "numba_engine4.py"
if a.engine == 7:
    src["numba_engine5.py"] = EXP / "numba_engine5.py"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, f in src.items():
        z.write(f, arc)
md5 = lambda b: hashlib.md5(b).hexdigest()
meta = json.loads(bytes(np.load(a.npz)["meta"]).decode())
pf = pq.ParquetFile(ROOT / "wnn_connectome_starterpack/datasets/valid.parquet"); hs, ht = file_columns(pf)
seqs = [read_sequence(pf, s, has_is_scored=hs, has_targets=ht) for s in (11, 500)]
N = 1200


def run(mod, force_onnx):
    if force_onnx:
        os.environ["WNN_FORCE_ONNX"] = "1"
    else:
        os.environ.pop("WNN_FORCE_ONNX", None)
    pm = mod.PredictionModel(); outs = []
    for sid, q in zip((11, 500), seqs):
        for r in range(N):
            p = pm.predict(types.SimpleNamespace(seq_ix=sid, step_in_seq=r, need_prediction=bool(q.need_prediction[r]), state=q.features[r]))
            assert (p is None) == (not q.need_prediction[r])
            if p is not None:
                assert p.shape == (2,) and p.dtype == np.float32 and np.isfinite(p).all()
            outs.append(p)
    os.environ.pop("WNN_FORCE_ONNX", None)
    return pm.backend, outs


with tempfile.TemporaryDirectory(prefix="verify_zip_") as tmp:
    out = pathlib.Path(tmp)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None; z.extractall(out)
    for arc, f in src.items():
        assert md5((out / arc).read_bytes()) == md5(pathlib.Path(f).read_bytes()), arc
    spec = importlib.util.spec_from_file_location("sol_" + md5(name.encode())[:8], out / "solution.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    be, got = run(mod, False)
    assert be == {5: "numba5", 7: "numba7"}.get(a.engine, "numba"), be
    be2, fb = run(mod, True)
    assert be2 == "onnx", be2
# torch reference for the numba path (weighted mean of the members, fresh state per sequence)
import torch
from teacher_preds import load_model
ref = []
with torch.no_grad():
    for q in seqs:
        x = torch.from_numpy(q.features[:N].astype(np.float32))[None]
        r = sum(m["weight"] * load_model(m["spec"], torch.device("cpu"))(x)[0][0].numpy() for m in meta)
        ref += [r[i] if q.need_prediction[i] else None for i in range(N)]
d = max(float(np.abs(g - r).max()) for g, r in zip(got, ref) if g is not None)
dfb = max(float(np.abs(g - r).max()) for g, r in zip(fb, ref) if g is not None)
n = sum(g is not None for g in got)
print(f"ZIP OK {zp.name}: {zp.stat().st_size / 1e6:.2f} MB, {len(meta)} members ({', '.join(m['arch'] for m in meta)}), "
      f"{n} predictions verified on 2 sequences; numba vs torch max diff {d:.2e}; ONNX fallback vs torch {dfb:.2e}")
