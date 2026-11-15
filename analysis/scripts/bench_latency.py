"""Alternating single-thread ONNX Runtime latency benchmark of two models.

Usage: python analysis/scripts/bench_latency.py REFERENCE.onnx CANDIDATE.onnx [repeats]

Reports "% of budget uncorrected" (ms/step divided by 3600 s / 37,274,573
valid-set steps) for each model, run ALTERNATELY so machine load affects both
equally. Absolute numbers swing with load; trust the ratio. Remember the real
test set is ~5.2% larger than valid (see analysis/results/timing_correlations.md).
"""
import sys, time
import numpy as np, onnxruntime as ort

def bench(path, n=20000):
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.use_per_session_threads = True
    o.add_session_config_entry("session.intra_op.allow_spinning", "0"); o.add_session_config_entry("session.inter_op.allow_spinning", "0")
    s = ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]; shapes = {i.name: list(i.shape) for i in s.get_inputs()}
    types = {i.name: i.type for i in s.get_inputs()}
    st = {k: np.zeros(shapes[k], dtype=np.int64 if "int64" in types[k] else np.float32) for k in names if k != "features"}
    out = [x.name for x in s.get_outputs()]; rng = np.random.default_rng(0)
    def step(f, st):
        r = dict(zip(out, s.run(out, {"features": f, **st}))); return {k: r.get("next_" + k, st[k]) for k in st}
    fs = [d if isinstance(d, int) else 1 for d in shapes["features"]]  # (112,) for standard exports, (1,1,112) for export_slim.build_multi
    for _ in range(300): st = step(rng.standard_normal(fs).astype(np.float32), st)
    F = rng.standard_normal([n] + fs).astype(np.float32); t = time.perf_counter()
    for i in range(n): st = step(F[i], st)
    return (time.perf_counter() - t) / n * 1000 / (3600 * 1000 / 37274573) * 100

if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ra, rb = [], []
    for _ in range(reps): ra.append(bench(a)); rb.append(bench(b))
    print(f"reference  median {np.median(ra):.1f}%  ({' '.join(f'{x:.1f}' for x in ra)})")
    print(f"candidate  median {np.median(rb):.1f}%  ({' '.join(f'{x:.1f}' for x in rb)})")
    print(f"candidate / reference = {np.median(rb)/np.median(ra):.3f}")
