"""Competition entrypoint running the ensemble with the numba engine v3 (040, 2026-11-07): one compiled call per row.

Zip layout:
    solution.py                 (this file)
    numba_engine3.py            (Engine: flattens the npz members, one njit call per row for all members)
    checkpoints/ens.npz         (members, int16 recurrent/head weights, per-member weights)
    checkpoints/model.onnx      (FALLBACK: an ONNX graph run with plain session.run if numba cannot be imported)

One compiled numba call per row for the whole ensemble; the kernel is are compiled once in __init__ (a few seconds), never during the
scored stream. Deterministic: fixed operation order, no threads, no randomness.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))


class _OnnxPlain:
    """Fallback: the plain session.run loop of solution_slim.py (the path that scored 0.66658 in 50m47s).
    NOT IOBinding: that path ran 61m22s on the platform (11-07) despite being faster locally."""

    def __init__(self):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.use_per_session_threads = True
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        o.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = s = ort.InferenceSession(str(_DIR / "checkpoints" / "model.onnx"), sess_options=o, providers=["CPUExecutionProvider"])
        inputs = {i.name: [d if isinstance(d, int) else 1 for d in i.shape] for i in s.get_inputs()}
        self._fshape = inputs.pop("features")
        self._names = list(inputs)
        self._outs = ["prediction"] + ["next_" + n for n in self._names]
        self._zero = {n: np.zeros(sh, np.float32) for n, sh in inputs.items()}
        self.state = dict(self._zero)

    def reset(self):
        self.state = dict(self._zero)

    def step(self, x):
        out = self.session.run(self._outs, {"features": x.reshape(self._fshape), **self.state})
        self.state = dict(zip(self._names, out[1:]))
        return np.asarray(out[0], dtype=np.float32).reshape(-1)


class PredictionModel:
    def __init__(self):
        self.backend = None
        if not os.environ.get("WNN_FORCE_ONNX"):
            try:
                from numba_engine3 import Engine
                self.engine = Engine(str(_DIR / "checkpoints" / "ens.npz"))
                self.backend = "numba"
            except Exception as e:  # numba missing or failing: fall back to the ONNX graph
                print(f"[solution] numba engine unavailable ({type(e).__name__}: {e}); using the ONNX fallback", flush=True)
        if self.backend is None:
            self.engine = _OnnxPlain()
            self.backend = "onnx"
        self.current_seq_ix = None

    def predict(self, data_point) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.engine.reset()
        p = self.engine.step(np.asarray(data_point.state, dtype=np.float32).reshape(-1))
        if not data_point.need_prediction:
            return None
        return p.astype(np.float32, copy=True).reshape(-1)


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(_DIR.parent.parent / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
