"""Competition entrypoint for slim graphs, with ONNX Runtime IOBinding (037-IOB, 2026-11-06).

Same contract and outputs as solution_slim.py (bit-identical predictions), but the per-row call avoids building
input/output dicts and fresh arrays: the features buffer, both recurrent-state buffer sets and the prediction buffer
are allocated ONCE and bound to the session; each step writes the row into the bound features buffer and runs one
of two pre-built bindings (state A -> B, then B -> A). A new sequence zeroes the state buffer that the next run
reads. Graphs with a `pos` input (LeanX) fall back to the plain path.

Packaging: copied to the ZIP root as `solution.py` (make_zip.py --solution solution_iob.py), next to
`checkpoints/model.onnx`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "model.onnx"


class PredictionModel:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(_MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])

        inputs = {i.name: [d if isinstance(d, int) else 1 for d in i.shape] for i in self.session.get_inputs()}
        self._feature_shape = inputs.pop("features")
        self._has_pos = inputs.pop("pos", None) is not None
        if self._has_pos:
            s = np.arange(20000, dtype=np.float64) / 20000.0
            self._pos_table = np.stack([s, np.sin(2 * np.pi * s), np.cos(2 * np.pi * s)], axis=1).astype(np.float32)
        self._state_names = list(inputs)
        self._output_names = ["prediction"] + ["next_" + n for n in self._state_names]
        self._zero_state = {n: np.zeros(s, dtype=np.float32) for n, s in inputs.items()}
        self.current_seq_ix = None
        self.state = dict(self._zero_state)
        if self._has_pos:
            return

        # IOBinding: every buffer allocated once; OrtValues wrap the numpy memory (CPU, contiguous float32)
        self._x = np.zeros(self._feature_shape, dtype=np.float32)
        self._xflat = self._x.reshape(-1)
        self._bufs = [{n: np.zeros(s, dtype=np.float32) for n, s in inputs.items()} for _ in range(2)]
        vals = [{n: ort.OrtValue.ortvalue_from_numpy(b[n]) for n in b} for b in self._bufs]
        out_shape = [d if isinstance(d, int) else 1 for d in self.session.get_outputs()[0].shape]
        self._y = np.zeros(out_shape, dtype=np.float32)
        self._yflat = self._y.reshape(-1)
        x_val = ort.OrtValue.ortvalue_from_numpy(self._x)
        y_val = ort.OrtValue.ortvalue_from_numpy(self._y)
        self._bindings = []
        for src, dst in ((0, 1), (1, 0)):
            b = self.session.io_binding()
            b.bind_ortvalue_input("features", x_val)
            b.bind_ortvalue_output("prediction", y_val)
            for n in self._state_names:
                b.bind_ortvalue_input(n, vals[src][n])
                b.bind_ortvalue_output("next_" + n, vals[dst][n])
            self._bindings.append(b)
        self._run_options = ort.RunOptions()
        self._k = 0  # index of the buffer set the next run READS

    def _predict_plain(self, data_point):
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state = dict(self._zero_state)
        features = np.asarray(data_point.state, dtype=np.float32).reshape(self._feature_shape)
        feed = {"features": features, **self.state}
        if self._has_pos:
            feed["pos"] = self._pos_table[min(int(data_point.step_in_seq), 19999)]
        out = self.session.run(self._output_names, feed)
        self.state = dict(zip(self._state_names, out[1:]))
        if not data_point.need_prediction:
            return None
        return np.asarray(out[0], dtype=np.float32).reshape(-1)

    def predict(self, data_point) -> np.ndarray | None:
        if self._has_pos:
            return self._predict_plain(data_point)
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            for buf in self._bufs[self._k].values():
                buf.fill(0.0)
        self._xflat[:] = np.asarray(data_point.state, dtype=np.float32).reshape(-1)
        self.session.run_with_iobinding(self._bindings[self._k], self._run_options)
        self._k ^= 1
        if not data_point.need_prediction:
            return None
        return self._yflat.copy()


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
