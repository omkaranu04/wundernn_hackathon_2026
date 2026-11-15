"""Competition entrypoint for graphs written by export_slim.py.

Same contract as solution.py (ONNX Runtime only, one row per call, state reset
on a new sequence), but it takes the recurrent state layout from the graph
itself - any inputs other than `features`, each with a `next_<name>` output -
and flattens the prediction, so it runs the 13-node slim graph (four state
tensors) as well as any merged / multi-member variant.

Packaging: this file is copied to the ZIP root as `solution.py`, next to
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
        self._has_pos = inputs.pop("pos", None) is not None   # LeanX graphs: position in the sequence (from step_in_seq)
        if self._has_pos:  # precomputed [s, sin 2 pi s, cos 2 pi s], s = step / 20000: a lookup per row, no trig
            s = np.arange(20000, dtype=np.float64) / 20000.0
            self._pos_table = np.stack([s, np.sin(2 * np.pi * s), np.cos(2 * np.pi * s)], axis=1).astype(np.float32)
        self._state_names = list(inputs)
        self._output_names = ["prediction"] + ["next_" + n for n in self._state_names]
        self._zero_state = {n: np.zeros(s, dtype=np.float32) for n, s in inputs.items()}
        self.current_seq_ix = None
        self.state = dict(self._zero_state)

    def predict(self, data_point) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state = dict(self._zero_state)   # run() never writes into its inputs, so sharing the zeros is safe

        features = np.asarray(data_point.state, dtype=np.float32).reshape(self._feature_shape)
        feed = {"features": features, **self.state}
        if self._has_pos:
            feed["pos"] = self._pos_table[min(int(data_point.step_in_seq), 19999)]
        out = self.session.run(self._output_names, feed)
        self.state = dict(zip(self._state_names, out[1:]))

        if not data_point.need_prediction:
            return None
        return np.asarray(out[0], dtype=np.float32).reshape(-1)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
