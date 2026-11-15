"""Competition submission entrypoint for experiment 013 (rolling-normalized
grouped-encoder LSTM).

Runs on ONNX Runtime only (numpy + onnxruntime, no torch needed at
inference time). Unlike every prior experiment, this one threads TWO kinds
of state between steps: the LSTM's (h, c) AND the rolling EWMA statistics
(mean/meansq, each shape (N_WINDOWS,112), vectorized across all 4 windows
at once as of the 2026-09-23 latency fix - see model.py/export_onnx.py)
- both reset to zero whenever a new sequence starts.

Packaging note: this file must sit at the ZIP root alongside
`checkpoints/model.onnx` (and `checkpoints/model.onnx.data` if the
exporter produced one), loaded via paths relative to this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "model.onnx"

ROLLING_STATE_NAMES = ["mean", "meansq"]  # each (N_WINDOWS, 112) - see model.py::rolling_normalize_step
_OUTPUT_NAMES = ["prediction", "next_h", "next_c"] + [f"next_{name}" for name in ROLLING_STATE_NAMES]


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
        self.session = ort.InferenceSession(
            str(_MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"],
        )

        shapes = {i.name: list(i.shape) for i in self.session.get_inputs()}
        self._zero_state = {
            "h": np.zeros(shapes["h"], dtype=np.float32),
            "c": np.zeros(shapes["c"], dtype=np.float32),
        }
        for name in ROLLING_STATE_NAMES:
            self._zero_state[name] = np.zeros(shapes[name], dtype=np.float32)
        self.current_seq_ix = None
        self.state = {k: v.copy() for k, v in self._zero_state.items()}

    def predict(self, data_point) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state = {k: v.copy() for k, v in self._zero_state.items()}

        features = np.asarray(data_point.state, dtype=np.float32)
        outputs = self.session.run(_OUTPUT_NAMES, {"features": features, **self.state})
        out_dict = dict(zip(_OUTPUT_NAMES, outputs))
        self.state = {"h": out_dict["next_h"], "c": out_dict["next_c"]}
        for name in ROLLING_STATE_NAMES:
            self.state[name] = out_dict[f"next_{name}"]

        if not data_point.need_prediction:
            return None
        return np.asarray(out_dict["prediction"], dtype=np.float32)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
