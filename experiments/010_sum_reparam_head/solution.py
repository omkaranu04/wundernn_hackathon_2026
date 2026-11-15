"""Competition submission entrypoint for experiment 010 (004's grouped-encoder
LSTM architecture, but the head predicts (t0, t0+t1) instead of (t0, t1) -
see model.py). The sum-reparam reconstruction happens inside the exported
ONNX graph, so `prediction` here is already in raw [t0, t1] units.

Runs on ONNX Runtime only (numpy + onnxruntime, no torch needed at inference
time) - same approach as the baseline and prior experiments.

Packaging note: this file must sit at the ZIP root alongside
`checkpoints/model.onnx` AND `checkpoints/model.onnx.data` (the ONNX
exporter's external weight-data file - easy to miss, see experiment 001's
README for the near-miss this caused there), loaded via paths relative to
this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "model.onnx"

_OUTPUT_NAMES = ["prediction", "next_h", "next_c"]


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
        self.current_seq_ix = None
        self.state = {k: v.copy() for k, v in self._zero_state.items()}

    def predict(self, data_point) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state = {k: v.copy() for k, v in self._zero_state.items()}

        features = np.asarray(data_point.state, dtype=np.float32)
        prediction, next_h, next_c = self.session.run(_OUTPUT_NAMES, {"features": features, **self.state})
        self.state = {"h": next_h, "c": next_c}

        if not data_point.need_prediction:
            return None
        return np.asarray(prediction, dtype=np.float32)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
