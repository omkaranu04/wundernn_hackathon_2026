"""Competition submission entrypoint for experiment 015 (012's nonlinear
grouped encoder + 1-layer LSTM + causal windowed self-attention, window=512).

Runs on ONNX Runtime only. State threaded per step: the LSTM's (h0, c0)
plus a circular buffer of cached key/value projections for the windowed
attention (cache_k, cache_v, valid_mask, write_pos) - same pattern as
experiment 001's solution.py.

Packaging note: this file must sit at the ZIP root alongside
`checkpoints/model.onnx` (and `checkpoints/model.onnx.data` if present).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "model.onnx"

_OUTPUT_NAMES = ["prediction", "next_h0", "next_c0", "next_cache_k", "next_cache_v", "next_valid_mask", "next_write_pos"]
_STATE_NAMES = ["h0", "c0", "cache_k", "cache_v", "valid_mask", "write_pos"]


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
            "h0": np.zeros(shapes["h0"], dtype=np.float32), "c0": np.zeros(shapes["c0"], dtype=np.float32),
            "cache_k": np.zeros(shapes["cache_k"], dtype=np.float32),
            "cache_v": np.zeros(shapes["cache_v"], dtype=np.float32),
            "valid_mask": np.zeros(shapes["valid_mask"], dtype=np.float32),
            "write_pos": np.zeros(shapes["write_pos"], dtype=np.int64),
        }

        self.current_seq_ix = None
        self.state = {k: v.copy() for k, v in self._zero_state.items()}

    def predict(self, data_point) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state = {k: v.copy() for k, v in self._zero_state.items()}

        features = np.asarray(data_point.state, dtype=np.float32)
        outputs = self.session.run(_OUTPUT_NAMES, {"features": features, **self.state})
        prediction, *next_state_values = outputs
        self.state = dict(zip(_STATE_NAMES, next_state_values))

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
