"""Competition submission entrypoint for experiment 011: grouped-encoder +
2-layer LSTM backbone (identical shape to experiment 004), with an XGBoost
regressor head instead of a neural Linear/MLP head - one independent
XGBRegressor per target, trained on the frozen backbone's hidden states
(see model.py / extract_embeddings.py / train_xgboost.py for the full
two-stage story - XGBoost isn't differentiable, so it can't be trained
jointly with the LSTM the way a neural head can).

Three onnxruntime sessions run per step: the backbone (features + LSTM
state -> hidden state + new state), and the two XGBoost heads (hidden
state -> t0, hidden state -> t1). Still numpy + onnxruntime only at
inference time, no torch and no xgboost needed - the XGBoost models were
converted to ONNX ahead of time (train_xgboost.py), specifically so this
doesn't depend on `xgboost` being present in the real scoring container
(only numpy/onnxruntime/pyarrow are guaranteed).

Packaging note: this file must sit at the ZIP root alongside
`checkpoints/backbone.onnx` (+ `.onnx.data` if present), `checkpoints/
xgb_t0.onnx`, and `checkpoints/xgb_t1.onnx`, loaded via paths relative to
this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
_BACKBONE_PATH = _CKPT_DIR / "backbone.onnx"
_XGB_T0_PATH = _CKPT_DIR / "xgb_t0.onnx"
_XGB_T1_PATH = _CKPT_DIR / "xgb_t1.onnx"

_BACKBONE_OUTPUT_NAMES = ["hidden", "next_h", "next_c"]


def _make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.use_per_session_threads = True
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class PredictionModel:
    def __init__(self):
        self.backbone = _make_session(_BACKBONE_PATH)
        self.xgb_t0 = _make_session(_XGB_T0_PATH)
        self.xgb_t1 = _make_session(_XGB_T1_PATH)

        shapes = {i.name: list(i.shape) for i in self.backbone.get_inputs()}
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
        hidden, next_h, next_c = self.backbone.run(
            _BACKBONE_OUTPUT_NAMES, {"features": features, **self.state}
        )
        self.state = {"h": next_h, "c": next_c}

        if not data_point.need_prediction:
            return None

        hidden_2d = hidden.reshape(1, -1)
        t0 = self.xgb_t0.run(None, {"input": hidden_2d})[0]
        t1 = self.xgb_t1.run(None, {"input": hidden_2d})[0]
        prediction = np.array([float(t0.reshape(-1)[0]), float(t1.reshape(-1)[0])], dtype=np.float32)
        return prediction


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
