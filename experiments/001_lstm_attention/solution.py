"""Competition submission entrypoint for experiment 001 (LSTM + causal windowed attention).

Runs on ONNX Runtime, not raw PyTorch - see `export_onnx.py` for why: pure
PyTorch eager inference (`inference.py::StreamingPredictor`) was measured at
~7.8x over the competition's 60-minute-for-the-whole-test-set budget even
after real optimizations, while the exported ONNX graph, further optimized,
got the model under budget. This mirrors the baseline's own approach
(`baseline/solution.py` also uses ONNX Runtime, not raw PyTorch) and only
needs `numpy`/`onnxruntime` at inference time - same guaranteed-present
dependencies as the baseline (see `requirements.txt`), no `torch` import
needed here at all.

Packaging note: per the Submission guide, this file must sit at the ZIP root
alongside its model artifact (`checkpoints/model.onnx` - NOT `best.pt`/
`config.json`, which are training-time-only and not needed for inference),
loaded via a path relative to this file - see `_MODEL_PATH` below.

STATUS: untested end-to-end. `export_onnx.py` has been verified against both
`model.forward()` and `StreamingPredictor` on randomly-initialized weights
(architecture/export correctness), and latency-benchmarked at the settled
config (d_model=24, lstm_layers=1, attention_window=32, ff_multiplier=2:
~0.0855 ms/step, ~88.5% of budget - see EXPERIMENTS.md). Not yet trained to
completion or scored end-to-end via ScorerStepByStep. Do NOT submit until:
  1. trained (train.py) on real data at the settled architecture,
  2. exported (export_onnx.py) to checkpoints/model.onnx,
  3. scored locally against datasets/valid.parquet via
     wnn_connectome_starterpack.utils.ScorerStepByStep (the __main__ block
     below does this) to get a real WP number,
  4. confirmed to produce identical predictions across repeated runs
     (determinism is a hard submission requirement) and to finish the full
     validation set within a proportionally scaled time check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "model.onnx"

# Matches export_onnx.py's OUTPUT_NAMES[1:] / STATE_NAMES ordering exactly -
# duplicated here (not imported) so this file has no dependency on
# export_onnx.py/model.py/torch at inference time, only onnxruntime/numpy.
_OUTPUT_NAMES = [
    "prediction", "next_h0", "next_c0", "next_h1", "next_c1",
    "next_cache_k", "next_cache_v", "next_valid_mask", "next_write_pos",
]
_STATE_NAMES = ["h0", "c0", "h1", "c1", "cache_k", "cache_v", "valid_mask", "write_pos"]


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

        # Read state shapes off the graph itself rather than hardcoding them,
        # so this file doesn't need to know d_model/window/n_heads/head_dim -
        # those live only in checkpoints/config.json (train.py/export_onnx.py
        # concerns), keeping this submission entrypoint self-contained.
        shapes = {i.name: list(i.shape) for i in self.session.get_inputs()}
        self._zero_state = {
            "h0": np.zeros(shapes["h0"], dtype=np.float32), "c0": np.zeros(shapes["c0"], dtype=np.float32),
            "h1": np.zeros(shapes["h1"], dtype=np.float32), "c1": np.zeros(shapes["c1"], dtype=np.float32),
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
        self.state = dict(zip(_STATE_NAMES, next_state_values))  # must update every row, incl. warm-up

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
