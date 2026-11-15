"""Export `RollingNormGroupedEncoderLSTMModel` to a single-step ONNX graph.

Unlike every other experiment, this one carries TWO independent kinds of
streaming state through `session.run()` calls: the LSTM's own (h, c) AND
the rolling EWMA statistics (mean/meansq, each shape (N_WINDOWS,N_PRICE))
used to build the rolling-normalized PRICE input features (only price is
rolling-normalized as of the 2026-09-23 group-scoping fix - see model.py's
module docstring). Both reset to zero at the start of every new sequence,
same as h/c always have.

UPDATE HISTORY (2026-09-23, see model.py for the full story):
- latency fix #1 (vectorization): rolling state used to be 8 separate
  named (112,) tensors, one pair per window, computed via a Python loop.
  Now 2 named (N_WINDOWS,112) tensors, one vectorized op.
- latency fix #2 (2 windows instead of 4): halved N_WINDOWS.
- latency fix #3 (group-scoping to price only): rolling state is now
  shaped (N_WINDOWS,N_PRICE) instead of (N_WINDOWS,112) - only price's 44
  columns are ever rolling-normalized; volume/trade/additional are raw,
  untouched, and no longer need any block-major re-slicing machinery.

The rolling-state tensors are explicit named forward() parameters (not
*args) - the dynamo-based exporter (used throughout this project; see
experiment 002's README for why the legacy exporter isn't safe here)
handles a fixed, explicit signature more reliably than variadic args.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from model import RollingNormGroupedEncoderLSTMModel, rolling_normalize_step, initial_rolling_state, ROLLING_STATE_NAMES, N_PRICE


class OnnxStepModel(nn.Module):
    def __init__(self, model: RollingNormGroupedEncoderLSTMModel):
        super().__init__()
        model.eval()
        self.model = model

    def forward(self, features: torch.Tensor, h: torch.Tensor, c: torch.Tensor, mean: torch.Tensor, meansq: torch.Tensor):
        price_raw, volume_raw, trade_raw, additional_raw = self.model.raw_groups(features)
        rolling_state = {"mean": mean, "meansq": meansq}
        price_expanded, new_rolling_state = rolling_normalize_step(price_raw, rolling_state)
        encoded = self.model.encode_from_price_expanded(price_expanded, volume_raw, trade_raw, additional_raw).view(1, 1, -1)
        lstm_out, (new_h, new_c) = self.model.lstm(encoded, (h, c))
        prediction = self.model.head(lstm_out.view(-1))
        new_state_tensors = tuple(new_rolling_state[name] for name in ROLLING_STATE_NAMES)
        return (prediction, new_h, new_c) + new_state_tensors


INPUT_NAMES = ["features", "h", "c"] + ROLLING_STATE_NAMES
OUTPUT_NAMES = ["prediction", "next_h", "next_c"] + [f"next_{name}" for name in ROLLING_STATE_NAMES]


def initial_state(num_layers: int, hidden_size: int):
    state = {"h": torch.zeros(num_layers, 1, hidden_size), "c": torch.zeros(num_layers, 1, hidden_size)}
    state.update(initial_rolling_state())
    return state


def export(model: RollingNormGroupedEncoderLSTMModel, output_path, opset: int = 18) -> None:
    model.eval()
    step_model = OnnxStepModel(model)
    step_model.eval()
    state = initial_state(model.num_layers, model.hidden_size)
    example_inputs = (torch.zeros(112), state["h"], state["c"]) + tuple(state[name] for name in ROLLING_STATE_NAMES)
    torch.onnx.export(
        step_model, example_inputs, str(output_path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        opset_version=opset, do_constant_folding=True,
    )
    import onnx
    from onnxsim import simplify as onnxsim_simplify

    model_proto = onnx.load(str(output_path))
    simplified, ok = onnxsim_simplify(model_proto)
    if ok:
        onnx.save(simplified, str(output_path))


def verify(model: RollingNormGroupedEncoderLSTMModel, onnx_path, n_steps: int = 3000, atol: float = 1e-2) -> None:
    """atol is looser than other experiments' ~1e-4-1e-6: training uses a
    truncated-kernel convolution approximation of the EWMA (fast, batched)
    while export uses the exact recursion (streaming-correct) - validated
    close but NOT bit-identical by construction. Isolated testing on the
    intermediate rolling-normalized features (see EXPERIMENTS.md) found
    occasional outliers up to ~0.055 on specific low-variance
    features/timesteps (division-by-small-std amplifies small upstream
    mean/var estimation error) - but end-to-end, through the encoder/LSTM/
    head, that washed out to ~5e-8 on random-init weights (the network's
    combination across features/time dilutes a rare single-channel spike).
    atol=1e-2 keeps real margin above what's been measured without being
    so loose it stops catching genuine export bugs."""
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    # Uniform-scale randn understates the real risk here: the discrepancy
    # between the two computation paths is amplified specifically on
    # LOW-VARIANCE features (dividing a small mean/var estimation mismatch
    # by a small std). Real features have heterogeneous scale (004's
    # docstring: price ~0.15 std, volume ~0.7, additional ~1.45) - so this
    # synthetic test mimics that spread rather than using one flat scale,
    # to actually exercise the worst case instead of an easy one.
    col_scale = torch.linspace(0.05, 2.0, 112)
    features_seq = torch.randn(n_steps, 112) * col_scale

    model.eval()
    with torch.no_grad():
        batched_out, _ = model(features_seq.unsqueeze(0))
    batched_out = batched_out.squeeze(0).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state = {k: v.numpy() for k, v in initial_state(model.num_layers, model.hidden_size).items()}
    onnx_out = []
    for i in range(n_steps):
        outputs = session.run(OUTPUT_NAMES, {"features": features_seq[i].numpy(), **state})
        out_dict = dict(zip(OUTPUT_NAMES, outputs))
        onnx_out.append(out_dict["prediction"])
        state = {"h": out_dict["next_h"], "c": out_dict["next_c"]}
        for name in ROLLING_STATE_NAMES:
            state[name] = out_dict[f"next_{name}"]
    onnx_out = np.stack(onnx_out)

    diff = np.abs(onnx_out - batched_out).max()
    print(f"ONNX vs model.forward() (batched): max abs diff = {diff:.3e}")
    assert diff < atol, f"ONNX output diverges from batched forward: {diff}"
    print("verify() PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output", default="checkpoints/model.onnx")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    model = RollingNormGroupedEncoderLSTMModel(
        hidden_size=config["hidden_size"], num_layers=config["num_layers"],
        price_dim=config["price_dim"], volume_dim=config["volume_dim"],
        trade_dim=config["trade_dim"], head_hidden=config["head_hidden"], dropout=0.0,
    )
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
        print(f"loaded trained weights from {best_path}")
    else:
        print(f"WARNING: {best_path} not found - exporting randomly-initialized weights (testing only)")
    model.eval()

    export(model, args.output)
    print(f"exported to {args.output}")
    if not args.skip_verify:
        verify(model, args.output)
