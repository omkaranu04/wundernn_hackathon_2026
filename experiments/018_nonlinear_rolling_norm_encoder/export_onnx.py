"""Export `NonlinearRollingNormGroupedEncoderLSTMModel` to a single-step
ONNX graph.

Combines 012's and 013's export shapes: the streaming state carries the
LSTM's (h, c) AND 013's rolling EWMA statistics (mean/meansq, each shape
(N_WINDOWS,N_PRICE)=(2,44)) - identical state contract to 013's exporter,
unchanged. The nonlinear expansion (012's `expand()`) is stateless/
pointwise and adds nothing to the streaming state, only to the per-step
encoder's input width (see model.py's module docstring for the exact
combination math).

Same two-computation-path subtlety as 013 (see its model.py docstring):
`rolling_normalize_batched` (training, truncated-kernel conv1d
approximation) and `rolling_normalize_step` (export, exact recursion)
compute the EWMA slightly differently by construction - `verify()` below
uses the same atol=1e-2 established there for the same reason. The
nonlinear transforms (`expand()`) are the exact same function on both
paths (no training/export discrepancy) - only the EWMA half has this.

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

from model import (
    NonlinearRollingNormGroupedEncoderLSTMModel,
    expand,
    rolling_normalize_step,
    initial_rolling_state,
    ROLLING_STATE_NAMES,
    N_PRICE,
)


class OnnxStepModel(nn.Module):
    def __init__(self, model: NonlinearRollingNormGroupedEncoderLSTMModel):
        super().__init__()
        model.eval()
        self.model = model

    def forward(self, features: torch.Tensor, h: torch.Tensor, c: torch.Tensor, mean: torch.Tensor, meansq: torch.Tensor):
        price_raw, volume_raw, trade_raw, additional_raw = self.model.raw_groups(features)
        price_nonlinear = expand(price_raw)
        rolling_state = {"mean": mean, "meansq": meansq}
        price_ewma, new_rolling_state = rolling_normalize_step(price_raw, rolling_state)
        encoded = self.model.encode_from_price_parts(
            price_nonlinear, price_ewma, volume_raw, trade_raw, additional_raw
        ).view(1, 1, -1)
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


def export(model: NonlinearRollingNormGroupedEncoderLSTMModel, output_path, opset: int = 18) -> None:
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


def verify(model: NonlinearRollingNormGroupedEncoderLSTMModel, onnx_path, n_steps: int = 3000, atol: float = 1e-2) -> None:
    """atol matches 013's: the EWMA half of this model has the same
    training/export discrepancy 013 validated (truncated-conv approximation
    vs. exact recursion) - see that experiment's export_onnx.py docstring
    for the full measured-discrepancy story. The nonlinear half (`expand()`)
    is identical on both paths and contributes no discrepancy of its own."""
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    # Heterogeneous per-column scale, same rationale as 013's verify():
    # EWMA normalization errors are worst on low-variance features, so a
    # uniform-scale randn test would understate real risk.
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
    model = NonlinearRollingNormGroupedEncoderLSTMModel(
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
