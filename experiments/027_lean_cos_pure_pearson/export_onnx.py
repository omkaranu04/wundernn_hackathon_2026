"""Export `LeanCosGroupedEncoderLSTMModel` to a single-step ONNX graph -
same pattern as experiment 004 (LSTM state threaded in/out of
`session.run()` calls). `encode()` (grouping + the 5-way nonlinear
expansion) is a pure function of the current step's raw 112-dim feature
vector, with no state of its own, so it needs no special streaming
handling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from model import LeanCosGroupedEncoderLSTMModel


class OnnxStepModel(nn.Module):
    def __init__(self, model: LeanCosGroupedEncoderLSTMModel):
        super().__init__()
        model.eval()
        self.model = model

    def forward(self, features: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """features: (112,). h, c: (num_layers, 1, hidden_size)."""
        encoded = self.model.encode(features).view(1, 1, -1)
        lstm_out, (new_h, new_c) = self.model.lstm(encoded, (h, c))
        prediction = self.model.head(lstm_out.view(-1))
        return prediction, new_h, new_c


INPUT_NAMES = ["features", "h", "c"]
OUTPUT_NAMES = ["prediction", "next_h", "next_c"]
STATE_NAMES = ["h", "c"]


def initial_state(num_layers: int, hidden_size: int):
    return {"h": torch.zeros(num_layers, 1, hidden_size), "c": torch.zeros(num_layers, 1, hidden_size)}


def export(model: LeanCosGroupedEncoderLSTMModel, output_path, opset: int = 18) -> None:
    model.eval()
    step_model = OnnxStepModel(model)
    step_model.eval()
    state = initial_state(model.num_layers, model.hidden_size)
    example_inputs = (torch.zeros(112), state["h"], state["c"])
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


def verify(model: LeanCosGroupedEncoderLSTMModel, onnx_path, n_steps: int = 800, atol: float = 1e-4) -> None:
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    features_seq = torch.randn(n_steps, 112)

    model.eval()
    with torch.no_grad():
        batched_out, _ = model(features_seq.unsqueeze(0))
    batched_out = batched_out.squeeze(0).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state = {k: v.numpy() for k, v in initial_state(model.num_layers, model.hidden_size).items()}
    onnx_out = []
    for i in range(n_steps):
        prediction, next_h, next_c = session.run(OUTPUT_NAMES, {"features": features_seq[i].numpy(), **state})
        onnx_out.append(prediction)
        state = {"h": next_h, "c": next_c}
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
    model = LeanCosGroupedEncoderLSTMModel(
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
