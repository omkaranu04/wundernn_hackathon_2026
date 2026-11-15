"""Export `NonlinearWindowedAttentionModel` to a single-step ONNX graph.

Same state-threading pattern as experiment 001's export_onnx.py (read that
file's docstring for the full reasoning): the LSTM state (h, c) plus a
circular buffer of cached key/value projections for the windowed attention,
threaded explicitly through session.run() calls. Uses the same proven
techniques: `nn.LSTMCell` instead of native `nn.LSTM` (faster once
integrated with the rest of the graph, per 001's measured negative
result), `index_copy` for the circular buffer write (traces to ONNX
Scatter, no Python branching needed), explicit `matmul` instead of
`einsum` (measured ~4.5x faster in exported ONNX graphs), and a
`valid_mask` + large-negative-bias trick instead of runtime-length slicing
(ONNX tracing can't express "slice to a variable length").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from model import NonlinearWindowedAttentionModel

I0_PRICE, I0_VOLUME, I0_DP, I0_DV = (0, 22), (22, 44), (44, 48), (48, 52)
I1_PRICE, I1_VOLUME, I1_DP, I1_DV = (52, 74), (74, 96), (96, 100), (100, 104)
ADDITIONAL = (104, 112)


def build_lstm_cells(lstm: nn.LSTM) -> nn.ModuleList:
    cells = nn.ModuleList(
        [nn.LSTMCell(lstm.input_size if layer == 0 else lstm.hidden_size, lstm.hidden_size)
         for layer in range(lstm.num_layers)]
    )
    with torch.no_grad():
        for layer, cell in enumerate(cells):
            cell.weight_ih.copy_(getattr(lstm, f"weight_ih_l{layer}"))
            cell.weight_hh.copy_(getattr(lstm, f"weight_hh_l{layer}"))
            cell.bias_ih.copy_(getattr(lstm, f"bias_ih_l{layer}"))
            cell.bias_hh.copy_(getattr(lstm, f"bias_hh_l{layer}"))
    cells.eval()
    return cells


class OnnxStepModel(nn.Module):
    def __init__(self, model: NonlinearWindowedAttentionModel):
        super().__init__()
        model.eval()
        self.hidden_size = model.hidden_size
        self.n_heads = model.attention.n_heads
        self.head_dim = model.attention.head_dim
        self.window = model.attention.window

        self.price_encoder = model.price_encoder
        self.volume_encoder = model.volume_encoder
        self.trade_encoder = model.trade_encoder
        self.lstm_cells = build_lstm_cells(model.lstm)
        self.attn_norm = model.attn_norm
        self.qkv = model.attention.qkv
        self.out_proj = model.attention.out_proj
        self.ff_norm = model.ff_norm
        self.feed_forward = model.feed_forward
        self.head = model.head

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        from model import expand
        price = torch.cat([features[I0_PRICE[0]:I0_PRICE[1]], features[I1_PRICE[0]:I1_PRICE[1]]], dim=-1)
        volume = torch.cat([features[I0_VOLUME[0]:I0_VOLUME[1]], features[I1_VOLUME[0]:I1_VOLUME[1]]], dim=-1)
        trade = torch.cat([
            features[I0_DP[0]:I0_DP[1]], features[I0_DV[0]:I0_DV[1]],
            features[I1_DP[0]:I1_DP[1]], features[I1_DV[0]:I1_DV[1]],
        ], dim=-1)
        additional = features[ADDITIONAL[0]:ADDITIONAL[1]]
        return torch.cat([
            self.price_encoder(expand(price)),
            self.volume_encoder(expand(volume)),
            self.trade_encoder(expand(trade)),
            expand(additional),
        ], dim=-1)

    def forward(
        self,
        features: torch.Tensor,  # (112,)
        h0: torch.Tensor, c0: torch.Tensor,  # (1, hidden_size)
        cache_k: torch.Tensor,   # (window, n_heads, head_dim)
        cache_v: torch.Tensor,   # (window, n_heads, head_dim)
        valid_mask: torch.Tensor,  # (window,)
        write_pos: torch.Tensor,   # (1,) int64
    ):
        x = self.encode(features).view(1, -1)  # (1, lstm_input_size)
        h0, c0 = self.lstm_cells[0](x, (h0, c0))
        lstm_out = h0.view(self.hidden_size)

        normed = self.attn_norm(lstm_out)
        qkv = self.qkv(normed)
        q, k, v = qkv.split(self.hidden_size)
        q = q.view(self.n_heads, self.head_dim)
        k = k.view(1, self.n_heads, self.head_dim)
        v = v.view(1, self.n_heads, self.head_dim)

        new_cache_k = cache_k.index_copy(0, write_pos, k)
        new_cache_v = cache_v.index_copy(0, write_pos, v)
        new_valid_mask = valid_mask.index_copy(0, write_pos, torch.ones(1, dtype=valid_mask.dtype))

        k_heads_first = new_cache_k.transpose(0, 1)
        q_col = q.unsqueeze(-1)
        scores = torch.matmul(k_heads_first, q_col).squeeze(-1).transpose(0, 1)
        scores = scores / (self.head_dim ** 0.5)
        scores = scores + (1.0 - new_valid_mask).unsqueeze(-1) * -1e9
        attn = torch.softmax(scores, dim=0)

        attn_heads_first = attn.transpose(0, 1).unsqueeze(1)
        v_heads_first = new_cache_v.transpose(0, 1)
        attended = torch.matmul(attn_heads_first, v_heads_first).squeeze(1).reshape(self.hidden_size)
        attended = self.out_proj(attended)

        x2 = lstm_out + attended
        x2 = x2 + self.feed_forward(self.ff_norm(x2))
        prediction = self.head(x2)

        new_write_pos = (write_pos + 1) % self.window

        return prediction, h0, c0, new_cache_k, new_cache_v, new_valid_mask, new_write_pos


INPUT_NAMES = ["features", "h0", "c0", "cache_k", "cache_v", "valid_mask", "write_pos"]
OUTPUT_NAMES = ["prediction", "next_h0", "next_c0", "next_cache_k", "next_cache_v", "next_valid_mask", "next_write_pos"]
STATE_NAMES = ["h0", "c0", "cache_k", "cache_v", "valid_mask", "write_pos"]


def initial_state(step_model: OnnxStepModel):
    h, w, hh, hd = step_model.hidden_size, step_model.window, step_model.n_heads, step_model.head_dim
    return {
        "h0": torch.zeros(1, h), "c0": torch.zeros(1, h),
        "cache_k": torch.zeros(w, hh, hd), "cache_v": torch.zeros(w, hh, hd),
        "valid_mask": torch.zeros(w, dtype=torch.float32),
        "write_pos": torch.zeros(1, dtype=torch.int64),
    }


def export(model: NonlinearWindowedAttentionModel, output_path, opset: int = 18) -> None:
    model.eval()
    step_model = OnnxStepModel(model)
    step_model.eval()
    state = initial_state(step_model)
    example_inputs = (
        torch.zeros(112), state["h0"], state["c0"],
        state["cache_k"], state["cache_v"], state["valid_mask"], state["write_pos"],
    )
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


def verify(model: NonlinearWindowedAttentionModel, onnx_path, n_steps: int = 800, atol: float = 1e-4) -> None:
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    features_seq = torch.randn(n_steps, 112)

    model.eval()
    with torch.no_grad():
        batched_out, _ = model(features_seq.unsqueeze(0))
    batched_out = batched_out.squeeze(0).numpy()

    step_model = OnnxStepModel(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state = {k: v.numpy() for k, v in initial_state(step_model).items()}
    onnx_out = []
    for i in range(n_steps):
        outputs = session.run(OUTPUT_NAMES, {"features": features_seq[i].numpy(), **state})
        prediction, *next_state_values = outputs
        onnx_out.append(prediction)
        state = dict(zip(STATE_NAMES, next_state_values))
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
    model = NonlinearWindowedAttentionModel(
        hidden_size=config["hidden_size"], num_layers=config["num_layers"],
        price_dim=config["price_dim"], volume_dim=config["volume_dim"], trade_dim=config["trade_dim"],
        n_heads=config["n_heads"], attention_window=config["attention_window"],
        ff_multiplier=config["ff_multiplier"], dropout=0.0,
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
