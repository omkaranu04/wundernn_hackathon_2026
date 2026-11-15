"""Export `SparseAttentionLSTMModel` to a single-step ONNX graph.

Same state-threading pattern as experiments 001/016: 2-layer LSTM state
(h0,c0,h1,c1) plus a circular key/value buffer (size max(OFFSETS)+1=2049),
write position, and step counter for the sparse attention's fixed-offset
validity check. LSTMCell instead of native LSTM, index_copy for the
circular write, matmul/index_select instead of einsum - see experiment
016's export_onnx.py for the full reasoning (identical here, just with a
raw Linear input projection instead of a grouped encoder).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from model import SparseAttentionLSTMModel, OFFSETS, BUFFER_SIZE


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
    def __init__(self, model: SparseAttentionLSTMModel):
        super().__init__()
        model.eval()
        self.d_model = model.d_model
        self.n_heads = model.attention.n_heads
        self.head_dim = model.attention.head_dim
        self.offsets = OFFSETS
        self.buffer_size = BUFFER_SIZE

        self.input_proj = model.input_proj
        self.lstm_cells = build_lstm_cells(model.lstm)
        self.attn_norm = model.attn_norm
        self.qkv = model.attention.qkv
        self.out_proj = model.attention.out_proj
        self.ff_norm = model.ff_norm
        self.feed_forward = model.feed_forward
        self.head = model.head

    def forward(
        self,
        features: torch.Tensor,  # (112,)
        h0: torch.Tensor, c0: torch.Tensor,
        h1: torch.Tensor, c1: torch.Tensor,
        cache_k: torch.Tensor, cache_v: torch.Tensor,
        write_pos: torch.Tensor, step_count: torch.Tensor,
    ):
        x = self.input_proj(features.view(1, -1))
        h0, c0 = self.lstm_cells[0](x, (h0, c0))
        h1, c1 = self.lstm_cells[1](h0, (h1, c1))
        lstm_out = h1.view(self.d_model)

        normed = self.attn_norm(lstm_out)
        qkv = self.qkv(normed)
        q, k, v = qkv.split(self.d_model)
        q = q.view(self.n_heads, self.head_dim)
        k = k.view(1, self.n_heads, self.head_dim)
        v = v.view(1, self.n_heads, self.head_dim)

        new_cache_k = cache_k.index_copy(0, write_pos, k)
        new_cache_v = cache_v.index_copy(0, write_pos, v)

        gathered_k, gathered_v, valid = [], [], []
        for o in self.offsets:
            slot = (write_pos - o) % self.buffer_size
            gathered_k.append(new_cache_k.index_select(0, slot))
            gathered_v.append(new_cache_v.index_select(0, slot))
            valid.append((step_count >= o).float())

        K = torch.cat(gathered_k, dim=0)
        V = torch.cat(gathered_v, dim=0)
        valid_mask = torch.cat(valid, dim=0)

        k_heads_first = K.transpose(0, 1)
        q_col = q.unsqueeze(-1)
        scores = torch.matmul(k_heads_first, q_col).squeeze(-1).transpose(0, 1)
        scores = scores / (self.head_dim ** 0.5)
        scores = scores + (1.0 - valid_mask).unsqueeze(-1) * -1e9
        attn = torch.softmax(scores, dim=0)

        attn_heads_first = attn.transpose(0, 1).unsqueeze(1)
        v_heads_first = V.transpose(0, 1)
        attended = torch.matmul(attn_heads_first, v_heads_first).squeeze(1).reshape(self.d_model)
        attended = self.out_proj(attended)

        x2 = lstm_out + attended
        x2 = x2 + self.feed_forward(self.ff_norm(x2))
        prediction = self.head(x2)

        new_write_pos = (write_pos + 1) % self.buffer_size
        new_step_count = step_count + 1

        return prediction, h0, c0, h1, c1, new_cache_k, new_cache_v, new_write_pos, new_step_count


INPUT_NAMES = ["features", "h0", "c0", "h1", "c1", "cache_k", "cache_v", "write_pos", "step_count"]
OUTPUT_NAMES = ["prediction", "next_h0", "next_c0", "next_h1", "next_c1",
                "next_cache_k", "next_cache_v", "next_write_pos", "next_step_count"]
STATE_NAMES = ["h0", "c0", "h1", "c1", "cache_k", "cache_v", "write_pos", "step_count"]


def initial_state(step_model: OnnxStepModel):
    d, buf, hh, hd = step_model.d_model, step_model.buffer_size, step_model.n_heads, step_model.head_dim
    return {
        "h0": torch.zeros(1, d), "c0": torch.zeros(1, d),
        "h1": torch.zeros(1, d), "c1": torch.zeros(1, d),
        "cache_k": torch.zeros(buf, hh, hd), "cache_v": torch.zeros(buf, hh, hd),
        "write_pos": torch.zeros(1, dtype=torch.int64),
        "step_count": torch.zeros(1, dtype=torch.int64),
    }


def export(model: SparseAttentionLSTMModel, output_path, opset: int = 18) -> None:
    model.eval()
    step_model = OnnxStepModel(model)
    step_model.eval()
    state = initial_state(step_model)
    example_inputs = (
        torch.zeros(112), state["h0"], state["c0"], state["h1"], state["c1"],
        state["cache_k"], state["cache_v"], state["write_pos"], state["step_count"],
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


def verify(model: SparseAttentionLSTMModel, onnx_path, n_steps: int = 3000, atol: float = 1e-4) -> None:
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
    model = SparseAttentionLSTMModel(
        d_model=config["d_model"], lstm_layers=config["lstm_layers"],
        n_heads=config["n_heads"], ff_multiplier=config["ff_multiplier"], dropout=0.0,
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
