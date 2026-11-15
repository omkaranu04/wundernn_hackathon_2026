"""Export `LSTMAttentionModel` to a single-step ONNX graph, mirroring the
baseline's approach (`baseline/solution.py`, `baseline.onnx`): the platform
calls a model once per row, so the exported graph takes the recurrent/
attention state as explicit inputs and returns the updated state as explicit
outputs, exactly like the baseline threads `hidden_0`/`hidden_1` through
`onnxruntime.InferenceSession.run()` calls.

Why this exists: `inference.py`'s `StreamingPredictor` (pure PyTorch eager
execution) was measured at ~0.75 ms/step even after real optimizations
(circular-buffer cache, `nn.LSTMCell`) - about 7.8x over the competition's
60-minute-for-the-whole-test-set budget. Profiling showed ~0.29 ms of that is
Python/dispatch overhead from chaining ~12 separate eager ops, largely
independent of model size - eager PyTorch is the wrong execution strategy for
this per-row contract at this budget. ONNX Runtime avoids most of that
overhead by running a compiled graph instead (this is very likely why the
baseline itself ships an ONNX model rather than raw PyTorch, and hits
~0.076 ms/step doing comparably simple GRU math). See EXPERIMENTS.md for the
measurements that motivated this.

Design difference from `StreamingPredictor`'s state (important - read before
changing either): `StreamingPredictor` tracks a Python `int` `cache_len` and
slices `cache_k[:cache_len]` while the window is still filling up. ONNX
tracing captures one fixed computation graph - it cannot express "slice to a
runtime-varying length" or an `if cache_len == window` branch. So the state
here instead keeps the cache at its full `window` size *always*, plus a
`valid_mask` (window,) of 1.0/0.0 marking which slots have ever been written
since the last sequence reset, added as a large negative bias to that slot's
attention score before softmax (mathematically equivalent to excluding it -
softmax(-1e9 contribution) ~= 0 - not literally identical to slicing it out,
but the difference is far below float32 precision). The circular write itself
uses `Tensor.index_copy`, which traces to an ONNX Scatter op and works
uniformly for any `write_pos` value, so no Python conditional is needed for
that either. This was verified to reproduce `model.forward()`'s batched
output (and `StreamingPredictor`'s streaming output) to float32 precision -
see `verify()` below.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from model import LSTMAttentionModel, build_lstm_cells


class OnnxStepModel(nn.Module):
    """Pure-tensor-in/pure-tensor-out reformulation of one `predict()` step,
    suitable for `torch.onnx.export` tracing (no Python control flow that
    depends on tensor values).

    Uses `nn.LSTMCell` (via `build_lstm_cells`), NOT the whole-sequence
    `nn.LSTM` called with seq_len=1 - this was tried and measured *worse*:
    at the original config (d_model=128, window=512), the native-LSTM
    version cost 0.616 ms/step vs. this version's 0.239 ms/step, over 2.5x
    worse, despite `nn.LSTM` tracing to a native fused ONNX `LSTM` op (just
    14 total nodes for a standalone 2-layer test) rather than LSTMCell's
    decomposition into ~10+ elementwise ops per layer. The hypothesis
    ("fused op = faster") held for a standalone LSTM but not once integrated
    with the rest of this graph (attention, FF, cache) - the `Slice`/
    `Concat`/`Transpose` glue `nn.LSTM`'s exporter emits to handle the
    2-layer stacked state apparently costs more than the decomposition saves
    here. Documented as a negative result in EXPERIMENTS.md - worth knowing
    before trying this "obvious" optimization again.
    """

    def __init__(self, model: LSTMAttentionModel):
        super().__init__()
        model.eval()
        self.d_model = model.d_model
        self.n_heads = model.attention.n_heads
        self.head_dim = model.attention.head_dim
        self.window = model.attention.window
        if self.window is None:
            raise ValueError("ONNX export requires a finite attention_window (see StreamingPredictor for why)")

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
        features: torch.Tensor,      # (112,)
        h0: torch.Tensor, c0: torch.Tensor,   # (1, d_model) each - layer 0 LSTM state
        h1: torch.Tensor, c1: torch.Tensor,   # (1, d_model) each - layer 1 LSTM state (unused if single-layer)
        cache_k: torch.Tensor,       # (window, n_heads, head_dim)
        cache_v: torch.Tensor,       # (window, n_heads, head_dim)
        valid_mask: torch.Tensor,    # (window,) float32, 1.0 = slot has been written
        write_pos: torch.Tensor,     # (1,) int64 - next slot to overwrite
    ):
        x = self.input_proj(features.view(1, -1))  # (1, d_model)
        h0, c0 = self.lstm_cells[0](x, (h0, c0))
        layer_input = h0
        for cell in self.lstm_cells[1:]:
            h1, c1 = cell(layer_input, (h1, c1))
            layer_input = h1
        lstm_out = layer_input.view(self.d_model)

        normed = self.attn_norm(lstm_out)
        qkv = self.qkv(normed)
        q, k, v = qkv.split(self.d_model)
        q = q.view(self.n_heads, self.head_dim)
        k = k.view(1, self.n_heads, self.head_dim)
        v = v.view(1, self.n_heads, self.head_dim)

        new_cache_k = cache_k.index_copy(0, write_pos, k)
        new_cache_v = cache_v.index_copy(0, write_pos, v)
        new_valid_mask = valid_mask.index_copy(0, write_pos, torch.ones(1, dtype=valid_mask.dtype))

        # Attention scores/output via explicit matmul rather than `einsum`:
        # measured ~4.5x slower per op in exported ONNX graphs for the same
        # work (60us vs 13us avg, profiled - ONNX Runtime evidently doesn't
        # optimize generic Einsum as well as Gemm/MatMul). Mathematically
        # identical to the einsum form this replaced - re-verified below.
        k_heads_first = new_cache_k.transpose(0, 1)            # (n_heads, window, head_dim)
        q_col = q.unsqueeze(-1)                                 # (n_heads, head_dim, 1)
        scores = torch.matmul(k_heads_first, q_col).squeeze(-1).transpose(0, 1)  # (window, n_heads)
        scores = scores / (self.head_dim ** 0.5)
        scores = scores + (1.0 - new_valid_mask).unsqueeze(-1) * -1e9
        attn = torch.softmax(scores, dim=0)                     # (window, n_heads)

        attn_heads_first = attn.transpose(0, 1).unsqueeze(1)    # (n_heads, 1, window)
        v_heads_first = new_cache_v.transpose(0, 1)              # (n_heads, window, head_dim)
        attended = torch.matmul(attn_heads_first, v_heads_first).squeeze(1).reshape(self.d_model)  # (d_model,)
        attended = self.out_proj(attended)

        x2 = lstm_out + attended
        x2 = x2 + self.feed_forward(self.ff_norm(x2))
        prediction = self.head(x2)

        new_write_pos = (write_pos + 1) % self.window

        return prediction, h0, c0, h1, c1, new_cache_k, new_cache_v, new_valid_mask, new_write_pos


INPUT_NAMES = ["features", "h0", "c0", "h1", "c1", "cache_k", "cache_v", "valid_mask", "write_pos"]
OUTPUT_NAMES = ["prediction", "next_h0", "next_c0", "next_h1", "next_c1",
                "next_cache_k", "next_cache_v", "next_valid_mask", "next_write_pos"]
STATE_NAMES = ["h0", "c0", "h1", "c1", "cache_k", "cache_v", "valid_mask", "write_pos"]  # order matches OUTPUT_NAMES[1:]


def initial_state(step_model: OnnxStepModel):
    """Zeroed state tensors for the start of a sequence - what `solution.py`
    should reset to whenever `seq_ix` changes (mirrors the baseline's
    `hidden_0.fill(0.0)` / `hidden_1.fill(0.0)` reset, plus resetting the
    cache/mask/write_pos this model additionally needs)."""
    d, w, hh, hd = step_model.d_model, step_model.window, step_model.n_heads, step_model.head_dim
    return {
        "h0": torch.zeros(1, d), "c0": torch.zeros(1, d),
        "h1": torch.zeros(1, d), "c1": torch.zeros(1, d),
        "cache_k": torch.zeros(w, hh, hd), "cache_v": torch.zeros(w, hh, hd),
        "valid_mask": torch.zeros(w, dtype=torch.float32),
        "write_pos": torch.zeros(1, dtype=torch.int64),
    }


def export(model: LSTMAttentionModel, output_path, opset: int = 18) -> None:
    model.eval()
    step_model = OnnxStepModel(model)
    step_model.eval()

    state = initial_state(step_model)
    example_inputs = (
        torch.zeros(112),
        state["h0"], state["c0"], state["h1"], state["c1"],
        state["cache_k"], state["cache_v"], state["valid_mask"], state["write_pos"],
    )

    torch.onnx.export(
        step_model,
        example_inputs,
        str(output_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset,
        do_constant_folding=True,
    )


def verify(model: LSTMAttentionModel, onnx_path, n_steps: int = 800, atol: float = 1e-4) -> None:
    """Checks the exported ONNX graph agrees with (a) `model.forward()`'s
    batched output and (b) `StreamingPredictor`'s PyTorch streaming output,
    on the same random input sequence. Raises AssertionError on mismatch.
    """
    import numpy as np
    import onnxruntime as ort

    from inference import StreamingPredictor

    torch.manual_seed(0)
    features_seq = torch.randn(n_steps, 112)

    model.eval()
    with torch.no_grad():
        batched_out, _ = model(features_seq.unsqueeze(0))
    batched_out = batched_out.squeeze(0).numpy()

    predictor = StreamingPredictor(model)
    streaming_out = np.stack([predictor.step(features_seq[i]).numpy() for i in range(n_steps)])

    step_model = OnnxStepModel(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state = {k: v.numpy() for k, v in initial_state(step_model).items()}
    onnx_out = []
    for i in range(n_steps):
        feed = {"features": features_seq[i].numpy(), **state}
        outputs = session.run(OUTPUT_NAMES, feed)
        prediction, *next_state_values = outputs
        onnx_out.append(prediction)
        state = dict(zip(STATE_NAMES, next_state_values))
    onnx_out = np.stack(onnx_out)

    diff_vs_batched = np.abs(onnx_out - batched_out).max()
    diff_vs_streaming = np.abs(onnx_out - streaming_out).max()
    print(f"ONNX vs model.forward() (batched):      max abs diff = {diff_vs_batched:.3e}")
    print(f"ONNX vs StreamingPredictor (streaming):  max abs diff = {diff_vs_streaming:.3e}")
    assert diff_vs_batched < atol, f"ONNX output diverges from batched forward: {diff_vs_batched}"
    assert diff_vs_streaming < atol, f"ONNX output diverges from streaming predictor: {diff_vs_streaming}"
    print("verify() PASSED - ONNX graph matches both reference implementations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="dir with config.json + best.pt; if best.pt is missing, exports randomly-initialized weights for architecture/latency testing only")
    parser.add_argument("--output", default="checkpoints/model.onnx")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    import json
    checkpoint_dir = Path(args.checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    model = LSTMAttentionModel(
        d_model=config["d_model"], lstm_layers=config["lstm_layers"],
        n_heads=config["n_heads"], attention_window=config["attention_window"],
        ff_multiplier=config.get("ff_multiplier", 4), dropout=0.0,
    )
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
        print(f"loaded trained weights from {best_path}")
    else:
        print(f"WARNING: {best_path} not found - exporting randomly-initialized weights "
              f"(fine for testing the export/latency, NOT valid for submission)")
    model.eval()

    export(model, args.output)
    print(f"exported to {args.output}")

    if not args.skip_verify:
        verify(model, args.output)
