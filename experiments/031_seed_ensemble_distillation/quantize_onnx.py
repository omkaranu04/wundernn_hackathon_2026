"""int8 dynamic quantisation of the LSTM weights of an exported graph.

    python quantize_onnx.py IN_model.onnx OUT_model.onnx

Only the LSTM ops are quantised (`DynamicQuantizeLSTM`); encoders and head
stay fp32. The result is a single self-contained file (no .data sidecar),
loaded by the unchanged solution.py. Measured 2026-10-12: about -0.0003 WP,
and it makes wide LSTMs cheap (see README).
"""
import sys, tempfile, os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

src, dst = sys.argv[1], sys.argv[2]
with tempfile.TemporaryDirectory() as tmp:
    single = os.path.join(tmp, "fp32.onnx")
    onnx.save(onnx.load(src), single)          # folds an external .data file into one file
    quantize_dynamic(single, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"])
m = onnx.load(dst)
assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer), "external data left behind"
print(f"wrote {dst}: {len(m.graph.node)} nodes, {os.path.getsize(dst) / 1e6:.2f} MB, ops {sorted(set(n.op_type for n in m.graph.node))}")
