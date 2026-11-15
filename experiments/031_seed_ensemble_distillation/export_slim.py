"""Slim ONNX export of a (possibly merged) lean model: the same function in
13 nodes instead of 46.

Per-step cost of our graphs is dominated by per-node overhead, not arithmetic
(node count vs latency r = 0.92 across 36 submissions). The standard export
spends 33 of its 46 nodes on plumbing: slicing the 112 inputs into groups,
concatenating them back, slicing and re-stacking the two-layer state. All of
it can be folded into the weights:

  xc  = Concat(x, Cos(x))                       (224,)   cos of ALL inputs, unused ones get zero weight
  e   = Relu(xc @ W_enc + b_enc)                one MatMul for all group encoders; the 16 raw
                                                `additional` inputs p pass through the Relu as
                                                (relu(p), relu(-p)) and are recombined by the
                                                LSTM input weights (+W, -W): exact
  LSTM layer 1 -> LSTM layer 2                  four separate state tensors h0,c0,h1,c1: no
                                                slicing/stacking; layer 1's Y_h feeds layer 2
  pred = (Relu(h @ W1 + b1)) @ W2 + b2          shape (1,1,2); solution_slim.py flattens it

    python export_slim.py --checkpoint-dir CKPT --weights merged.pt --output slim.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from model import ADDITIONAL, I0_DP, I0_DV, I0_PRICE, I0_VOLUME, I1_DP, I1_DV, I1_PRICE, I1_VOLUME, LeanCosGroupedEncoderLSTMModel

N_IN = 112


def _lstm_onnx(w_ih, w_hh, b_ih, b_hh, H):
    """PyTorch gate order (i, f, g, o) -> ONNX (i, o, f, c)."""
    order = np.concatenate([np.arange(0, H), np.arange(3 * H, 4 * H), np.arange(H, 2 * H), np.arange(2 * H, 3 * H)])
    return w_ih[order][None], w_hh[order][None], np.concatenate([b_ih[order], b_hh[order]])[None]


def _gru_onnx(w_ih, w_hh, b_ih, b_hh, H):
    """PyTorch gate order (r, z, n) -> ONNX (z, r, h); use with linear_before_reset=1 (PyTorch's n-gate form)."""
    order = np.concatenate([np.arange(H, 2 * H), np.arange(0, H), np.arange(2 * H, 3 * H)])
    return w_ih[order][None], w_hh[order][None], np.concatenate([b_ih[order], b_hh[order]])[None]


def build(model: LeanCosGroupedEncoderLSTMModel) -> onnx.ModelProto:
    assert model.num_layers == 2 and not model.bidirectional
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    H = model.hidden_size
    groups = [("price_encoder", np.arange(*I0_PRICE)), ("volume_encoder", np.arange(*I0_VOLUME)),
              ("trade_encoder", np.concatenate([np.arange(*I0_DP), np.arange(*I0_DV), np.arange(*I1_DP), np.arange(*I1_DV)]))]
    add = np.arange(*ADDITIONAL); add_pos = np.concatenate([add, N_IN + add])       # [a, cos(a)] in xc
    n_enc = sum(sd[f"{g}.net.0.weight"].shape[0] for g, _ in groups); n_add = len(add_pos)
    W_enc = np.zeros((n_enc + 2 * n_add, 2 * N_IN), np.float32); b_enc = np.zeros(n_enc + 2 * n_add, np.float32)
    r = 0
    for g, idx in groups:
        w = sd[f"{g}.net.0.weight"]; d = len(idx)
        W_enc[r:r + w.shape[0], idx] = w[:, :d]; W_enc[r:r + w.shape[0], N_IN + idx] = w[:, d:]
        b_enc[r:r + w.shape[0]] = sd[f"{g}.net.0.bias"]; r += w.shape[0]
    W_enc[np.arange(n_enc, n_enc + n_add), add_pos] = 1.0
    W_enc[np.arange(n_enc + n_add, n_enc + 2 * n_add), add_pos] = -1.0
    w_ih0 = sd["lstm.weight_ih_l0"]
    w_ih0 = np.concatenate([w_ih0[:, :n_enc], w_ih0[:, n_enc:], -w_ih0[:, n_enc:]], axis=1)
    W0, R0, B0 = _lstm_onnx(w_ih0, sd["lstm.weight_hh_l0"], sd["lstm.bias_ih_l0"], sd["lstm.bias_hh_l0"], H)
    W1, R1, B1 = _lstm_onnx(sd["lstm.weight_ih_l1"], sd["lstm.weight_hh_l1"], sd["lstm.bias_ih_l1"], sd["lstm.bias_hh_l1"], H)

    init = {"W_enc": W_enc.T.copy(), "b_enc": b_enc, "W0": W0, "R0": R0, "B0": B0, "W1": W1, "R1": R1, "B1": B1,
            "Wh1": sd["head.0.weight"].T.copy(), "bh1": sd["head.0.bias"], "Wh2": sd["head.3.weight"].T.copy(), "bh2": sd["head.3.bias"],
            "shape_x": np.array([1, 1, -1], np.int64)}
    nodes = [
        helper.make_node("Cos", ["features"], ["cosx"]),
        helper.make_node("Concat", ["features", "cosx"], ["xc"], axis=0),
        helper.make_node("MatMul", ["xc", "W_enc"], ["enc_mm"]),
        helper.make_node("Add", ["enc_mm", "b_enc"], ["enc_pre"]),
        helper.make_node("Relu", ["enc_pre"], ["enc"]),
        helper.make_node("Reshape", ["enc", "shape_x"], ["x3"]),
        helper.make_node("LSTM", ["x3", "W0", "R0", "B0", "", "h0", "c0"], ["", "next_h0", "next_c0"], hidden_size=H),
        helper.make_node("LSTM", ["next_h0", "W1", "R1", "B1", "", "h1", "c1"], ["", "next_h1", "next_c1"], hidden_size=H),
        helper.make_node("MatMul", ["next_h1", "Wh1"], ["hd_mm"]),
        helper.make_node("Add", ["hd_mm", "bh1"], ["hd_pre"]),
        helper.make_node("Relu", ["hd_pre"], ["hd"]),
        helper.make_node("MatMul", ["hd", "Wh2"], ["out_mm"]),
        helper.make_node("Add", ["out_mm", "bh2"], ["prediction"]),
    ]
    st = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1, H])
    graph = helper.make_graph(
        nodes, "lean_slim",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [N_IN])] + [st(n) for n in ("h0", "c0", "h1", "c1")],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 1, 2])] + [st(n) for n in ("next_h0", "next_c0", "next_h1", "next_c1")],
        [numpy_helper.from_array(v, k) for k, v in init.items()],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def build_x(model) -> onnx.ModelProto:
    """LeanXModel (model.py, 2026-11-02): build()'s 13-node graph plus, when enabled, instrument-1 group encoders
    (more rows of the one encoder MatMul), a `pos` input (3,) = [s, sin 2 pi s, cos 2 pi s] concatenated to the LSTM
    input (solution_slim.py feeds it from step_in_seq), and the residual MLP h + W2 relu(W1 h) before the head."""
    assert model.num_layers == 2
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    H = model.hidden_size
    groups = [("price_encoder", np.arange(*I0_PRICE)), ("volume_encoder", np.arange(*I0_VOLUME)),
              ("trade_encoder", np.concatenate([np.arange(*I0_DP), np.arange(*I0_DV), np.arange(*I1_DP), np.arange(*I1_DV)]))]
    if model.use_i1:
        groups += [("i1_price_encoder", np.arange(*I1_PRICE)), ("i1_volume_encoder", np.arange(*I1_VOLUME))]
    add = np.arange(*ADDITIONAL); add_pos = np.concatenate([add, N_IN + add])
    n_enc = sum(sd[f"{g}.net.0.weight"].shape[0] for g, _ in groups); n_add = len(add_pos)
    W_enc = np.zeros((n_enc + 2 * n_add, 2 * N_IN), np.float32); b_enc = np.zeros(n_enc + 2 * n_add, np.float32); r = 0
    for g, idx in groups:
        w = sd[f"{g}.net.0.weight"]; d = len(idx)
        W_enc[r:r + w.shape[0], idx] = w[:, :d]; W_enc[r:r + w.shape[0], N_IN + idx] = w[:, d:]
        b_enc[r:r + w.shape[0]] = sd[f"{g}.net.0.bias"]; r += w.shape[0]
    W_enc[np.arange(n_enc, n_enc + n_add), add_pos] = 1.0
    W_enc[np.arange(n_enc + n_add, n_enc + 2 * n_add), add_pos] = -1.0
    w = sd["lstm.weight_ih_l0"]  # columns: [groups (n_enc), additional (16), pos (3 if used)]
    w_ih0 = np.concatenate([w[:, :n_enc], w[:, n_enc:n_enc + n_add], -w[:, n_enc:n_enc + n_add], w[:, n_enc + n_add:]], axis=1)
    W0, R0, B0 = _lstm_onnx(w_ih0, sd["lstm.weight_hh_l0"], sd["lstm.bias_ih_l0"], sd["lstm.bias_hh_l0"], H)
    W1, R1, B1 = _lstm_onnx(sd["lstm.weight_ih_l1"], sd["lstm.weight_hh_l1"], sd["lstm.bias_ih_l1"], sd["lstm.bias_hh_l1"], H)
    init = {"W_enc": W_enc.T.copy(), "b_enc": b_enc, "W0": W0, "R0": R0, "B0": B0, "W1": W1, "R1": R1, "B1": B1,
            "Wh1": sd["head.0.weight"].T.copy(), "bh1": sd["head.0.bias"], "Wh2": sd["head.3.weight"].T.copy(), "bh2": sd["head.3.bias"],
            "shape_x": np.array([1, 1, -1], np.int64)}
    nodes = [helper.make_node("Cos", ["features"], ["cosx"]), helper.make_node("Concat", ["features", "cosx"], ["xc"], axis=0),
             helper.make_node("MatMul", ["xc", "W_enc"], ["enc_mm"]), helper.make_node("Add", ["enc_mm", "b_enc"], ["enc_pre"]),
             helper.make_node("Relu", ["enc_pre"], ["enc"])]
    if model.use_pos:
        nodes.append(helper.make_node("Concat", ["enc", "pos"], ["encp"], axis=0)); x_in = "encp"
    else:
        x_in = "enc"
    nodes += [helper.make_node("Reshape", [x_in, "shape_x"], ["x3"]),
              helper.make_node("LSTM", ["x3", "W0", "R0", "B0", "", "h0", "c0"], ["", "next_h0", "next_c0"], hidden_size=H),
              helper.make_node("LSTM", ["next_h0", "W1", "R1", "B1", "", "h1", "c1"], ["", "next_h1", "next_c1"], hidden_size=H)]
    h_out = "next_h1"
    if model.use_res:
        init.update({"Wr1": sd["res.0.weight"].T.copy(), "br1": sd["res.0.bias"], "Wr2": sd["res.2.weight"].T.copy(), "br2": sd["res.2.bias"]})
        nodes += [helper.make_node("MatMul", ["next_h1", "Wr1"], ["r_mm"]), helper.make_node("Add", ["r_mm", "br1"], ["r_pre"]),
                  helper.make_node("Relu", ["r_pre"], ["r_act"]), helper.make_node("MatMul", ["r_act", "Wr2"], ["r2_mm"]),
                  helper.make_node("Add", ["r2_mm", "br2"], ["r2"]), helper.make_node("Add", ["next_h1", "r2"], ["h_res"])]
        h_out = "h_res"
    nodes += [helper.make_node("MatMul", [h_out, "Wh1"], ["hd_mm"]), helper.make_node("Add", ["hd_mm", "bh1"], ["hd_pre"]),
              helper.make_node("Relu", ["hd_pre"], ["hd"]), helper.make_node("MatMul", ["hd", "Wh2"], ["out_mm"]),
              helper.make_node("Add", ["out_mm", "bh2"], ["prediction"])]
    st = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1, H])
    ins = [helper.make_tensor_value_info("features", TensorProto.FLOAT, [N_IN])] + [st(n) for n in ("h0", "c0", "h1", "c1")]
    if model.use_pos:
        ins.append(helper.make_tensor_value_info("pos", TensorProto.FLOAT, [3]))
    graph = helper.make_graph(nodes, "leanx_slim", ins,
                              [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 1, 2])] + [st(n) for n in ("next_h0", "next_c0", "next_h1", "next_c1")],
                              [numpy_helper.from_array(v, k) for k, v in init.items()])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def build_dist(model) -> onnx.ModelProto:
    """LeanDistModel (032): build()'s graph, whose head now emits 2 x n_bins logits, plus
    Reshape -> Softmax over bins -> point = sum p|c|c / (sum p|c| + 1e-3), the model's |y|-weighted mean."""
    m = build(model)
    g = m.graph; nb = model.n_bins; c = model.centers.detach().cpu().numpy().astype(np.float32)
    last = [n for n in g.node if n.output and n.output[0] == "prediction"][0]
    last.output[0] = "logits_flat"
    g.initializer.extend([numpy_helper.from_array(np.array([1, 1, 2, nb], np.int64), "shape_logits"),
                          numpy_helper.from_array((np.abs(c) * c).reshape(nb, 1), "C_num"),
                          numpy_helper.from_array(np.abs(c).reshape(nb, 1), "C_den"),
                          numpy_helper.from_array(np.array(1e-3, np.float32), "eps_den"),
                          numpy_helper.from_array(np.array([1, 1, 2], np.int64), "shape_pred")])
    g.node.extend([helper.make_node("Reshape", ["logits_flat", "shape_logits"], ["logits4"]),
                   helper.make_node("Softmax", ["logits4"], ["probs"], axis=-1),
                   helper.make_node("MatMul", ["probs", "C_num"], ["num"]), helper.make_node("MatMul", ["probs", "C_den"], ["den0"]),
                   helper.make_node("Add", ["den0", "eps_den"], ["den"]), helper.make_node("Div", ["num", "den"], ["pt4"]),
                   helper.make_node("Reshape", ["pt4", "shape_pred"], ["prediction"])])
    onnx.checker.check_model(m)
    return m


def build_fusion(model) -> onnx.ModelProto:
    """FusionGRUModel (032), folded for speed. State Z = [x_prev, ema_f, ema_s] (336,) and started (1,).
    Z_eff = s*Z + (1-s)*[x,x,x] (a sequence start makes x_prev = ema = x, as in training); the five feature channels
    [x, x - x_prev, x - ema_f', x - ema_s', cos x] with ema' = a*ema_eff + (1-a)*x are LINEAR in [x, Z_eff, cos x], so
    the embedding layer becomes ONE MatMul on Concat(x, Z_eff, cos x) with folded weights; Z' = D*Z_eff + (1-D)*[x,x,x]
    with D = [0, a_f, a_s] per block (the x_prev slot takes x)."""
    from model import FUSION_RATES
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    H, E, N = model.hidden_size, model.embed_dim, N_IN; af, as_ = FUSION_RATES
    We = sd["embed.weight"]                     # (E, 560): blocks [x, dx, x-ef, x-es, cos]
    Wx, Wd, Wf, Ws, Wc = (We[:, i * N:(i + 1) * N] for i in range(5))
    use_sin = getattr(model, "use_sin", False)      # 033: 6th channel sin x
    Wsn = We[:, 5 * N:6 * N] if use_sin else None
    # x - ef' = x - af*ef - (1-af)*x = af*x - af*ef  (same for es); dx = x - xp
    A_x = Wx + Wd + af * Wf + as_ * Ws
    W_fold = np.concatenate([A_x, -Wd, -af * Wf, -as_ * Ws, Wc] + ([Wsn] if use_sin else []), axis=1)   # on [x, xp, ef, es, cos x(, sin x)]
    Wg, Rg, Bg = _gru_onnx(*(sd[k].astype(np.float32) for k in ("lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0")), H)
    D = np.concatenate([np.zeros(N), np.full(N, af), np.full(N, as_)]).astype(np.float32)
    f32 = lambda v: np.asarray(v, np.float32)
    init = {"Wf": f32(W_fold.T), "be": f32(sd["embed.bias"]), "W": Wg, "R": Rg, "B": Bg, "D": D, "Dc": f32(1 - D), "one": f32([1.0]),
            "Wh1": f32(sd["head.0.weight"].T), "bh1": f32(sd["head.0.bias"]), "Wh2": f32(sd["head.3.weight"].T), "bh2": f32(sd["head.3.bias"]),
            "shape_e": np.array([1, 1, E], np.int64)}
    n = helper.make_node
    nodes = [n("Concat", ["features", "features", "features"], ["x3"], axis=0), n("Sub", ["one", "started"], ["ns"]),
             n("Mul", ["started", "Z"], ["sZ"]), n("Mul", ["ns", "x3"], ["nx3"]), n("Add", ["sZ", "nx3"], ["Ze"]),
             n("Cos", ["features"], ["cosx"])] + ([n("Sin", ["features"], ["sinx"])] if use_sin else []) + [
             n("Concat", ["features", "Ze", "cosx"] + (["sinx"] if use_sin else []), ["v"], axis=0),
             n("MatMul", ["v", "Wf"], ["e_mm"]), n("Add", ["e_mm", "be"], ["e_pre"]), n("Relu", ["e_pre"], ["emb"]),
             n("Reshape", ["emb", "shape_e"], ["emb3"]),
             n("GRU", ["emb3", "W", "R", "B", "", "h"], ["", "next_h"], hidden_size=H, linear_before_reset=1),
             n("Concat", ["next_h", "emb3"], ["hc"], axis=-1),
             n("MatMul", ["hc", "Wh1"], ["hd_mm"]), n("Add", ["hd_mm", "bh1"], ["hd_pre"]), n("Relu", ["hd_pre"], ["hd"]),
             n("MatMul", ["hd", "Wh2"], ["out_mm"]), n("Add", ["out_mm", "bh2"], ["prediction"]),
             n("Mul", ["D", "Ze"], ["dZ"]), n("Mul", ["Dc", "x3"], ["dx3"]), n("Add", ["dZ", "dx3"], ["next_Z"]),
             n("Identity", ["one"], ["next_started"])]
    v = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    ins = [v("features", [N]), v("h", [1, 1, H]), v("Z", [3 * N]), v("started", [1])]
    outs = [v("prediction", [1, 1, 2]), v("next_h", [1, 1, H]), v("next_Z", [3 * N]), v("next_started", [1])]
    graph = helper.make_graph(nodes, "fusion_gru", ins, outs, [numpy_helper.from_array(val, k) for k, val in init.items()])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def verify_stream(ref, path, n_steps=1200, atol=1e-4):
    """Run any streaming graph (states = inputs other than features/pos, each with next_<name>) step by step on random
    input from zero state and compare with `ref` (n_steps, 2), the reference prediction for the same input."""
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins = {i.name: [d if isinstance(d, int) else 1 for d in i.shape] for i in s.get_inputs()}
    fshape = ins.pop("features"); has_pos = ins.pop("pos", None) is not None
    st = {k: np.zeros(v, np.float32) for k, v in ins.items()}; names = list(st)
    torch.manual_seed(1); x = torch.randn(1, n_steps, N_IN)
    out = np.zeros((n_steps, 2), np.float32)
    for t in range(n_steps):
        feed = {"features": x[0, t].numpy().reshape(fshape), **st}
        if has_pos:
            feed["pos"] = pos_features(t)
        r = s.run(["prediction"] + ["next_" + k for k in names], feed)
        out[t] = r[0].reshape(-1); st = dict(zip(names, r[1:]))
    err = float(np.abs(out - ref(x)).max())
    print(f"streaming ONNX vs reference: max abs diff = {err:.3e} over {n_steps} steps")
    if err > atol:
        raise SystemExit("stream verify FAILED")


def combine(paths, out_path):
    """Members' graphs (each with input `features` (112,) and output `prediction` (1,1,2)) -> ONE graph whose
    prediction is their mean. Member k's names get the prefix m<k>_; its state input X / output next_X become
    m<k>_X / next_m<k>_X, so solution_slim.py's state convention still holds."""
    graphs = [onnx.load(p) for p in paths]; K = len(graphs)
    nodes, inits, ins, outs = [], [], [helper.make_tensor_value_info("features", TensorProto.FLOAT, [N_IN])], []
    for k, m in enumerate(graphs):
        g = m.graph; p = f"m{k}_"
        state_in = [i.name for i in g.input if i.name != "features"]
        ren = {name: p + name for name in state_in}
        ren.update({"next_" + name: "next_" + p + name for name in state_in})
        ren["prediction"] = p + "prediction"
        f_in = [i for i in g.input if i.name == "features"][0]
        f3 = len(f_in.type.tensor_type.shape.dim) == 3
        if f3:   # build_multi graphs take features (1,1,112): reshape the shared (112,) input for this member
            inits.append(numpy_helper.from_array(np.array([1, 1, N_IN], np.int64), p + "shape_f3"))
            nodes.append(helper.make_node("Reshape", ["features", p + "shape_f3"], [p + "features3"]))
        rn = lambda x: "" if x == "" else ((p + "features3" if f3 else "features") if x == "features" else ren.get(x, p + x))
        for j, node in enumerate(g.node):
            nn_ = onnx.NodeProto(); nn_.CopyFrom(node)
            nn_.input[:] = [rn(x) for x in node.input]; nn_.output[:] = [rn(x) for x in node.output]; nn_.name = f"{p}{node.name or 'n'}_{j}"
            nodes.append(nn_)
        for t in g.initializer:
            tt = onnx.TensorProto(); tt.CopyFrom(t); tt.name = p + t.name; inits.append(tt)
        for i in g.input:
            if i.name != "features":
                vi = onnx.ValueInfoProto(); vi.CopyFrom(i); vi.name = p + i.name; ins.append(vi)
        for o in g.output:
            if o.name != "prediction":
                vo = onnx.ValueInfoProto(); vo.CopyFrom(o); vo.name = rn(o.name); outs.append(vo)
    inits.append(numpy_helper.from_array(np.array(1.0 / K, np.float32), "inv_k"))
    acc = "m0_prediction"
    for k in range(1, K):
        nodes.append(helper.make_node("Add", [acc, f"m{k}_prediction"], [f"sum_{k}"])); acc = f"sum_{k}"
    nodes.append(helper.make_node("Mul", [acc, "inv_k"], ["prediction"]))
    graph = helper.make_graph(nodes, "combined", ins, [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 1, 2])] + outs, inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    onnx.checker.check_model(m); onnx.save(m, out_path)
    print(f"combined {K} graphs -> {out_path}: {len(nodes)} nodes")


def pos_features(step: int) -> np.ndarray:
    s = step / 20000.0
    return np.array([s, np.sin(2 * np.pi * s), np.cos(2 * np.pi * s)], np.float32)


def verify_x(model, path, n_steps=600, atol=1e-4):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    H = model.hidden_size; has_pos = any(i.name == "pos" for i in s.get_inputs())
    torch.manual_seed(1); x = torch.randn(1, n_steps, N_IN)
    with torch.no_grad():
        ref = model(x)[0][0].numpy()
    st = {k: np.zeros((1, 1, H), np.float32) for k in ("h0", "c0", "h1", "c1")}
    out = np.zeros_like(ref)
    for t in range(n_steps):
        feed = {"features": x[0, t].numpy(), **st}
        if has_pos:
            feed["pos"] = pos_features(t)
        p, *nxt = s.run(["prediction", "next_h0", "next_c0", "next_h1", "next_c1"], feed)
        out[t] = p.reshape(-1); st = dict(zip(("h0", "c0", "h1", "c1"), nxt))
    err = float(np.abs(out - ref).max())
    print(f"leanx slim ONNX vs model.forward(): max abs diff = {err:.3e} over {n_steps} steps")
    if err > atol:
        raise SystemExit("leanx verify FAILED")


def build_multi(members) -> onnx.ModelProto:
    """K members as ONE graph with K SEPARATE LSTM chains (12 + 2K nodes), output = mean.

    A block-diagonal merged LSTM (merge_models.py) multiplies K^2 blocks of which
    K are non-zero; separate LSTM ops do only the K useful ones, and an extra node
    costs well under a microsecond. Members may have different widths.
    Inputs here are 3-D: features (1,1,112); states h0_k, c0_k, h1_k, c1_k (1,1,H_k).
    """
    K = len(members)
    trade_idx = np.concatenate([np.arange(*I0_DP), np.arange(*I0_DV), np.arange(*I1_DP), np.arange(*I1_DV)])
    groups = [("price_encoder", np.arange(*I0_PRICE)), ("volume_encoder", np.arange(*I0_VOLUME)), ("trade_encoder", trade_idx)]
    # 033: FullFeatureModel members use [x, cos x, sin x] of ALL inputs; the shared expansion then has a sin block too
    is_full = lambda m: type(m).__name__ == "FullFeatureModel"
    NT = 3 if any(is_full(m) for m in members) else 2
    enc_rows, enc_bias, sizes, init, nodes, hs, ins, outs, h_outs, uses_pos = [], [], [], {}, [], [], [], [], [], False
    head_w1, head_b1, head_w2, head_b2 = [], [], [], 0.0
    for k, model in enumerate(members):
        assert model.num_layers == 2 and not model.bidirectional
        sd = {n: v.detach().cpu().numpy().astype(np.float32) for n, v in model.state_dict().items()}
        H = model.hidden_size
        i1g = [("i1_price_encoder", np.arange(*I1_PRICE)), ("i1_volume_encoder", np.arange(*I1_VOLUME))]
        if is_full(model):   # FullFeatureModel order: price, volume, i1 price, i1 volume, trade, additional
            mgroups = groups[:2] + i1g + groups[2:]; mt = 3
        else:
            mgroups = groups + (i1g if getattr(model, "use_i1", False) else []); mt = 2
        add = np.arange(*ADDITIONAL); add_pos = np.concatenate([add + j * N_IN for j in range(mt)]); n_add = len(add_pos)
        n_enc = sum(sd[f"{g}.net.0.weight"].shape[0] for g, _ in mgroups)
        W = np.zeros((n_enc + 2 * n_add, NT * N_IN), np.float32); b = np.zeros(n_enc + 2 * n_add, np.float32); r = 0
        for g, idx in mgroups:
            w = sd[f"{g}.net.0.weight"]; d = len(idx)
            for j in range(mt):   # transform blocks [x | cos x | sin x] of the shared expansion
                W[r:r + w.shape[0], j * N_IN + idx] = w[:, j * d:(j + 1) * d]
            b[r:r + w.shape[0]] = sd[f"{g}.net.0.bias"]; r += w.shape[0]
        W[np.arange(n_enc, n_enc + n_add), add_pos] = 1.0; W[np.arange(n_enc + n_add, n_enc + 2 * n_add), add_pos] = -1.0
        enc_rows.append(W); enc_bias.append(b); sizes.append(n_enc + 2 * n_add)
        w0 = sd["lstm.weight_ih_l0"]  # columns [groups, additional (16), pos (3, LeanX only)]
        w0 = np.concatenate([w0[:, :n_enc], w0[:, n_enc:n_enc + n_add], -w0[:, n_enc:n_enc + n_add], w0[:, n_enc + n_add:]], axis=1)
        e_in = f"e_{k}"
        if getattr(model, "use_pos", False):  # position features appended to this member's LSTM input
            nodes.append(helper.make_node("Concat", [f"e_{k}", "pos3"], [f"ep_{k}"], axis=-1)); e_in = f"ep_{k}"; uses_pos = True
        is_gru = type(model.lstm).__name__ == "GRU"   # LeanGRUModel (2026-11-02): GRU core, one state per layer
        conv = _gru_onnx if is_gru else _lstm_onnx
        init[f"W0_{k}"], init[f"R0_{k}"], init[f"B0_{k}"] = conv(w0, sd["lstm.weight_hh_l0"], sd["lstm.bias_ih_l0"], sd["lstm.bias_hh_l0"], H)
        init[f"W1_{k}"], init[f"R1_{k}"], init[f"B1_{k}"] = conv(sd["lstm.weight_ih_l1"], sd["lstm.weight_hh_l1"], sd["lstm.bias_ih_l1"], sd["lstm.bias_hh_l1"], H)
        if is_gru:
            nodes.append(helper.make_node("GRU", [e_in, f"W0_{k}", f"R0_{k}", f"B0_{k}", "", f"h0_{k}"], ["", f"next_h0_{k}"], hidden_size=H, linear_before_reset=1))
            nodes.append(helper.make_node("GRU", [f"next_h0_{k}", f"W1_{k}", f"R1_{k}", f"B1_{k}", "", f"h1_{k}"], ["", f"next_h1_{k}"], hidden_size=H, linear_before_reset=1))
        else:
            nodes.append(helper.make_node("LSTM", [e_in, f"W0_{k}", f"R0_{k}", f"B0_{k}", "", f"h0_{k}", f"c0_{k}"], ["", f"next_h0_{k}", f"next_c0_{k}"], hidden_size=H))
            nodes.append(helper.make_node("LSTM", [f"next_h0_{k}", f"W1_{k}", f"R1_{k}", f"B1_{k}", "", f"h1_{k}", f"c1_{k}"], ["", f"next_h1_{k}", f"next_c1_{k}"], hidden_size=H))
        h_last = f"next_h1_{k}"
        if getattr(model, "use_res", False):  # LeanX residual MLP: h + W2 relu(W1 h)
            init.update({f"Wr1_{k}": sd["res.0.weight"].T.copy(), f"br1_{k}": sd["res.0.bias"], f"Wr2_{k}": sd["res.2.weight"].T.copy(), f"br2_{k}": sd["res.2.bias"]})
            nodes += [helper.make_node("MatMul", [h_last, f"Wr1_{k}"], [f"rm1_{k}"]), helper.make_node("Add", [f"rm1_{k}", f"br1_{k}"], [f"rp_{k}"]),
                      helper.make_node("Relu", [f"rp_{k}"], [f"ra_{k}"]), helper.make_node("MatMul", [f"ra_{k}", f"Wr2_{k}"], [f"rm2_{k}"]),
                      helper.make_node("Add", [f"rm2_{k}", f"br2_{k}"], [f"r2_{k}"]), helper.make_node("Add", [h_last, f"r2_{k}"], [f"hr_{k}"])]
            h_last = f"hr_{k}"
        h_outs.append(h_last)
        hs.append((H, sd["head.0.weight"].shape[0]))
        head_w1.append(sd["head.0.weight"]); head_b1.append(sd["head.0.bias"]); head_w2.append(sd["head.3.weight"] / K); head_b2 = head_b2 + sd["head.3.bias"] / K
        for n in (("h0", "h1") if is_gru else ("h0", "c0", "h1", "c1")):
            ins.append(helper.make_tensor_value_info(f"{n}_{k}", TensorProto.FLOAT, [1, 1, H]))
            outs.append(helper.make_tensor_value_info(f"next_{n}_{k}", TensorProto.FLOAT, [1, 1, H]))
    Wh1 = np.zeros((sum(h for _, h in hs), sum(H for H, _ in hs)), np.float32); r = c = 0
    for (H, hh), w in zip(hs, head_w1):
        Wh1[r:r + hh, c:c + H] = w; r += hh; c += H
    init.update({"W_enc": np.concatenate(enc_rows).T.copy(), "b_enc": np.concatenate(enc_bias), "split": np.array(sizes, np.int64),
                 "Wh1": Wh1.T.copy(), "bh1": np.concatenate(head_b1), "Wh2": np.concatenate(head_w2, axis=1).T.copy(), "bh2": head_b2.astype(np.float32)})
    pre = [helper.make_node("Cos", ["features"], ["cosx"])] + ([helper.make_node("Sin", ["features"], ["sinx"])] if NT == 3 else []) + [
           helper.make_node("Concat", ["features", "cosx"] + (["sinx"] if NT == 3 else []), ["xc"], axis=-1),
           helper.make_node("MatMul", ["xc", "W_enc"], ["enc_mm"]), helper.make_node("Add", ["enc_mm", "b_enc"], ["enc_pre"]),
           helper.make_node("Relu", ["enc_pre"], ["enc"]), helper.make_node("Split", ["enc", "split"], [f"e_{k}" for k in range(K)], axis=-1)]
    if uses_pos:  # one shared `pos` input (3,) (solution_slim.py feeds it), reshaped to (1,1,3) for the 3-D chains
        init["shape_pos"] = np.array([1, 1, 3], np.int64)
        pre.append(helper.make_node("Reshape", ["pos", "shape_pos"], ["pos3"]))
        ins.append(helper.make_tensor_value_info("pos", TensorProto.FLOAT, [3]))
    post = [helper.make_node("Concat", h_outs, ["hcat"], axis=-1) if K > 1 else helper.make_node("Identity", [h_outs[0]], ["hcat"]),
            helper.make_node("MatMul", ["hcat", "Wh1"], ["hd_mm"]), helper.make_node("Add", ["hd_mm", "bh1"], ["hd_pre"]), helper.make_node("Relu", ["hd_pre"], ["hd"]),
            helper.make_node("MatMul", ["hd", "Wh2"], ["out_mm"]), helper.make_node("Add", ["out_mm", "bh2"], ["prediction"])]
    graph = helper.make_graph(pre + nodes + post, "lean_multi",
                              [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1, N_IN])] + ins,
                              [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 1, 2])] + outs,
                              [numpy_helper.from_array(v, n) for n, v in init.items()])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def build_batched(members) -> onnx.ModelProto:
    """K equal-width members as ONE graph whose node count does not depend on K.

    Measured 2026-10-13: an ONNX `LSTM` node costs ~20 us per call before any
    arithmetic, so (a) K separate LSTM chains are slow and (b) a block-diagonal
    merged LSTM wastes K^2 matmul blocks. Here the LSTM cell is written with
    primitive ops and the K members are a BATCH dimension: MatMul broadcasts
    (K,1,d) @ (K,d,4H), i.e. K small matmuls in one node, and every gate op runs
    once on (K,1,H). Gates are stored as [i, f, o | g] so one Sigmoid and one
    Tanh cover them. Output = mean over the K members.
    Inputs: features (112,); states h0, c0, h1, c1 of shape (K,1,H).
    """
    K = len(members); H = members[0].hidden_size
    groups = [("price_encoder", np.arange(*I0_PRICE)), ("volume_encoder", np.arange(*I0_VOLUME)),
              ("trade_encoder", np.concatenate([np.arange(*I0_DP), np.arange(*I0_DV), np.arange(*I1_DP), np.arange(*I1_DV)]))]
    add = np.arange(*ADDITIONAL); add_pos = np.concatenate([add, N_IN + add]); n_add = len(add_pos)
    enc_rows, enc_bias, WL, BL, H1, HB1, H2, HB2 = [], [], [[], []], [[], []], [], [], [], []
    perm = np.concatenate([np.arange(0, 2 * H), np.arange(3 * H, 4 * H), np.arange(2 * H, 3 * H)])  # torch (i,f,g,o) -> (i,f,o,g)
    for model in members:
        assert model.hidden_size == H and model.num_layers == 2 and not model.bidirectional
        sd = {n: v.detach().cpu().numpy().astype(np.float32) for n, v in model.state_dict().items()}
        n_enc = sum(sd[f"{g}.net.0.weight"].shape[0] for g, _ in groups); m = n_enc + 2 * n_add
        W = np.zeros((m, 2 * N_IN), np.float32); b = np.zeros(m, np.float32); r = 0
        for g, idx in groups:
            w = sd[f"{g}.net.0.weight"]; d = len(idx)
            W[r:r + w.shape[0], idx] = w[:, :d]; W[r:r + w.shape[0], N_IN + idx] = w[:, d:]; b[r:r + w.shape[0]] = sd[f"{g}.net.0.bias"]; r += w.shape[0]
        W[np.arange(n_enc, n_enc + n_add), add_pos] = 1.0; W[np.arange(n_enc + n_add, m), add_pos] = -1.0
        enc_rows.append(W); enc_bias.append(b)
        w0 = sd["lstm.weight_ih_l0"]; w0 = np.concatenate([w0[:, :n_enc], w0[:, n_enc:], -w0[:, n_enc:]], axis=1)
        WL[0].append(np.concatenate([w0, sd["lstm.weight_hh_l0"]], axis=1)[perm].T)            # (m+H, 4H)
        BL[0].append((sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"])[perm][None])               # (1, 4H)
        WL[1].append(np.concatenate([sd["lstm.weight_ih_l1"], sd["lstm.weight_hh_l1"]], axis=1)[perm].T)
        BL[1].append((sd["lstm.bias_ih_l1"] + sd["lstm.bias_hh_l1"])[perm][None])
        H1.append(sd["head.0.weight"].T); HB1.append(sd["head.0.bias"][None]); H2.append(sd["head.3.weight"].T); HB2.append(sd["head.3.bias"][None])
    init = {"W_enc": np.concatenate(enc_rows).T.copy(), "b_enc": np.concatenate(enc_bias), "shape_e": np.array([K, 1, -1], np.int64),
            "WL0": np.stack(WL[0]), "BL0": np.stack(BL[0]), "WL1": np.stack(WL[1]), "BL1": np.stack(BL[1]),
            "Wh1": np.stack(H1), "bh1": np.stack(HB1), "Wh2": np.stack(H2), "bh2": np.stack(HB2),
            "split_sg": np.array([3 * H, H], np.int64), "split_ifo": np.array([H, H, H], np.int64), "axis0": np.array([0], np.int64)}
    nodes = [helper.make_node("Cos", ["features"], ["cosx"]), helper.make_node("Concat", ["features", "cosx"], ["xc"], axis=0),
             helper.make_node("MatMul", ["xc", "W_enc"], ["enc_mm"]), helper.make_node("Add", ["enc_mm", "b_enc"], ["enc_pre"]),
             helper.make_node("Relu", ["enc_pre"], ["enc"]), helper.make_node("Reshape", ["enc", "shape_e"], ["x0"])]
    x = "x0"
    for l in (0, 1):
        t = f"l{l}_"
        nodes += [helper.make_node("Concat", [x, f"h{l}"], [t + "xh"], axis=-1),
                  helper.make_node("MatMul", [t + "xh", f"WL{l}"], [t + "mm"]), helper.make_node("Add", [t + "mm", f"BL{l}"], [t + "z"]),
                  helper.make_node("Split", [t + "z", "split_sg"], [t + "s", t + "g"], axis=-1),
                  helper.make_node("Sigmoid", [t + "s"], [t + "S"]), helper.make_node("Tanh", [t + "g"], [t + "G"]),
                  helper.make_node("Split", [t + "S", "split_ifo"], [t + "i", t + "f", t + "o"], axis=-1),
                  helper.make_node("Mul", [t + "f", f"c{l}"], [t + "fc"]), helper.make_node("Mul", [t + "i", t + "G"], [t + "ig"]),
                  helper.make_node("Add", [t + "fc", t + "ig"], [f"next_c{l}"]),
                  helper.make_node("Tanh", [f"next_c{l}"], [t + "tc"]), helper.make_node("Mul", [t + "o", t + "tc"], [f"next_h{l}"])]
        x = f"next_h{l}"
    nodes += [helper.make_node("MatMul", ["next_h1", "Wh1"], ["hd_mm"]), helper.make_node("Add", ["hd_mm", "bh1"], ["hd_pre"]), helper.make_node("Relu", ["hd_pre"], ["hd"]),
              helper.make_node("MatMul", ["hd", "Wh2"], ["out_mm"]), helper.make_node("Add", ["out_mm", "bh2"], ["out_k"]),
              helper.make_node("ReduceMean", ["out_k", "axis0"], ["prediction"], keepdims=0)]
    st = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [K, 1, H])
    graph = helper.make_graph(nodes, "lean_batched",
                              [helper.make_tensor_value_info("features", TensorProto.FLOAT, [N_IN])] + [st(n) for n in ("h0", "c0", "h1", "c1")],
                              [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 2])] + [st(n) for n in ("next_h0", "next_c0", "next_h1", "next_c1")],
                              [numpy_helper.from_array(np.ascontiguousarray(v), n) for n, v in init.items()])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def verify_multi(members, path, n_steps=400, atol=1e-4):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    torch.manual_seed(1); x = torch.randn(1, n_steps, N_IN)
    with torch.no_grad():
        ref = (sum(m(x)[0] for m in members) / len(members))[0].numpy()
    names = [i.name for i in s.get_inputs() if i.name not in ("features", "pos")]
    has_pos = any(i.name == "pos" for i in s.get_inputs())
    fshape = [i.shape for i in s.get_inputs() if i.name == "features"][0]
    st = {i.name: np.zeros(i.shape, np.float32) for i in s.get_inputs() if i.name not in ("features", "pos")}
    out = np.zeros_like(ref)
    for t in range(n_steps):
        feed = {"features": x[0, t].numpy().reshape(fshape), **st}
        if has_pos:
            feed["pos"] = pos_features(t)
        res = s.run(["prediction"] + ["next_" + n for n in names], feed)
        out[t] = res[0].reshape(-1); st = dict(zip(names, res[1:]))
    err = float(np.abs(out - ref).max())
    print(f"multi ONNX ({len(members)} members) vs mean of members: max abs diff = {err:.3e} over {n_steps} steps")
    if err > atol:
        raise SystemExit("multi verify FAILED")


def verify(model, path, n_steps=600, atol=1e-4):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    H = model.hidden_size
    torch.manual_seed(1); x = torch.randn(1, n_steps, N_IN)
    with torch.no_grad():
        ref = model(x)[0][0].numpy()
    st = {k: np.zeros((1, 1, H), np.float32) for k in ("h0", "c0", "h1", "c1")}
    out = np.zeros_like(ref)
    for t in range(n_steps):
        p, *nxt = s.run(["prediction", "next_h0", "next_c0", "next_h1", "next_c1"], {"features": x[0, t].numpy(), **st})
        out[t] = p.reshape(-1); st = dict(zip(("h0", "c0", "h1", "c1"), nxt))
    err = float(np.abs(out - ref).max())
    print(f"slim ONNX vs model.forward(): max abs diff = {err:.3e} over {n_steps} steps")
    if err > atol:
        raise SystemExit("slim verify FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir"); ap.add_argument("--weights", default="best_ema.pt"); ap.add_argument("--output", required=True)
    ap.add_argument("--combine", nargs="+", help="member ONNX files -> one graph (--output) whose prediction is their mean")
    ap.add_argument("--multi", nargs="+", help="CKPT_DIR:WEIGHTS ... -> one graph with separate LSTM / GRU chains, output = mean")
    a = ap.parse_args()
    if a.combine:
        combine(a.combine, a.output)
        import onnxruntime as ort  # check: the combined graph equals the mean of the member graphs, step by step
        sess = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in a.combine]
        def ref(x):
            outs = []
            for s_ in sess:
                ins = {i.name: [d if isinstance(d, int) else 1 for d in i.shape] for i in s_.get_inputs()}; fs = ins.pop("features")
                st = {k: np.zeros(v, np.float32) for k, v in ins.items()}; nm = list(st); o = np.zeros((x.shape[1], 2), np.float32)
                for t in range(x.shape[1]):
                    r = s_.run(["prediction"] + ["next_" + k for k in nm], {"features": x[0, t].numpy().reshape(fs), **st}); o[t] = r[0].reshape(-1); st = dict(zip(nm, r[1:]))
                outs.append(o)
            return sum(outs) / len(outs)
        verify_stream(ref, a.output, n_steps=600)
        return
    if a.multi:
        from teacher_preds import load_model
        members = [load_model(spec, torch.device("cpu")) for spec in a.multi]
        m = build_multi(members); Path(a.output).parent.mkdir(parents=True, exist_ok=True); onnx.save(m, a.output)
        print(f"wrote {a.output}: {len(m.graph.node)} nodes, {len(members)} members")
        verify_multi(members, a.output)
        return
    cfg = json.load(open(Path(a.checkpoint_dir) / "config.json"))
    if cfg.get("arch") in ("fusion_gru", "fusion_gru_sin"):
        from teacher_preds import load_model
        model = load_model(f"{a.checkpoint_dir}:{a.weights}", torch.device("cpu"))
        m = build_fusion(model); Path(a.output).parent.mkdir(parents=True, exist_ok=True); onnx.save(m, a.output)
        print(f"wrote {a.output}: {len(m.graph.node)} nodes (fusion_gru)")
        with torch.no_grad():
            verify_stream(lambda x: model(x)[0][0].numpy(), a.output)
        return
    if cfg.get("arch") == "lean_dist":
        from teacher_preds import load_model
        model = load_model(f"{a.checkpoint_dir}:{a.weights}", torch.device("cpu"))
        m = build_dist(model); Path(a.output).parent.mkdir(parents=True, exist_ok=True); onnx.save(m, a.output)
        print(f"wrote {a.output}: {len(m.graph.node)} nodes (lean_dist, {model.n_bins} bins)")
        verify(model, a.output)
        return
    if cfg.get("arch", "lean").startswith("leanx"):
        from teacher_preds import load_model
        model = load_model(f"{a.checkpoint_dir}:{a.weights}", torch.device("cpu"))
        m = build_x(model); Path(a.output).parent.mkdir(parents=True, exist_ok=True); onnx.save(m, a.output)
        print(f"wrote {a.output}: {len(m.graph.node)} nodes ({cfg['arch']})")
        verify_x(model, a.output)
        return
    model = LeanCosGroupedEncoderLSTMModel(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], price_dim=cfg["price_dim"], volume_dim=cfg["volume_dim"],
                                           trade_dim=cfg["trade_dim"], head_hidden=cfg["head_hidden"], dropout=0.0, aux_scored=cfg.get("aux_scored", False)).eval()
    model.load_state_dict(torch.load(Path(a.checkpoint_dir) / a.weights, map_location="cpu"))
    m = build(model); Path(a.output).parent.mkdir(parents=True, exist_ok=True); onnx.save(m, a.output)
    print(f"wrote {a.output}: {len(m.graph.node)} nodes")
    verify(model, a.output)


if __name__ == "__main__":
    main()
