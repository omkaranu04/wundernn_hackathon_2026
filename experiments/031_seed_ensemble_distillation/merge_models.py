"""Merge K trained lean models into ONE lean model that outputs their mean.

An ensemble of K models normally costs K graphs. But K independent LSTMs of
hidden size H are exactly one LSTM of hidden size K*H whose weight matrices
are block-diagonal, and the same holds for the group encoders and the head.
So the merged model is an ordinary `LeanCosGroupedEncoderLSTMModel` with K
times the widths: the SAME 46-node ONNX graph, exported by the unchanged
`export_onnx.py`, run by the unchanged `solution.py`. With int8 LSTM weights
(`quantize_onnx.py`) the wider matmuls are cheap.

Writes `ema.pt`-style weights (`merged.pt`) and a `config.json` into --out,
after checking that the merged model's output equals the mean of the
members' outputs on random input.

    python merge_models.py --out results/merged_E01 --models CKPT_A:ema.pt CKPT_B:ema.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LeanCosGroupedEncoderLSTMModel
from teacher_preds import load_model

ADD = 16  # expanded `additional` inputs, shared by all members (not duplicated)


def merge(members):
    K = len(members)
    m0 = members[0]
    H, L = m0.hidden_size, m0.num_layers
    P = m0.price_encoder.net[0].out_features; V = m0.volume_encoder.net[0].out_features
    T = m0.trade_encoder.net[0].out_features; HH = m0.head[0].out_features
    for m in members:
        assert (m.hidden_size, m.num_layers, m.price_encoder.net[0].out_features, m.volume_encoder.net[0].out_features,
                m.trade_encoder.net[0].out_features, m.head[0].out_features) == (H, L, P, V, T, HH), "members must have equal widths"
    big = LeanCosGroupedEncoderLSTMModel(hidden_size=K * H, num_layers=L, price_dim=K * P, volume_dim=K * V, trade_dim=K * T,
                                         head_hidden=K * HH, dropout=0.0, aux_scored=False)
    sd = {k: torch.zeros_like(v) for k, v in big.state_dict().items()}
    enc = P + V + T  # a member's encoded width before the shared `additional` block
    for k, m in enumerate(members):
        s = m.state_dict()
        for name, n in (("price_encoder", P), ("volume_encoder", V), ("trade_encoder", T)):
            sd[f"{name}.net.0.weight"][k * n:(k + 1) * n] = s[f"{name}.net.0.weight"]
            sd[f"{name}.net.0.bias"][k * n:(k + 1) * n] = s[f"{name}.net.0.bias"]
        for g in range(4):  # PyTorch gate order i, f, g, o: each gate block holds all K members
            rows = slice(g * K * H + k * H, g * K * H + (k + 1) * H); src = slice(g * H, (g + 1) * H)
            w = s["lstm.weight_ih_l0"][src]
            sd["lstm.weight_ih_l0"][rows, k * P:(k + 1) * P] = w[:, 0:P]
            sd["lstm.weight_ih_l0"][rows, K * P + k * V:K * P + (k + 1) * V] = w[:, P:P + V]
            sd["lstm.weight_ih_l0"][rows, K * (P + V) + k * T:K * (P + V) + (k + 1) * T] = w[:, P + V:enc]
            sd["lstm.weight_ih_l0"][rows, K * enc:K * enc + ADD] = w[:, enc:enc + ADD]
            sd["lstm.weight_hh_l0"][rows, k * H:(k + 1) * H] = s["lstm.weight_hh_l0"][src]
            sd["lstm.bias_ih_l0"][rows] = s["lstm.bias_ih_l0"][src]; sd["lstm.bias_hh_l0"][rows] = s["lstm.bias_hh_l0"][src]
            for layer in range(1, L):
                sd[f"lstm.weight_ih_l{layer}"][rows, k * H:(k + 1) * H] = s[f"lstm.weight_ih_l{layer}"][src]
                sd[f"lstm.weight_hh_l{layer}"][rows, k * H:(k + 1) * H] = s[f"lstm.weight_hh_l{layer}"][src]
                sd[f"lstm.bias_ih_l{layer}"][rows] = s[f"lstm.bias_ih_l{layer}"][src]
                sd[f"lstm.bias_hh_l{layer}"][rows] = s[f"lstm.bias_hh_l{layer}"][src]
        sd["head.0.weight"][k * HH:(k + 1) * HH, k * H:(k + 1) * H] = s["head.0.weight"]
        sd["head.0.bias"][k * HH:(k + 1) * HH] = s["head.0.bias"]
        sd["head.3.weight"][:, k * HH:(k + 1) * HH] = s["head.3.weight"] / K
        sd["head.3.bias"] += s["head.3.bias"] / K
    big.load_state_dict(sd)
    return big.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="CHECKPOINT_DIR:WEIGHTS_FILE ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    members = [load_model(s, torch.device("cpu")) for s in args.models]
    big = merge(members)

    torch.manual_seed(0)
    x = torch.randn(2, 600, 112)
    with torch.no_grad():
        ref = sum(m(x)[0] for m in members) / len(members)
        got = big(x)[0]
    err = (ref - got).abs().max().item()
    print(f"merged {len(members)} models: max |merged - mean of members| = {err:.3e} on random input")
    if err > 1e-4:
        raise SystemExit("merge check FAILED")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(big.state_dict(), out / "merged.pt")
    cfg = {"hidden_size": big.hidden_size, "num_layers": big.num_layers,
           "price_dim": big.price_encoder.net[0].out_features, "volume_dim": big.volume_encoder.net[0].out_features,
           "trade_dim": big.trade_encoder.net[0].out_features, "head_hidden": big.head[0].out_features,
           "aux_scored": False, "bidirectional": False, "merged_from": args.models}
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {out / 'merged.pt'} ({sum(p.numel() for p in big.parameters()):,} parameters incl. structural zeros)")


if __name__ == "__main__":
    main()
