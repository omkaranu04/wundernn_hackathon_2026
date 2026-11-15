"""STAGE 2b: train two independent XGBoost regressors (t0, t1) on the
frozen backbone's extracted hidden states (extract_embeddings.py's
output), then convert both to ONNX via onnxmltools so inference stays on
onnxruntime only - same safe, dependency-light pattern as every other
experiment's solution.py, and avoids depending on `xgboost` actually being
present in the real scoring container (only numpy/onnxruntime/pyarrow are
guaranteed - see PROBLEM_STATEMENT.md).

Early stopping uses embeddings extracted from a SEPARATE sample of
valid.parquet sequences (extract_embeddings.py's X_valid/y_valid) - a
genuinely held-out set, not a split of the training sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


def train_one(X_train, y_train, X_valid, y_valid, params, target_name: str):
    model = xgb.XGBRegressor(**params, early_stopping_rounds=30, eval_metric="rmse")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    best_iter = model.best_iteration
    pred = model.predict(X_valid)
    rmse = float(np.sqrt(np.mean((pred - y_valid) ** 2)))
    corr = float(np.corrcoef(pred, y_valid)[0, 1])
    print(f"  [{target_name}] best_iteration={best_iter}, valid RMSE={rmse:.4f}, valid corr={corr:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    X_train = np.load(emb_dir / "X_train.npy")
    y_train = np.load(emb_dir / "y_train.npy")
    X_valid = np.load(emb_dir / "X_valid.npy")
    y_valid = np.load(emb_dir / "y_valid.npy")
    print(f"X_train {X_train.shape}, X_valid {X_valid.shape}")

    params = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree, reg_lambda=args.reg_lambda,
        tree_method="hist", random_state=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("training XGBoost regressor for t0...")
    model_t0 = train_one(X_train, y_train[:, 0], X_valid, y_valid[:, 0], params, "t0")
    print("training XGBoost regressor for t1...")
    model_t1 = train_one(X_train, y_train[:, 1], X_valid, y_valid[:, 1], params, "t1")

    hidden_size = X_train.shape[1]
    for name, model in [("t0", model_t0), ("t1", model_t1)]:
        onnx_model = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, hidden_size]))])
        onnx_path = out_dir / f"xgb_{name}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  saved {onnx_path}")

    with open(out_dir / "xgb_config.json", "w") as f:
        json.dump({"hidden_size": hidden_size}, f, indent=2)

    print("done. Next: export_backbone_onnx.py, then solution.py --validation ... for the real WP number.")


if __name__ == "__main__":
    main()
