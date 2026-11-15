# 011 — LSTM backbone + XGBoost regressor head

**Status: complete and submitted. Local WP = 0.658580 (worst of the
fleet). Platform WP = 0.6305 — also worst of the fleet, below even
experiment 001.**

## Idea

Keep the LSTM as the differentiable, stateful sequence backbone (it
handles the causal/streaming requirement no tree model has a native
mechanism for), but replace the final neural head with two independent
XGBoost regressors — one per target — operating on the LSTM's hidden
state.

## Why this needs a genuinely different pipeline

XGBoost is **not differentiable** — no gradient flows through a tree
ensemble's hard splits, so "LSTM + XGBoost head" cannot be trained
end-to-end via backprop the way every other experiment's "LSTM + Linear/
MLP head" has been. This forces a **two-stage, non-differentiable
pipeline**, a first for this project:

1. **`train.py` (stage 1)**: train the backbone (`model.py`'s
   `BackboneModel` — grouped encoder + 2-layer/128-hidden LSTM, byte-for-
   byte identical shape to experiment 004) end-to-end via its own
   discardable Linear/MLP head, same 30-epoch warmup+decay+early-stop
   recipe as experiments 004/010. The head only exists to shape good
   hidden-state representations; it's thrown away afterward.
2. **`extract_embeddings.py` (stage 2a)**: freeze the trained backbone,
   run it over a **subsample** of sequences (300 train, 100 valid — the
   full 212M-row dataset would need ~108GB just for the embeddings
   at hidden_size=128, impractical), and save `(hidden_state, t0, t1)`
   for every `need_prediction` row. 300 sequences alone gives ~6M rows —
   far more than the Quant Challenge 2025 Rank-3 LightGBM solution's
   entire 80,000-row training set (see
   `../../EXTERNAL_REFERENCE_rank3_quant_challenge_2025.md`).
3. **`train_xgboost.py` (stage 2b)**: train two independent
   `XGBRegressor`s (t0, t1) on those embeddings, early-stopping against
   the held-out valid-sequence embeddings (a genuinely separate sample,
   not a split of the training sample). Both converted to ONNX via
   `onnxmltools` immediately afterward.
4. **`export_backbone_onnx.py`**: exports just the frozen encoder+LSTM
   (no head) to its own single-step ONNX graph.
5. **`solution.py`**: runs **three** `onnxruntime` sessions per step —
   backbone (features + state → hidden state + new state), then the two
   XGBoost heads (hidden state → t0, hidden state → t1).

**Why convert to ONNX rather than ship `xgboost` directly**: the
competition only guarantees `numpy`/`onnxruntime`/`pyarrow` in the real
scoring container (see `PROBLEM_STATEMENT.md` §6); `xgboost` being in the
"broader set of popular ML libs" mentioned there isn't confirmed, and
there's no internet access at inference time to install it if missing.
Converting ahead of time keeps this submission on the same safe,
dependency-light pattern as every other experiment.

New dependencies installed into the `wundernn` conda env: `xgboost`,
`onnxmltools`, `skl2onnx` (and their transitive deps, incl. `scikit-learn`,
`scipy`) — confirmed not to disturb the existing `torch`/CUDA setup
(re-checked `torch.cuda.is_available()` and the `dynamo=True` default
after installing).

## Pre-training validation (before committing to the full pipeline)

- **XGBoost → ONNX → onnxruntime round-trip**, tested in isolation on
  synthetic data first: exact numerical match (max abs diff 1.19e-07),
  confirmed input name `"input"` shape `(None, hidden_size)`, output name
  `"variable"` shape `(None, 1)` — the contract `solution.py` relies on.
- **Full 3-session orchestration**, smoke-tested end-to-end with a real
  (randomly-initialized) backbone and two throwaway XGBoost heads trained
  on unrelated synthetic data: correct warm-up `None`-handling, correct
  `(2,)` output shape, all finite. The fake heads' predictions came out
  suspiciously constant across steps — diagnosed and confirmed to be an
  artifact of training those fake heads on a mismatched distribution
  (`standard_normal`) rather than real LSTM hidden states (verified
  separately: real hidden states range ±0.11 and change measurably every
  step, ~0.03-0.04 max diff between consecutive steps) — not a bug in the
  session-orchestration/state-threading code.
- **Backbone export correctness**: verified, max abs diff 8.94e-08 vs.
  batched `encode_and_recur()`.

## Training

Backbone (stage 1): 30 epochs (warmup 3 + `ReduceLROnPlateau` decay +
early stopping patience 5), on GPU, running concurrently with experiment
004's own 30-epoch run and experiment 010. Stages 2a/2b (embedding
extraction + XGBoost training) run automatically once stage 1 finishes,
via `auto_pipeline.sh`.

## Status / next steps

- [x] Two-stage pipeline designed and all scripts written
- [x] XGBoost→ONNX round-trip validated in isolation
- [x] Full 3-session solution.py orchestration smoke-tested end-to-end
- [x] Backbone export correctness verified
- [x] Backbone trained (stage 1) — early-stopped epoch 19, best
      valid_loss=0.3434 at epoch 14 (comparable to 004/010/012's backbones,
      ~0.343-0.346)
- [x] Embeddings extracted (stage 2a)
- [x] XGBoost regressors trained + converted to ONNX (stage 2b)
- [x] Combined (backbone + 2 XGB heads) latency benchmarked on real
      trained weights: **122.9% of budget uncorrected — the worst latency
      of any experiment in this project**
- [x] Loss curve plotted (backbone only)
- [x] Packaged and verified: `submissions/011_xgboost_head_2026-09-17.zip`
- [x] Scored via `solution.py --validation ...`: **local WP = 0.658580**
      (t0=0.651217, t1=0.665943)
- [x] Compared against the rest of the fleet: **worst local WP of the
      entire fleet**, below even experiment 001's original attention model
      (0.664580). The backbone alone reached a comparable valid_loss to
      the other 30-epoch runs, so the drop is specifically in the XGBoost
      stage — a frozen backbone optimized via a discardable neural head
      doesn't necessarily hand off hidden states that a completely
      separately-trained tree ensemble can regress well; the backbone was
      never tuned with XGBoost's actual loss in the loop. Combined with the
      worst latency in the project, the two-stage hypothesis is not
      supported by these results.
- [x] Submitted to the real platform: **0.6305**, runtime 46m56s (78.2%
      of budget) — no timeout despite the 122.9% uncorrected flag (the
      worst latency of any experiment), but the worst platform WP of the
      fleet too, directionally consistent with the local result
