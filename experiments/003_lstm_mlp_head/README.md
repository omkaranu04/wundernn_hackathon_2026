# 003 — Stacked LSTM (2 layers) + MLP prediction head

**Status: complete. Local WP = 0.634654 — new best local result, matching
experiment 002 with ~46% fewer parameters (280,898 vs. 520,450).**

## Idea

Isolate the effect of the *prediction head* as a single variable: keep the
LSTM backbone identical in shape to the baseline's GRU (2 layers,
hidden_size=128), and replace experiment 002's direct `Linear(128, 2)`
mapping with a small MLP: `Linear(128,128) → GELU → Linear(128,64) → GELU →
Linear(64,2)`. Everything else (data loading, training loop, ONNX export
pattern, submission entrypoint) is unchanged from experiment 002.

## Architecture

```
features (T, 112) -> 2-layer nn.LSTM(hidden_size=128)
                   -> Linear(128,128) -> GELU -> Dropout
                   -> Linear(128,64)  -> GELU -> Dropout
                   -> Linear(64,2)
```

No input projection, no attention, no windowed cache — same design
philosophy as experiment 002 (see its `model.py`/README for why a plain
recurrent stack doesn't need the reformulation experiment 001's attention
mechanism needed for streaming-inference correctness). The MLP head adds no
complication to that property: it's a pure function of the current step's
LSTM hidden state, with no state of its own to thread between calls.

## Pre-training latency check (random weights, before committing to training)

Learned from experiment 002's thin margin (80.7% of budget on the real
platform) to always benchmark before training, not just before submitting.
Results (single-threaded ONNX Runtime, matching the competition's 1-vCPU
constraint):

| | ms/step (synthetic) | % of budget (no correction) |
|---|---:|---:|
| Experiment 002 (4-layer LSTM, no head) | 0.1239 | 128.3% (over, without correction) |
| **This model (2-layer LSTM + MLP head)** | **0.0874** | **90.5%** |

Even with **zero** platform-speedup correction applied, this comes in under
100% of budget — a much safer starting point than experiment 002 had.
Applying the range of correction factors observed across prior submissions
(1.3x-1.8x) puts the estimate at 50-70% of budget. Correctness verified via
`export_onnx.py`'s `verify()` (max abs diff ~1.5e-8 vs. `model.forward()`),
and `solution.py` smoke-tested for determinism before training.

## Training

Smoke-tested at ~1.87 s/batch on `valid.parquet` (vs. experiment 002's
3.2 s/batch — faster, as expected from half the LSTM depth). Launched full
training on `train.parquet`, 5 epochs, batch size 4:

- PID: (see EXPERIMENTS.md for current run details)
- Estimated: ~1.6-1.7h/epoch → ~8-8.5h total (vs. experiment 002's ~13.4h)

## Files

Same structure as experiment 002: `model.py`, `dataset.py`, `train.py`,
`export_onnx.py`, `solution.py`. No `inference.py` needed (no growing cache
requiring a streaming-specific reformulation).

## Status / next steps

- [x] Architecture designed, latency-benchmarked before training (safe margin)
- [x] Export pipeline (dynamo + onnxsim) verified correct
- [x] `solution.py` smoke-tested: correct shapes, deterministic
- [x] Trained on full `train.parquet`: 5 epochs, **8h38m39s**, best
      `valid_loss=0.3581` (a parameter count of 280,898 — see EXPERIMENTS.md's
      parameter/timing table)
- [x] Exported to ONNX on trained weights, re-verified correct (diff ~8.3e-7),
      re-benchmarked latency: **0.0882 ms/step, 91.3% of budget uncorrected**
      (consistent with the pre-training estimate of 90.5%)
- [x] Scored via `solution.py --validation ...`: **WP = 0.634654**
      (t0=0.634576, t1=0.634733, 1872/1873 eligible blocks). Took 1h07m32s
      wall-clock locally.
- [x] Packaged and verified: `../../submissions/003_lstm_mlp_head_2026-09-13.zip`
      — tested from a fresh extraction directory (`ZIP VERIFICATION PASSED`:
      correct shapes/None-handling, deterministic) before trusting it, via
      the automated post-training pipeline (`auto_pipeline.sh`).
- [x] Compared: **new best local result**, edging out experiment 002
      (0.634285 local / 0.6106 platform) by +0.000369 — a negligible margin
      in absolute WP terms, but achieved with **~46% fewer parameters**
      (280,898 vs. 520,450). Also beats experiment 001 (0.622269 local /
      0.5977 platform) and the baseline (0.589595 local / 0.5719 platform).
- [x] Submitted to the real platform (2026-09-13 23:23, ID `1CTSEB85`):
      **platform WP = 0.6097**, runtime **36m51s (61.4% of budget)**, 627MB.
      Beats baseline (0.5719) and experiment 001 (0.5977) on the real test
      set; essentially ties experiment 002's platform score (0.6106, -0.0009).
      Real per-step latency 0.0593ms/step vs. our synthetic 0.0882ms/step -
      a 1.49x platform speedup, consistent with the established pattern
      (see EXPERIMENTS.md for the cross-experiment comparison, including
      experiment 004's contradictory result).

## Key finding: the MLP head substitutes for LSTM depth

Experiment 002 needed 4 LSTM layers (520K params, tightest platform margin
of any submission at 80.7% of budget) to reach 0.634285 local WP. This
experiment reaches essentially the same score (0.634654) with only 2 LSTM
layers plus a 3-layer MLP head (280,898 params total) - and with a much
safer latency margin (91.3% vs. experiment 002's 128.3% uncorrected). If
this pattern holds on the real platform too, it suggests **head capacity is
a more parameter-efficient way to add model capacity than recurrent depth**
for this task, at least in this size range - worth deliberately exploring
further (e.g. a deeper/wider head with an even shallower LSTM) in a future
experiment.

## Metric correction (2026-09-16)

The organizers found a mistake in their WP calculation and re-evaluated
every submission under a corrected "Global Weighted Pearson correlation"
metric (pooled across the whole dataset, not averaged per sequence — see
`PROBLEM_STATEMENT.md` §3 and `EXPERIMENTS.md`). This experiment's real
platform score changed: **~~0.6097~~ → 0.6499**. Runtime/memory are
unaffected. The local WP figure above (0.634654) was computed with the
old, now-stale local scorer.
