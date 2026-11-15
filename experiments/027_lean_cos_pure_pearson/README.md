# 027: Lean cos-expanded grouped LSTM with a pure global Pearson loss

A new approach specified by the user (2026-10-10), combining lessons from
experiments 019-026.

## Spec and how it was implemented

1. **Inputs (68 of 112 features):** instrument-0 price (22) + instrument-0
   volume (22) + ALL trade features dp/dv of both instruments (16) + the 8
   `additional`. Instrument 1's price and volume (44) are dropped (021/022:
   effectively unused). Verified by perturbation: dropped inputs have exactly
   zero influence on the output.
2. **Expansion + encoders:** every group expanded to `[x, cos(x)]` (021: `x`
   and `cos(x)` are the channels 012 relies on). Grouped MLPs as in 012
   (Linear+ReLU+Dropout): price 44->24, volume 44->24, trade 32->12.
   **Interpretation choice:** `additional` (expanded to 16 dims) goes straight
   into the LSTM with no MLP, as in 004/012 - the spec said "grouped MLPs" and
   012's additional group never had one; adding a 16->k MLP is a one-line
   change if intended.
3. **Recurrent core:** 2-layer LSTM, hidden 128; head Linear(128,64)-ReLU-
   Dropout-Linear(64,2) predicting (t0, t1) directly (012's head).
   Per-step vector 76 dims (012: 100). **248,510 params** (012: 269,822).
4. **Loss: pure global weighted Pearson** - no MSE term. "Global" is
   implemented as the pooled-over-the-batch correlation (all scored-eligible
   rows of the 4 sequences in a batch pooled into one weighted correlation per
   target, weights |clipped target|, mean of the two targets) - the closest
   affordable proxy for the dataset-global metric. `--mse-weight 0
   --pearson-scope batch`.
5. **Recipe:** unchanged - Adam 1e-3, batch 4, 3-epoch warmup,
   ReduceLROnPlateau, early stopping (patience 5), 30 epochs max, seed 0,
   checkpoint selection on corrected global WP with the true valid mask.

## Pre-training checks (random-init weights)

- Export vs batched forward: 1.5e-08 max diff. **46 ONNX nodes** (012: 62,
  004: 43).
- Latency, alternating same-run benchmarks: ~12% faster than 012 (103.0% vs
  116.6% uncorrected) and ~9% slower than 004 (107.2% vs 98.4%); absolute
  numbers inflated by machine load.

## Known risks

- **Pure Pearson is scale-invariant**, so nothing anchors the prediction
  scale; outputs beyond +-2 are clipped by the metric (and the loss clamps
  there, giving zero gradient). The training script logs the validation
  prediction std per target (`pred_std`) every epoch to watch for drift. The
  3-epoch LR warmup is the only protection against the unstable early
  gradient (the shared loss's docstring warns Pearson-only is unstable when a
  batch's weighted variance is near zero).
- Pooling over a batch is noisier than the true global metric.
- Three things change at once vs 012 (inputs, expansion, loss) - hence the
  control arm below.

## Runs (server A)

| arm | loss | status |
|---|---|---|
| **pure** (the spec) | pure pooled Pearson | **done**: best valid WP 0.680912 (epoch 9), early-stopped epoch 14 |
| control (my addition) | standard 0.5 MSE + 0.5 per-sequence Pearson, same architecture | **done**: best valid WP 0.683450 (epoch 9), early-stopped epoch 14 (auto-launched by the watcher ~1 minute after the pure arm) |

The control separates "lean architecture" from "pure Pearson loss"; without
it a result cannot be attributed. Compare both to 012 (local WP 0.682697,
platform 0.6556) and to the 026 control (real scorer 0.684065, same selection
rule). Differences under ~0.004 are unresolved.

## Results

Batched corrected global WP on valid (true scored mask), single seed, both
selected on WP, same recipe:

| | best valid WP | epoch | stopped | WP epochs 9 / 10 / 11 | val prediction std at best |
|---|---|---|---|---|---|
| 027 pure global Pearson (spec) | 0.680912 | 9 | 14 | 0.6809 / 0.6767 / 0.6806 | 1.16 / 1.19 |
| 027 control (same lean model, standard loss) | **0.683450** | 9 | 14 | 0.6835 / 0.6786 / 0.6819 | 0.83 / 0.90 |
| 026 control (012: 112 feats, 5 channels, std loss) | 0.684065 | 12 | 17 | 0.6798 / 0.6793 / 0.6832 | - |
| 026 weighted (012 + scored-prob weights) | 0.685846 | 11 | 16 | 0.6828 / 0.6826 / 0.6858 | - |

- **Pure Pearson did not help; it cost about 0.0025 WP** relative to the same
  model with the standard loss (0.6809 vs 0.6835) and 0.0032 vs the 026
  control. It is also noisier from epoch to epoch (0.6767-0.6809 over epochs
  9-14). Each of the three loss variants tested so far (pooled+MSE in 019,
  pure pooled here) has landed slightly BELOW the standard 0.5/0.5
  per-sequence blend; the MSE term seems to help rather than hurt.
- **The lean architecture holds up.** With the standard loss it reaches
  0.6835 against 012-architecture's 0.6841 (026 control): -0.0006, inside
  noise, using 68 of 112 features, 2 channels instead of 5, 248,510 params
  (vs 269,822) and a 46-node graph that benchmarked ~12% faster than 012
  on random init. This confirms experiments 021/022 (instrument-1 price and
  volume are dead weight; `x` and `cos(x)` are what 012 uses) on a full
  training run - the simpler model is as good.
- **Prediction scale (the flagged risk):** under pure Pearson the output std
  rose steadily from 0.11 (epoch 0) to ~1.2 and levelled off, below the
  +-2 clip, so no clipping or collapse problem; the standard-loss control
  stayed ~0.8-0.9.
- Not exported to ONNX/scored by the real scorer; no zip. Weights/logs in
  `results/`, `logs/` (copied back from server A, checksum-verified, deleted
  there).


## Submission zips (2026-10-10)

`submissions/027_lean_cos_pure_pearson_control_2026-10-10.zip` and
`submissions/027_lean_cos_pure_pearson_pure_2026-10-10.zip` (1.85 MB each;
export vs batched forward 5.96e-07 / 9.54e-07, 46 nodes, fresh-extraction
checks passed). Trained-weights latency, alternating runs vs 012 (same
machine state): control 106.7% vs 119.1% (~10% faster); pure 105.7% vs
118.2% (~11% faster). Only batched WP exists (0.683450 / 0.680912) - not run
through the real scorer. Platform results (submitted 2026-10-10): **control 0.6555** (ID 1K7GAMQ0, 42m42s) and **pure Pearson 0.6530** (ID 103467LH, 40m57s). The lean 68-feature model ties 012 (0.6556) on the platform at ~6% lower runtime than 026-control; the pure-Pearson loss is 0.0025 below its same-model control, exactly the local gap. Verdict: pure Pearson loses; keep the MSE term.
