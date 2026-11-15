# 009 — Per-instrument split-grouped-encoder LSTM

**Status: trained (on GPU), locally scored, and submitted. Hypothesis
not confirmed — see result below.**

## Idea

User-specified architecture:

```
i0_p  (22) -> 8   \
i1_p  (22) -> 8    \
i0_v  (22) -> 8     \
i1_v  (22) -> 8      >-- concat (44) -> LSTM(hidden=64, layers=4) -> head (64->16->2)
i0_dp (4)  -> 2      /
i0_dv (4)  -> 2     /
i1_dp (4)  -> 2    /
i1_dv (4)  -> 2   /
a0-a7 (8)  -> 4  /
```

A finer-grained version of experiment 004's grouped-encoder idea: 004
merged i0+i1 together within each feature type (one shared encoder
covering both instruments' price columns, etc.); this keeps i0 and i1
**separate** per type — 9 independent projections instead of 004's ~5 —
testing whether letting the model learn instrument-specific compressions
first, before any cross-instrument mixing happens inside the LSTM,
captures more than sharing an encoder across both instruments does.

## First attempt: 9 literal `nn.Linear` branches — a real latency risk, not trained

Built exactly as specified (9 separate `Linear -> GELU` modules, one per
group, concatenated). Pre-training checks:

- **Correctness**: verified, max abs diff 2.98e-08 vs. batched forward.
- **Latency**: **117 ONNX nodes, 0.1329 ms/step, 137.6% of budget
  uncorrected.** Each small `Linear->GELU` branch decomposes into several
  ONNX primitives (`Gemm`, `Erf`, `Mul`, `Div`, `Add` for the GELU alone),
  so 9 small branches multiply out to ~90+ nodes before the LSTM even
  runs.

This is explicitly a **branching architecture**, and experiment 004
already taught us — the hard way, via a real platform result — that
branching designs can't be assumed to get any platform speedup (004's
real run went *slower* than its local benchmark, 0.87x, the only time
that's happened across 9 submissions). Our established rule for that
architecture class: target <70% of budget uncorrected, not the ~90-100%
that's fine for simple sequential ones. 137.6% is roughly double that
threshold, and already worse than 004's *original, pre-fix* attempt
(123.6%, which took three rounds of tuning to rescue). **Flagged to the
user before training; not trained as-is.**

## The fix: block-diagonal single-Linear re-engineering

Presented four options (fix via re-engineering, fix via simplifying like
004 did, accept the risk, or drop the idea). Chose the re-engineering
path: `model.py`'s `BlockDiagonalEncoder` computes the **mathematically
identical function** — each of the 9 groups still gets its own learned
linear projection, with zero cross-group mixing — but as a single
`Linear(112, 44)` whose weight is multiplied by a fixed (non-trainable)
block-diagonal 0/1 mask before use, followed by **one shared `GELU`**
over the whole 44-dim output. This is valid because GELU is elementwise:
applying it once to the concatenation of 9 group outputs is identical to
applying it per-group first, then concatenating.

**Verified equivalence** (not just asserted): perturbing one group's raw
inputs and checking that only that group's assigned output slice changes,
for all 9 groups — passed cleanly, confirming zero cross-group leakage in
the masked-weight implementation, exactly matching what 9 separate
`nn.Linear` modules would give.

**Result**: 117 → **42** ONNX nodes, 137.6% → **92.0%** of budget
uncorrected. The graph's `Concat` count dropped to 2 — the same count as
experiment 002's plain sequential graph — meaning this is no longer
meaningfully "branchy" in the ONNX-graph sense; it's structurally close
to a simple sequential architecture with one extra masked-Linear block up
front. 92.0% is comparable to experiment 003's own historical margin
(91.3% uncorrected, which worked out to just 61.4% real platform usage).

## Architecture (final)

```
features (112) -> BlockDiagonalEncoder (9 independent per-group
                   projections, implemented as 1 masked Linear + 1 GELU) -> 44
                -> LSTM(hidden_size=64, num_layers=4) -> 64
                -> Linear(64,16) -> GELU -> Dropout -> Linear(16,2)
```

134,046 params.

## Pre-training checks (final, block-diagonal version)

- **Correctness**: `export_onnx.py`'s `verify()` — max abs diff = 2.98e-08
  vs. batched `model.forward()`. Clean pass.
- **Block-diagonal isolation**: perturbation test (see above) — PASSED,
  confirms the masked-Linear implementation is mathematically identical
  to 9 separate encoders.
- **Latency**: 0.0889 ms/step, **92.0% of budget uncorrected** — clears
  a reasonable safety margin, comparable to experiment 003's own safe
  historical result.

## Training

Launched on `train.parquet`, 5 epochs, batch size 4, default args
(hidden_size=64, num_layers=4, head_dims=[16], loss=mse+pearson).
Automated post-training pipeline (`auto_pipeline.sh`) armed to export,
re-benchmark, score (corrected global metric), package, and verify
immediately once training finishes.

**Restarted on GPU (2026-09-16, ~56% into epoch 1 on CPU):** the
`wundernn` conda environment's PyTorch build turned out to be CPU-only
despite an idle GTX 1080 Ti sitting on the machine the whole time —
installed the CUDA build of the *same* torch version already in use
(2.14.0+cu126, not a version downgrade, to keep the validated `dynamo=True`
ONNX export default intact) and restarted training from scratch with
`--device cuda`. Measured **~15x speedup** in a synthetic benchmark
(0.126 s/batch GPU vs. ~1.87 s/batch CPU); real training throughput came
in a bit lower (~0.3-0.4 s/batch, data-loading-bound rather than
GPU-compute-bound) but still a large win — a run that would have taken
7-15+ hours on CPU should finish in well under 2 hours. Only the
*training* device changed; export/scoring/inference remain CPU-only
(ONNX Runtime), matching the actual submission environment exactly, so
this has zero effect on what gets submitted.

## Files

`model.py`, `train.py`, `export_onnx.py` are new; `dataset.py` and
`solution.py` are copied unchanged from experiment 002 (only docstrings
updated). No `inference.py` needed, same reasoning as every prior
experiment.

## Status / next steps

- [x] Architecture designed per user spec: 9 per-instrument/per-type
      projections -> LSTM(64, 4 layers) -> head (64->16->2)
- [x] First attempt (9 literal branches) benchmarked and flagged as a
      real latency risk (137.6% of budget uncorrected, 117 nodes) —
      not trained
- [x] Re-engineered as a mathematically-equivalent block-diagonal masked
      Linear, verified equivalent via perturbation test
- [x] Re-benchmarked: 92.0% of budget uncorrected, 42 nodes — clears a
      reasonable safety margin
- [x] Trained on full `train.parquet` (on GPU, after restarting from a
      CPU run — see "Training" above): 5 epochs, **58m57s total**, best
      `valid_loss=0.3701`
- [x] Exported to ONNX on trained weights, re-verified correct (max abs
      diff 4.172e-07), **re-benchmarked latency: 0.0875 ms/step, 90.6% of
      budget uncorrected** — clears the safe bar, consistent with the
      pre-training estimate (92.0%)
- [x] Scored via `solution.py --validation ...` (corrected global
      metric): **WP = 0.671075** (t0=0.665366, t1=0.676783)
- [x] Packaged and verified: `../../submissions/009_split_grouped_encoder_lstm_2026-09-16.zip`
      (`ZIP VERIFICATION PASSED`)
- [x] Compared against experiments 000-008 — see "Result" below
- [x] **Submitted** (2026-09-16 18:30, ID `1XUGUBSI`): **platform WP =
      0.6431**, runtime **38m49s (64.7% of budget)**, 625MB

## Result: hypothesis not confirmed, consistent on both metrics

| # | Approach | Local WP | Platform WP |
|---|---|---:|---:|
| 001 | LSTM + causal windowed attention | 0.664580 | 0.6403 |
| 002 | Stacked LSTM, 4 layers (plain head) | **0.677570** | 0.6507 |
| 003 | Stacked LSTM, 2 layers + MLP head | 0.676568 | 0.6499 |
| 004 | Per-feature-type encoders (i0+i1 merged) + LSTM | 0.677315 | **0.6508** |
| 005 | Stacked LSTM, 3 layers + MLP head | 0.677159 | 0.6473 |
| 006 | Wide single LSTM(224) + sinh head | 0.675941 | 0.6502 |
| 007 | Stacked LSTM, 4 layers + sinh head | 0.676501 | 0.6495 |
| 008 | Stacked LSTM, 6 layers/32-dim (narrow-deep) | 0.672887 | 0.6449 |
| **009** | **Per-instrument split-grouped encoder (this experiment)** | **0.671075** | **0.6431** |

009 ranks **8th of 9** post-baseline submissions on *both* metrics —
only ahead of experiment 001 (the attention model), and notably below
experiment 004, the architecturally closest comparison: 004 merges i0+i1
into shared per-type encoders, this experiment keeps them fully separate.
Keeping i0/i1 separate did **not** capture more than sharing did — if
anything, giving the model less shared structure to lean on made things
slightly worse (0.671075 vs 004's 0.677315 locally; 0.6431 vs 0.6508 on
platform).

**This is now four consecutive architecture variations (006, 007, 008,
009) that have failed to beat the plain stacked-LSTM cluster
(002/003/004/005, all landing 0.6473-0.6508 on the platform).** Combined
with the earlier finding that 002/003/005 already clustered tightly
despite very different depths/parameter counts, this is a fairly strong
signal that this general architecture family (recurrent backbone + linear
head, whatever the exact encoder/depth/width/activation) has a ceiling
around 0.647-0.651 platform WP for this task — further gains likely need
a genuinely different lever (loss function shaping, engineered features,
more training data/epochs) rather than another backbone variant.
