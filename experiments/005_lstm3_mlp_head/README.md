# 005 — Stacked LSTM (3 layers) + MLP prediction head

**Status: training launched, not yet scored.**

## Idea

A direct follow-up to experiment 003 (2-layer LSTM + 3-layer MLP head,
0.634654 local WP / 0.6097 platform WP — our best local result). After
finding that a **4-layer** LSTM + the same MLP head benchmarked too risky
(145.9% of budget uncorrected — see the conversation history / EXPERIMENTS.md
for the full reasoning), this experiment tests the middle ground: **3**
LSTM layers with the same MLP head, to see whether the extra recurrent
depth adds anything over experiment 003's result without the 4-layer
version's latency risk.

## Architecture

```
features (T, 112) -> 3-layer nn.LSTM(hidden_size=128)
                   -> Linear(128,128) -> GELU -> Dropout
                   -> Linear(128,64)  -> GELU -> Dropout
                   -> Linear(64,2)
```

Identical to experiment 003 except `num_layers=3` instead of `2`. Code is
copied from experiment 003 (`model.py`/`dataset.py`/`export_onnx.py`/
`solution.py` are unchanged - already parameterized by `num_layers`), not
edited in place, per the project convention of never modifying a completed
experiment. Parameter count: 412,994 (vs. experiment 003's 280,898).

## Pre-training latency check (random weights, before committing to training)

This is a **deliberate, informed bet, not an oversight** - unlike
experiments 003/004, this config does *not* clear the "safe even with zero
platform speedup" bar:

| Correction factor | % of budget |
|---|---:|
| None (1.0x) | 120.0% |
| Exp004's factor (0.87x, worst seen) | 138.0% |
| Exp003's factor (1.49x) | **80.6%** |
| Exp002's factor (1.59x) | 75.5% |

For comparison: the 4-layer version of this same combination benchmarked at
145.9% uncorrected (rejected as too risky). This 3-layer version is
meaningfully safer (120.0%), and under the two correction factors actually
observed for LSTM+MLP-head-style architectures so far (1.49x-1.59x), it
lands at 75.5-80.6% - comparable to experiment 002's own real-world margin
(80.7% actual, which worked). It does *not* have the safety margin
experiments 003/004 had (both were under 93% uncorrected, safe regardless
of any correction factor).

Correctness verified via `export_onnx.py`'s `verify()` (max abs diff
~3.0e-8 vs. `model.forward()`), and `solution.py` smoke-tested for
determinism before training.

## Training

Smoke-tested at ~2.56-2.63 s/batch on `valid.parquet` (between experiment
003's 1.87s/batch and experiment 002's 3.2s/batch, as expected for 3
layers). Launched full training on `train.parquet`, 5 epochs, batch size 4,
estimated ~10.5-11h total. Automated post-training pipeline
(`auto_pipeline.sh`) armed to export, score, package, and verify
immediately once training finishes - see EXPERIMENTS.md for current run
details (PID, log paths).

## Files

Same structure as experiments 002/003: `model.py`, `dataset.py`, `train.py`,
`export_onnx.py`, `solution.py`, `auto_pipeline.sh`. No `inference.py`
needed.

## Status / next steps

- [x] Architecture designed (reused from experiment 003, `num_layers=3`)
- [x] Latency-benchmarked before training: 120.0% of budget uncorrected -
      a deliberate bet, not a safe-regardless-of-factor config
- [x] Export pipeline (dynamo + onnxsim) verified correct
- [x] `solution.py` smoke-tested: correct shapes, deterministic
- [x] Trained on full `train.parquet`: 5 epochs, **10h39m24s**, best
      `valid_loss=0.3563`
- [x] Exported to ONNX on trained weights, re-verified correct,
      re-benchmarked latency: **0.1167 ms/step, 120.9% of budget
      uncorrected** — matches the 120.0% pre-training estimate almost
      exactly. **This model still depends on a platform correction factor
      to fit within 60 minutes - it has not cleared the "safe regardless of
      factor" bar the way experiments 003/004 did.**
- [x] Scored via `solution.py --validation ...`: **WP = 0.635169**
      (t0=0.633118, t1=0.637221, 1872/1873 eligible blocks) - **new best
      local result**, beating experiment 003's 0.634654. Local scoring took
      1h21m40s.
- [x] Packaged and verified: `../../submissions/005_lstm3_mlp_head_2026-09-14.zip`
      (`ZIP VERIFICATION PASSED` via the automated pipeline).
- [x] Compared: beats experiment 003 (0.634654 local / 0.6097 platform),
      experiment 002 (0.634285 local / 0.6106 platform), experiment 004
      (0.633608 local / 0.6089 platform), experiment 001 (0.622269 local /
      0.5977 platform), and the baseline (0.589595 local / 0.5719 platform)
      **locally**.
- [x] Submitted to the real platform (2026-09-14 13:03, ID `23LRV6SU`):
      **platform WP = 0.6108** — **new best platform score**, edging out
      experiment 002 (0.6106) and experiment 003 (0.6097). Runtime
      **61m34s** and 630MB memory.

## The latency gamble: it paid off this time, but the risk was real

This config never cleared the "safe regardless of platform factor" bar
(120.9% of budget uncorrected, vs. experiment 003's 91.3%) - see the
pre-submission analysis above, which explicitly called this a gamble, not
a safe bet. **The gamble paid off in the sense that it scored the best WP
of any submission so far - but it did so by running 61m34s, over the
nominal 60-minute limit**, matching experiment 004's 61m32s overrun. This
is now the *second* submission of ours to exceed the nominal limit and
still be scored, confirming there's some tolerance beyond the stated
cutoff. See `EXPERIMENTS.md`'s policy note: **we are not going to design
around or rely on that tolerance going forward** - it's undocumented,
observed on only 2 data points, and could plausibly not always apply.
Future experiments should continue targeting genuine margin under 60
minutes on their own merits, not a number that happens to work because of
an unstated grace period. This experiment's result stands as a useful data
point, not a template to repeat.

## Metric correction (2026-09-16)

The organizers found a mistake in their WP calculation and re-evaluated
every submission under a corrected "Global Weighted Pearson correlation"
metric (pooled across the whole dataset, not averaged per sequence — see
`PROBLEM_STATEMENT.md` §3 and `EXPERIMENTS.md`). This experiment's real
platform score changed: **~~0.6108~~ → 0.6473** — a significant reversal.
Under the old metric this was our best platform score; under the
corrected metric it's now **near the bottom** of the post-baseline
cluster, behind 002, 003, 004, and 006. The "latency gamble paid off"
framing above should be read with that in mind — the gamble bought a
runtime that (barely) worked, but the resulting model turned out not to
be our best once scored correctly. Runtime/memory are unaffected. The
local WP figure above (0.635169, our former best local score) was
computed with the old, now-stale local scorer.
