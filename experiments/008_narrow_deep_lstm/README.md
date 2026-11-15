# 008 — Narrow-and-deep stacked LSTM (hidden=32, layers=6)

**Status: trained, locally scored, and submitted (2026-09-16 11:24, ID
23SV983I). Hypothesis not confirmed on accuracy, on either metric — see
result below.**

## Idea

User hypothesis: for a fixed compute/latency budget, trading LSTM width
for depth might capture more temporal structure than the reverse — i.e.
reduce `hidden_size` and stack more layers, rather than stay wide and
shallow. Same architecture shape as experiment 002 (plain multi-layer
`nn.LSTM` + direct `Linear` head, no MLP, no input projection) so that
depth-vs-width is the only variable changed.

## Latency sweep: this is NOT a smooth width-for-depth tradeoff

Before picking a specific (hidden_size, num_layers) pair, benchmarked 14
combinations (single-threaded ONNX Runtime, random weights, same session
options as `solution.py`) to map out the actual latency landscape rather
than guess:

| hidden | layers | params | ONNX nodes | ms/step | % of budget |
|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 520,450 | 28 | 0.1282 | 132.7% (experiment 002, for reference) |
| 96 | 5 | 378,818 | 33 | 0.1219 | 126.2% |
| 80 | 6 | 321,442 | 38 | 0.1226 | 126.9% |
| 64 | 8 | 278,658 | 48 | 0.1330 | 137.7% |
| 64 | 10 | 345,218 | 58 | 0.1627 | 168.5% |
| 48 | 12 | 238,178 | 68 | 0.1654 | 171.2% |
| **32** | **6** | **60,994** | **38** | **0.0870** | **90.1%** ← only config under 100% |
| 32 | 8 | 77,890 | 48 | 0.1047-0.1079 | 108.4-111.7% |
| 32 | 10 | 94,786 | 58 | 0.1259 | 130.4% |
| 32 | 12 | 111,682 | 68 | 0.1451 | 150.2% |
| 24 | 8 | 46,898 | 48 | 0.0993 | 102.9% |
| 24 | 10 | 56,498 | 58 | 0.1211 | 125.4% |
| 16 | 8 | 23,586 | 48 | 0.0939 | 97.2% |
| 40 | 8 | 116,562 | 48 | 0.1092 | 113.1% |

**Key finding**: ONNX node count (one `LSTM` op per layer, plus graph
glue) dominates per-step latency far more than raw parameter count or
FLOPs — consistent with the project's established "node count dominates
small-graph latency" lesson (first found in experiment 004's encoder
branching, confirmed again in experiment 007's `sinh` addition). Evidence
here: `hidden=48, layers=12` (238,178 params — under half of experiment
002's) is the *slowest* config in the whole sweep (171.2%), because 12
LSTM-layer nodes cost more in fixed per-node dispatch overhead than the
narrower width saves in compute. Even `hidden=16, layers=8` (23,586
params — tiny) still costs more (97.2%) than `hidden=32, layers=6`
(90.1%), because it pays for 8 layers' worth of node overhead instead of
6's.

**`hidden=32, layers=6` is a genuine local optimum**, not the edge of a
monotonic curve — nearby points in both directions (wider-shallower like
80/6, or narrower-deeper like 32/8) measure worse. It is also the *only*
configuration across all 14 tested that clears the project's "safe
regardless of platform correction factor" bar (<90-95% of budget
uncorrected) — every other depth-focused submission so far (002, 005,
007) has needed the platform's observed 1.49x-1.9x speedup to actually
fit under 60 minutes. This one doesn't need that speedup to already look
safe on paper.

## Architecture

```
features (T, 112) -> nn.LSTM(hidden_size=32, num_layers=6) -> Linear(32, 2)
```

60,994 params — 88% fewer than experiment 002's 520,450, while having 2
more recurrent layers.

## Pre-training checks (random weights, before committing to training)

- **Correctness**: `export_onnx.py`'s `verify()` — max abs diff = 3.35e-08
  vs. batched `model.forward()`. Clean pass.
- **Latency**: re-confirmed with the official single-threaded benchmark
  (matching `solution.py`'s exact session options): **0.0873 ms/step,
  90.4% of the 0.0966 ms/step budget, uncorrected** — consistent with the
  sweep's 90.1%. Clears the safety bar with no platform-speedup
  dependence needed.

## Training

Launched on `train.parquet`, 5 epochs, batch size 4, default args
(hidden_size=32, num_layers=6, loss=mse+pearson). Automated post-training
pipeline (`auto_pipeline.sh`) armed to export, re-benchmark, score (using
the corrected "Global Weighted Pearson correlation" metric — see
`EXPERIMENTS.md`'s metric-correction note, `wnn_connectome_starterpack/
utils.py` has already been updated by the organizers), package, and
verify immediately once training finishes.

## Files

`model.py`, `train.py`, `export_onnx.py` are new (re-parameterized copies
of experiment 002's); `dataset.py` and `solution.py` are copied unchanged
(only docstrings updated). No `inference.py` needed, same reasoning as
002/003/005/006/007.

## Status / next steps

- [x] Latency-swept across 14 (hidden_size, num_layers) combinations to
      find a genuine local optimum rather than guessing
- [x] Architecture designed: hidden=32, layers=6, isolating depth-vs-width
      as the sole variable vs. experiment 002
- [x] Export pipeline verified correct (max abs diff 3.35e-08)
- [x] Latency-benchmarked before training: 90.4% of budget uncorrected —
      clears the safe bar, no platform-speedup dependence needed
- [x] Trained on full `train.parquet`: 5 epochs, **7h10m10s**, best
      `valid_loss=0.3645`
- [x] Exported to ONNX on trained weights, re-verified correct (max abs
      diff 5.066e-07), **re-benchmarked latency: 0.0858 ms/step, 88.8% of
      budget uncorrected** — even slightly better than the pre-training
      estimate (90.4%), still clears the safe bar with no platform-speedup
      dependence needed
- [x] Scored via `solution.py --validation ...` (corrected global metric):
      **WP = 0.672887** (t0=0.667781, t1=0.677992)
- [x] Packaged and verified: `../../submissions/008_narrow_deep_lstm_2026-09-16.zip`
      (`ZIP VERIFICATION PASSED`)
- [x] Compared against experiments 001-007 — see "Result" below
- [x] **Submitted anyway (2026-09-16 11:24, ID 23SV983I)** despite the
      not-recommended verdict below — real platform WP: **0.6449**,
      runtime **37m27s (62.4% of budget)**, 623MB. Confirms the local
      finding: 7th of 8 on the real platform too, only ahead of
      experiment 001 (0.6403) — below 002/003/004/005/006/007
      (0.6473-0.6508)

## Result: the depth-for-width hypothesis is not confirmed on accuracy

| # | Approach | Local WP (corrected) |
|---|---|---:|
| 001 | LSTM + causal windowed attention | 0.664580 |
| **002** | **Stacked LSTM, 4 layers/128-dim (the config this experiment varies)** | **0.677570** |
| 003 | Stacked LSTM, 2 layers + MLP head | 0.676568 |
| 004 | Per-feature-type encoders + LSTM | 0.677315 |
| 005 | Stacked LSTM, 3 layers + MLP head | 0.677159 |
| 006 | Wide single LSTM(224) + sinh head | 0.675941 |
| 007 | Stacked LSTM, 4 layers + sinh head | 0.676501 |
| **008** | **Stacked LSTM, 6 layers/32-dim (this experiment)** | **0.672887** |

008 lands **7th of 8** — only beating experiment 001, and a real,
non-trivial 0.0047 below experiment 002 (the config it directly varies).
So: trading width for depth, at least at this specific point (32-wide,
6 layers vs. 128-wide, 4 layers), **did not capture more than width did**
— if anything it captured somewhat less, despite adding 2 recurrent
layers. The one place the hypothesis *did* pay off was latency: 88.8% of
budget uncorrected is the best safety margin of any submission so far,
needing no platform speedup at all (every other depth-focused submission
— 002, 005, 007 — has needed the platform's observed 1.49x-1.9x speedup
to actually fit under 60 minutes).

**Not submitting**: it doesn't beat any already-submitted result on
accuracy (002 is both already submitted and better on every metric we
have), so there's no accuracy case for spending a submission slot on it.
The latency-safety margin is real and could have standalone value (e.g.
as an insurance/fallback submission if a future higher-accuracy model's
timing risk doesn't pan out), but that's a different use case than "best
model" and not a reason to submit it now.

**Takeaway for future experiments**: this doesn't rule out depth mattering
in general — it rules out *this specific* narrow-and-deep trade being a
free accuracy win. The latency sweep (see above) also revealed the
(hidden_size, num_layers) latency landscape has real local optima rather
than a smooth tradeoff — if depth is worth revisiting, it'd be worth
sweeping accuracy the same systematic way latency was swept here, rather
than assuming any one depth/width point is representative of "depth" as a
lever in general.
