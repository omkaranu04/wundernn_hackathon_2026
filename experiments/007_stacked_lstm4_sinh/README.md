# 007 — Experiment 002's 4-layer stacked LSTM + `sinh`-activated head

**Status: trained, locally scored, and submitted (2026-09-15 23:58, ID
1G6E2VEI). `sinh` did NOT help, isolated — confirmed both by the old local
scorer AND the corrected platform metric — see results below.**

## Idea

Isolate the `sinh` output transform (see experiment 006's README for the
full arcsinh/log reasoning) as the *sole* experimental variable against
**experiment 002** — our best real-platform-confirmed result within the
time limit (0.6106 WP, 48m25s / 80.7% of the 60-minute budget, no reliance
on any grace period), rather than 005's higher-but-over-the-limit 0.6108.
Same isolation logic experiment 003 used when it tested the MLP head as
the sole variant vs. 002.

Backbone is byte-for-byte identical to experiment 002: `nn.LSTM(hidden_
size=128, num_layers=4)` -> `Linear(128, 2)`. The only change: the
`Linear` output is passed through `sinh(clamp(z, -8, 8))` instead of used
raw, same convention as experiment 006.

## Latency risk — flagged and accepted explicitly before training

Experiment 002 was already the tightest-margin submission to date
(48m25s / 80.7% actual), so this needed real scrutiny, not an assumption
that "sinh is basically free."

**Pre-training benchmarks** (single-threaded ONNX Runtime, matching
`solution.py`'s session options, random weights):

| Model | ONNX nodes | ms/step | % of budget (uncorrected) |
|---|---:|---:|---:|
| 002 (original, documented) | 28 | 0.1240 | 128.3% |
| 002 (re-measured just now, for a fair comparison) | 28 | 0.1262 | 130.6% |
| **007 (002 + sinh)** | 30 | **0.1357** | **140.5%** |

The re-measurement of 002 (under the exact same system load as 007's
benchmark, including experiment 006's training running concurrently)
landed within ~2 points of its original documented number — confirming
the concurrent job isn't meaningfully contaminating these single-threaded
measurements. So the ~10-point jump from 130.6% to 140.5% is real, not
noise: it's exactly proportional to the 2 extra ONNX nodes `sinh`/`clamp`
add (28 -> 30, a 7.1% node-count increase ≈ the observed 7.5% latency
increase) — consistent with this project's established finding that
per-step latency for these small graphs is dominated by fixed per-node
dispatch overhead, not raw FLOP count (see experiment 004's README).

**Projected real runtime**: experiment 002's real platform run got a
1.59x speedup over its synthetic number. Applying the range of correction
factors observed so far for non-branching ("simple sequential")
architectures (1.49x-1.9x — this graph stays purely sequential, no
branching, so it should stay in that class rather than experiment 004's
0.87x branching-architecture outlier) to 007's 140.5% raw projects to
roughly **74-94% of budget, ~44-57 minutes real** — plausibly fine, but
the low end of that range (1.49x, the most conservative factor observed)
leaves only ~3-4 minutes of spare margin, well under experiment 002's own
11m35s.

**Decision**: flagged this explicitly to the user given the "don't cross
the mark" constraint on this specific model; decided to proceed with
training and get the real trained-weight benchmark + real platform number
rather than pre-emptively changing the config. `auto_pipeline.sh`
re-benchmarks latency on trained weights (not just random-weight ones)
immediately after export, before packaging — see its output for the
number that actually matters.

## Files

`model.py`, `train.py`, `export_onnx.py` are new; `dataset.py` and
`solution.py` are copied unchanged from experiment 002 (only docstrings
updated). No `inference.py` needed, same reasoning as 002/003/005/006.

## Status / next steps

- [x] Architecture designed: 002's exact 4-layer LSTM backbone + `sinh`-
      activated head, isolating `sinh` as the sole variable
- [x] Export pipeline (dynamo, required for `sinh` support) verified
      correct: max abs diff 3.73e-08
- [x] Latency benchmarked and risk explicitly flagged before training:
      140.5% of budget uncorrected (vs. 002's own 130.6% under identical
      load) — a real, node-count-proportional increase, not noise.
      Decision: proceed, re-check on trained weights before submitting.
- [x] Trained on full `train.parquet`: 5 epochs, **14h49m29s**, best
      `valid_loss=0.3555`
- [x] Exported to ONNX on trained weights, re-verified correct (max abs
      diff 1.431e-06), **re-benchmarked latency on trained weights:
      0.1241 ms/step, 128.5% of budget uncorrected** — notably better
      than the 140.5% pre-training estimate (that number was measured
      while experiment 006 was training concurrently and contending for
      CPU; this one was measured solo, after 006 finished, so it's the
      fairer/more accurate figure). Essentially matches experiment 002's
      own historical number (128.3%) — the real-world latency cost of
      `sinh` turned out to be much smaller than the load-contaminated
      benchmark suggested.
- [x] Scored via `solution.py --validation ...`: **WP = 0.633685**
      (t0=0.632919, t1=0.634451, 1872/1873 eligible blocks)
- [x] Packaged and verified: `../../submissions/007_stacked_lstm4_sinh_2026-09-14.zip`
      (`ZIP VERIFICATION PASSED`)
- [x] Compared against experiments 000-006 — see "Result" below
- [ ] **Not submitting** — see "Result" below for why

## Result: `sinh`, isolated, did not help — it slightly hurt

| # | Approach | Local WP |
|---|---|---:|
| **002** | **4-layer LSTM, plain Linear head (no sinh)** | **0.634285** |
| **007** | **002 + `sinh(clamp(z,-8,8))` output activation** | **0.633685** |

007 is the cleanest possible test of "does the `sinh`/arcsinh output
reparameterization help" — identical backbone to 002, sinh is the only
variable changed. Result: **007 scored *lower* than 002 (-0.0006)**, not
higher. Combined with experiment 006 (encoder + wide LSTM + sinh, also
not a new best, also below the 002/003/005 cluster), this is now two
independent data points where adding `sinh` did not produce the expected
gain from better-handling the targets' fat tails.

**Original decision: not submitting 007.** It's strictly dominated by 002
locally: 007 scores lower AND carries a real latency risk 002 doesn't have
(128.5% uncorrected vs. 002's 128.3%, needing a platform speedup to fit
under 60 minutes, same class of risk flagged before training). It was
submitted anyway (2026-09-15 23:58, ID 1G6E2VEI) — see the metric
correction below for the real outcome, which confirms the negative result
held up.

**Takeaway for future experiments**: the `sinh`/arcsinh output
reparameterization, as implemented here (final-layer `sinh(clamp(z,-8,8))`
in place of a plain Linear/MLP head, loss functions unchanged), does not
appear to be the free win the target-distribution analysis suggested. It's
possible a different clamp bound, a different loss (e.g. computing loss in
the pre-sinh/arcsinh space rather than raw units), or combining it with
something else would fare differently, but as a standalone architectural
change on top of an already-strong backbone, it's a negative result on two
different architectures now (006, 007) — confirmed on the real, corrected
platform metric too (see below), not just the stale local scorer.

## Metric correction (2026-09-16) and real platform result

The organizers found a mistake in their WP calculation and re-evaluated
every submission under a corrected "Global Weighted Pearson correlation"
metric (pooled across the whole dataset, not averaged per sequence — see
`PROBLEM_STATEMENT.md` §3 and `EXPERIMENTS.md`). This experiment's real
(corrected) platform score is **0.6495** vs. **experiment 002's 0.6507**
— `sinh` is still worse than the plain head under the corrected metric,
confirming the local-scorer finding wasn't an artifact of the (now-stale)
old metric. The latency risk we'd flagged also resolved favorably: real
runtime was **50m30s** (84% of budget), comfortably under the limit —
within the ~44-57 min range our re-benchmark had projected, and notably
better than the pessimistic under-load estimate that triggered the
original risk flag.
