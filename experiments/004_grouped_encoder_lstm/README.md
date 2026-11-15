# 004 — Grouped feature-type encoders + stacked LSTM + MLP head

**Status: complete. Local WP = 0.633608** (233,194 params — the fewest of
any experiment so far, essentially tying experiments 002/003 which both
have more than double the parameters).

## Idea

Motivated directly by `analysis/README.md`'s findings: price, volume, and
trade features have very different scales and shapes (price features
heavily quantized with low variance ~0.15; volume features variance ~0.7;
the additional `a0`-`a7` features on yet another scale ~1.45, with
`a5`/`a6`/`a7` showing a bimodal sentinel-value pattern). Rather than
feeding all 112 raw, heterogeneously-scaled features directly into an
LSTM's input-to-hidden weight matrix, each feature *type* gets its own
small MLP encoder first — concatenating `i0` and `i1`'s columns of that
type together, so the encoder can also learn cross-instrument structure for
that type before anything else mixes it in.

## Architecture (final, v3 — see revision history in `model.py`)

```
i0+i1 price   (44) -> Linear(44,16) -> ReLU              = 16
i0+i1 volume  (44) -> Linear(44,16) -> ReLU              = 16
i0+i1 trade   (16) -> Linear(16,8)  -> ReLU              = 8   (dp+dv, both instruments, merged)
additional a0-a7 (8) -> untouched, passed through as-is  = 8
                                                  total = 48

-> 2-layer LSTM (hidden_size=128, same shape as experiments 002/003)
-> Linear(128,64) -> ReLU -> Linear(64,2)
```

## The latency journey (why this took three iterations)

Learned from experiment 002's thin real-platform margin (80.7% of budget)
to benchmark latency *before* committing to any training run — this
experiment is the clearest example yet of why that discipline matters.

**v1** (2-layer encoders per group — price/volume: 44→24→16; separate
8→4 encoders for trade-price and trade-volume; GELU activations):
measured **123.6% of budget** (synthetic, uncorrected) — *worse* than
experiment 002, despite the LSTM's input shrinking from 112 to 48. The
smaller LSTM input didn't pay for itself: the multi-branch slicing/
concatenation/activation structure generates a lot of small ONNX graph ops
(20 `Reshape`s, 7 `Concat`s, 3 `Split`s per step), and each carries real
per-op dispatch overhead in ONNX Runtime regardless of its tiny individual
FLOP count — the same "many small ops" tax experiment 001's attention
mechanism ran into.

**v2** (swap `GELU` → `ReLU`): profiling v1 showed GELU's 7 activations
decomposing into `Mul`/`Div`/`Erf`/`Add` primitives (no native fused `Gelu`
op until ONNX opset 20; this exports at opset 18) accounted for ~31% of
profiled per-step time. The "obvious" fix gave almost nothing: **122.6%**,
barely different from v1. Lesson: the profiler's per-node overhead numbers
don't translate linearly to real wall-clock savings — the actual bottleneck
was node *count*, not which specific op each node was.

**v3** (this version — remove the encoders' hidden layer, single
`Linear+ReLU` per group instead of two; merge the two separate trade
encoders into one): **92.1% of budget** (synthetic, uncorrected) — a real,
substantial fix, confirming node count (fewer branches, fewer ops) was
indeed the lever that mattered, not activation choice. With any of the
correction factors observed across prior submissions (1.3x-1.8x), this
lands at 51-71% of budget — safer than experiment 002's actual real-world
margin (80.7%) even in the conservative case.

All three versions verified numerically correct against `model.forward()`
(diffs ~1.5e-8) before being benchmarked or discarded.

## Training status

`solution.py` smoke-tested: correct output shape/reset/None-handling,
deterministic. Training was launched on the full `train.parquet` (5 epochs,
batch size 4, ~2.85-3s/batch, estimated ~12-12.5h total) — but **experiment
003's training job was still running at the time** (28% into its first of
5 epochs), and running both simultaneously pushed this machine's load
average from ~8 to ~12.7 on a 16-core shared box other students are
actively using. Paused experiment 004's training (killed early, negligible
lost work - only `checkpoints/config.json` had been written) rather than
let two of our own training jobs contend for CPU indefinitely.

**Update**: rather than wait for experiment 003 to finish entirely,
`train.py` was given an explicit CPU thread cap (`--num-threads`, default
4, via `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`NUMEXPR_NUM_THREADS` + `torch.set_num_threads()` - PyTorch's CPU backend
otherwise grabs as many cores as it can find). Relaunched alongside
experiment 003 (left running uncapped - restarting it just to add caps
would have wasted ~45 minutes of real progress). Settled at ~323% CPU
(~3.2 of its 4-core cap); overall machine load returned to ~8.7, close to
the original single-job baseline, instead of the ~12.7 seen running both
uncapped. Both jobs now proceed concurrently without meaningfully
contending for CPU.

## Files

Same structure as experiments 002/003: `model.py`, `dataset.py`, `train.py`,
`export_onnx.py`, `solution.py`. No `inference.py` needed.

## Status / next steps

- [x] Architecture designed, iterated on latency 3x, settled on a safe config
- [x] Export pipeline (dynamo + onnxsim) verified correct at each iteration
- [x] `solution.py` smoke-tested: correct shapes, deterministic
- [x] Trained on full `train.parquet`: 5 epochs, ran concurrently with
      experiment 003 via CPU thread-capping (`--num-threads 4`), best
      `valid_loss=0.3576`
- [x] Exported to ONNX on trained weights, re-verified correct, re-benchmarked
      latency: **0.0865 ms/step, 89.5% of budget uncorrected** (vs. the
      92.1% pre-training estimate — margin held, even improved slightly)
- [x] Scored via `solution.py --validation ...`: **WP = 0.633608**
      (t0=0.632589, t1=0.634626, 1872/1873 eligible blocks). Took 1h06m49s
      wall-clock locally.
- [x] Packaged and verified: `../../submissions/004_grouped_encoder_lstm_2026-09-13.zip`
      — built once by hand on request, then rebuilt and re-verified
      automatically by `auto_pipeline.sh` (`ZIP VERIFICATION PASSED` both
      times).
- [x] Compared: beats the baseline (0.589595, +0.044/+7.5%) and experiment
      001 (0.622269, +0.0113/+1.8%); essentially **ties** experiment 002
      (0.634285) and experiment 003 (0.634654) despite having the fewest
      parameters of any experiment (233,194 — less than half of experiment
      002's 520,450) and the smallest effective LSTM input (48-dim, vs.
      112-dim raw features for experiments 002/003).
- [x] Submitted to the real platform (2026-09-13 23:31, ID `00901FPP`):
      **platform WP = 0.6089**, runtime **61m32s**, 627MB. Beats baseline
      and experiment 001 on the real test set; ties experiments 002/003
      (within ~0.002).

## Critical finding: runtime exceeded the nominal 60-minute budget

**61m32s = 3,692 seconds - over the stated 3,600-second limit** - yet the
submission still passed CHECK/SCORE and was scored normally. This is the
first submission of ours where that's happened.

Reconciling against our own benchmark: real per-step latency was
**0.0990 ms/step**, vs. our synthetic (uncorrected, single-threaded local
benchmark) of **0.0865 ms/step** - a **0.87x factor**, meaning **the
platform ran slower than our local machine** for this model. Every prior
submission (baseline, 001, 002, 003) had the reverse: the platform ran
1.5-1.9x *faster* than our local synthetic benchmark. This is a direct
counterexample to that pattern, and to the safety heuristic built on it
("if the uncorrected local estimate is under 100% of budget, it's safe").

**Best hypothesis** (unconfirmed): this experiment's multi-branch encoder
structure - separate slicing/concatenation/small-MLP branches per feature
type, which needed 3 rounds of latency tuning to even get *locally* under
budget (see the "latency journey" section above) - generates many small
graph ops relative to its actual compute. If the platform's general speed
advantage comes from something that specifically helps compute-bound work
(e.g. better vectorized-kernel throughput) rather than per-op dispatch
overhead, a dispatch-heavy graph like this one might not get that benefit,
or could even be disadvantaged by whatever's different about the platform's
environment (container/virtualization overhead compounding with a high
op count, for instance).

**Practical implication for future experiments**: don't trust the
platform-speedup assumption for architectures with many small graph
operations (branching, multi-encoder, attention-like structures). For
those specifically, target a much larger safety margin on the *uncorrected*
synthetic estimate (e.g. comfortably under 70% of budget) rather than
treating "under 100% uncorrected" as sufficient - that bar was met here
(89.5%) and the submission still nearly timed out. Simple, sequential
architectures (plain stacked LSTM, as in experiments 002/003) have not
shown this failure mode so far.

## Key finding: matches larger models with under half the parameters

Combined with experiment 003's finding (MLP head substitutes for LSTM
depth), this experiment adds a second data point: **the grouped
feature-type encoders let a much smaller LSTM (48-dim input instead of
112-dim raw features) match models 2x+ its size.** Three of our four
trained approaches (002, 003, 004) now cluster within ~0.001 WP of each
other despite parameter counts ranging from 233K to 520K — suggesting
this task's ceiling (with this training setup, data, and loss) may be
close to saturated around ~0.633-0.635 local WP, and further gains might
need a genuinely different lever (more/better features, a different loss,
more training data or epochs) rather than more parameters in this same
architectural family.

## Metric correction (2026-09-16)

The organizers found a mistake in their WP calculation and re-evaluated
every submission under a corrected "Global Weighted Pearson correlation"
metric (pooled across the whole dataset, not averaged per sequence — see
`PROBLEM_STATEMENT.md` §3 and `EXPERIMENTS.md`). This experiment's real
platform score changed: **~~0.6089~~ → 0.6508** — and under the corrected
metric, this is now the **best score of any submission** (narrowly ahead
of experiment 002 at 0.6507), a striking reversal from the old metric
where it was our lowest-scoring post-baseline submission. Runtime/memory
are unaffected. The local WP figure above (0.633608) was computed with
the old, now-stale local scorer.

## Extended re-training (2026-09-16/17) — a real improvement

`train.py` in this directory was updated in place (epochs 5 -> 15,
`ReduceLROnPlateau` LR decay + early stopping, GPU), run concurrently
with experiments 002 and 006 on the one available GPU. Original 5-epoch
checkpoints preserved in `checkpoints_5epoch_backup/`. See
`EXPERIMENTS.md`'s "In-progress: extended re-training" section for the
full rationale and setup.

**Result: this one paid off.** Ran the full 15 epochs (early stopping,
patience 5, never triggered — best epoch was 12, only 2 stale epochs by
the end), LR decayed once from 1e-3 to 5e-4 at epoch 11, total training
time 9h46m48s (slower than a solo GPU run would be, due to 3-way GPU
contention with 002/006 running at the same time).

| | Original (5 epochs) | Extended (15 epochs, LR decay) |
|---|---:|---:|
| Local WP | 0.677315 | **0.680497** (+0.0032) |
| best valid_loss | 0.3576 | 0.3441 |

Re-exported and re-verified correct (max abs diff 7.153e-07),
re-benchmarked latency on the new trained weights: **93.8% of budget
uncorrected** — still clears the safety bar, consistent with the
original run's margin. Packaged and verified:
`../../submissions/004_grouped_encoder_lstm_extended_2026-09-16.zip`.
**Submitted 2026-09-17 09:01 (ID `1D02W6EF`): platform WP = 0.6531 —
NEW BEST PLATFORM SCORE OVERALL**, beating this experiment's own
original 5-epoch result (0.6508, +0.0023) and the previous best (002's
0.6507). Runtime **37m3s (61.7% of budget)**, 627MB — comfortable
margin, consistent with the safe re-benchmarked latency (93.8%
uncorrected). The local WP gain from extended training (+0.0032)
translated directly into a real platform gain, confirming this was a
genuine improvement rather than overfitting to the local validation set.

## Pushed further: 30 epochs with warmup (2026-09-17/18, complete)

004 was the only one of the three extended runs (002/004/006) that
hadn't shown clear valid-loss degradation by the end of its 15-epoch
budget — best epoch was 12, only 2 stale epochs by 14, vs. 002/006 which
both climbed visibly after their second LR decay. That made it the best
candidate to see whether more training helps further. `train.py` updated
in place again: `--epochs` 15 → 30, plus a new linear LR warmup phase
(`--warmup-epochs`, default 3, ramps from 0.1×lr up to the full 1e-3
before the existing `ReduceLROnPlateau` decay/early-stopping logic starts
tracking). 15-epoch checkpoints preserved in `checkpoints_15epoch_backup/`.

Trained solo on GPU this time (002/006 already finished, no 3-way
contention) — ran at ~2.5-2.6 batch/s vs. the shared run's ~1.2.
Early-stopped at epoch 19 (best valid_loss=0.3434 at epoch 14), ~9.8h
wall-clock. `auto_pipeline_30ep.sh` exported (verified, 5.364e-07 max
diff), re-benchmarked latency (**95.6% of budget uncorrected** — still
clears the safety bar), plotted the loss curve (`loss_curve_30ep.png`),
and ran WP scoring **in parallel** with zip packaging+verification
(rather than sequentially) — packaging only needed the exported ONNX
files, not the score.

**Result: local WP = 0.682425** (t0=0.677896, t1=0.686955) — up from the
15-epoch result's 0.680497 (+0.0019), and the current best local WP of
any experiment in this project. Packaged/verified:
`submissions/004_grouped_encoder_lstm_30ep_2026-09-17.zip`.

**SUBMITTED 2026-09-17 23:02 (ID 0K1FIBZU): platform WP = 0.6530**,
runtime 38m54s (64.8% of budget), 627MB — comfortable margin, consistent
with the 95.6%-uncorrected latency estimate. This is essentially flat
against the 15-epoch submission's platform score of 0.6531 (−0.0001).
That's notable: the 15→30 epoch step's local gain (+0.0019) did **not**
translate into a platform gain, unlike the 5→15 epoch step, where a
local gain of +0.0032 became a real +0.0023 on the platform. Read
together, this suggests the local valid-set improvement from 15→30
epochs is close to the noise floor (or mildly overfit to the local valid
split) rather than a generalizable improvement — pushing this particular
architecture further with more epochs looks like it's plateaued.
