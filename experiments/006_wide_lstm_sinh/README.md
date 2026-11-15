# 006 — MLP encoder + single wide LSTM layer + sinh-activated MLP head

**Status: trained, locally scored, and submitted (2026-09-15 23:58, ID
018BCF3P). Not a new best locally, but see the metric-correction note at
the bottom — the picture changed after the platform's WP calculation was
corrected.**

## Idea

A genuinely different lever from experiments 002/003/004/005, which all
clustered within ~0.002 WP of each other (0.6089-0.6108 on the platform)
regardless of stacked LSTM depth (2-4 layers) or per-feature-type input
grouping — suggesting that family of architectures may be near a local
ceiling. This experiment changes two things at once, each independently
motivated by the full-dataset analysis in `analysis/`:

1. **Depth -> width.** A small MLP encoder (`Linear(112,64) -> GELU`)
   feeds a *single*, wide `nn.LSTM` layer (`hidden_size=224`) instead of
   stacking 2-4 narrower layers at `hidden_size=128`. Untested shape for
   this project so far.
2. **Raw-scale head -> `sinh`-activated head.** `analysis/results/
   feature_statistics.md` shows both targets are signed, ~std-1 but
   extremely fat-tailed (`t0`: min -125.6, max 74.5, excess kurtosis
   22.4; `t1`: min -180, max 77.3, excess kurtosis 90.1) with ~22% exact
   zeros (the zero-spike finding in `analysis/README.md`). Plain `log`
   doesn't work on signed/zero data. `arcsinh` does: it's ~identity near
   0 (so the bulk of the distribution, where std≈1, is barely touched)
   and ~`log(2x)` for large `|x|` (so it compresses the rare huge-outlier
   rows that otherwise dominate a squared-error loss). Rather than
   transforming targets before the loss (which would require re-deriving
   the WP metric's `|target|`-weighting in transformed space), the head's
   final layer is followed by `sinh` (arcsinh's inverse) — the network's
   pre-activation output lives in an arcsinh-like space internally, but
   `model.forward()` and the exported ONNX graph both emit raw-unit
   predictions directly. `shared/losses.py`'s existing loss functions are
   used completely unchanged.

## Architecture

```
features (T, 112) -> Linear(112, 64) -> GELU                      [encoder]
                   -> nn.LSTM(input=64, hidden_size=224, layers=1) [wide, shallow]
                   -> Linear(224,128) -> GELU -> Dropout
                   -> Linear(128,64)  -> GELU -> Dropout
                   -> Linear(64,2)
                   -> clamp(-8, 8) -> sinh                         [raw-unit output]
```

304,258 params — between experiment 003 (280,898) and experiment 005
(412,994).

The `clamp(-8, 8)` before `sinh` is a numerical safety net only
(`sinh(8) ≈ 1490`, ~8x past the observed `|target|` max of ~180) for
early training with random/large weights; it isn't expected to bind once
the model has learned reasonable output scale.

## Export note: `dynamo=True` is required here, not optional

Experiments 002-005 called `torch.onnx.export(...)` without an explicit
`dynamo` argument and got the (safe, correct) dynamo-based exporter
because that's this environment's default (`torch.onnx.export`'s
`dynamo` parameter defaults to `True` in the `wundernn` conda env's torch
version). This experiment passes `dynamo=True` explicitly in
`export_onnx.py` because it's the first model in this project using an op
(`aten::sinh`) that the **legacy** TorchScript-based exporter doesn't
support at all (confirmed by accidentally running the export once in a
different, older-torch conda env, where the default is `dynamo=False` and
export hard-failed with `UnsupportedOperatorError: aten::sinh`). Being
explicit removes any dependence on which environment's default happens to
be active.

## Pre-training checks (random weights, before committing to training)

- **Correctness**: `export_onnx.py`'s `verify()` — ONNX step-by-step
  output vs. `model.forward()`'s batched output over 800 random steps —
  **max abs diff = 2.98e-08**. Clean pass.
- **Latency**: single-threaded ONNX Runtime benchmark (same session
  options as `solution.py`: `intra_op_num_threads=1`, sequential
  execution, spinning disabled), 20,000 steps after warmup:
  **0.0868 ms/step, 89.8% of the 0.0966 ms/step budget, uncorrected.**
  This clears the project's "ideally <90-95% uncorrected" safety bar —
  comparable to experiment 003's 91.3% (which ended up using only 61.4%
  of the real budget on the platform) and meaningfully safer than
  experiment 005's 120.9% gamble. No dependence on any platform
  correction factor needed.

## Training

Launched on `train.parquet`, 5 epochs, batch size 4, default args
(encoder_dim=64, hidden_size=224, num_layers=1, head_dims=[128,64],
clamp=8.0, loss=mse+pearson). ~1.5-1.7s/batch observed in the first few
batches (2,652 batches/epoch on the train split). Automated post-training
pipeline (`auto_pipeline.sh`, watching PID 562487) armed to export, score,
package, and verify immediately once training finishes.

## Files

Same structure as experiments 002/003/005: `model.py`, `dataset.py`
(unchanged, copied from 005), `train.py`, `export_onnx.py`, `solution.py`,
`auto_pipeline.sh`. No `inference.py` needed (streaming and batched
forward are naturally numerically identical for a plain recurrent stack,
same reasoning as 002/003/005).

## Status / next steps

- [x] Architecture designed: MLP encoder -> single wide LSTM -> MLP head
      with `sinh`-activated output
- [x] Latency-benchmarked before training: 89.8% of budget uncorrected —
      clears the safe bar, no platform-speedup dependence needed
- [x] Export pipeline (dynamo + onnxsim) verified correct (2.98e-08 max
      abs diff)
- [x] Trained on full `train.parquet`: 5 epochs, **7h29m55s**, best
      `valid_loss=0.3598`
- [x] Exported to ONNX on trained weights, re-verified correct (max abs
      diff 1.311e-06)
- [x] Scored via `solution.py --validation ...`: **WP = 0.633488**
      (t0=0.632033, t1=0.634943, 1872/1873 eligible blocks)
- [x] Packaged and verified: `../../submissions/006_wide_lstm_sinh_2026-09-14.zip`
      (`ZIP VERIFICATION PASSED`)
- [x] Compared against experiments 000-005 — see "Result" below
- [ ] Submitted to the real platform — **not yet decided, see "Result"**

## Result: the wide-single-layer + sinh combination did not beat the stacked-LSTM family

| # | Approach | Local WP |
|---|---|---:|
| 002 | Stacked LSTM, 4 layers | 0.634285 |
| 003 | Stacked LSTM, 2 layers + MLP head | 0.634654 |
| 004 | Per-feature-type encoders + LSTM | 0.633608 |
| 005 | Stacked LSTM, 3 layers + MLP head | **0.635169** (best local) |
| **006** | **Encoder + single wide LSTM + sinh head** | **0.633488** |

006 lands at the *bottom* of the post-baseline cluster — essentially tied
with experiment 004 (0.633608), about 0.0017 below experiment 005's best.
So neither half of this experiment's hypothesis was confirmed on its own:
going wide-and-shallow instead of deep-and-narrow didn't help, and the
`sinh`/arcsinh-space output reparameterization didn't help either (at
least not enough to show up net positive against a plain-Linear head with
similar capacity). This doesn't cleanly indict either idea individually —
both were changed at once, so this result can't say whether `sinh` alone
would have helped a stacked-LSTM backbone (that's exactly what experiment
007 is isolating right now) or whether the architecture shift alone (wide
single layer, no sinh) would have been closer to competitive.

Given it's not a new best and doesn't have a clean story for *why* it
would help beyond 005/002, the initial recommendation was not to submit
this one pending 007's result — it was submitted anyway (2026-09-15
23:58, ID 018BCF3P). See the metric-correction note below for the real
outcome.

## Metric correction (2026-09-16) and real platform result

The organizers found a mistake in their WP calculation and re-evaluated
every submission under a corrected "Global Weighted Pearson correlation"
metric (pooled across the whole dataset, not averaged per sequence — see
`PROBLEM_STATEMENT.md` §3 and `EXPERIMENTS.md`). This experiment's real
(corrected) platform score is **0.6502**, runtime **36m45s** (very
comfortable margin), 628MB. That's actually **mid-pack** under the
corrected metric — ahead of experiments 003 (0.6499), 007 (0.6495), and
005 (0.6473), though still behind 004 (0.6508) and 002 (0.6507). The old
local scorer's "bottom of the cluster" verdict (used for the
not-a-new-best conclusion above) turned out not to hold once scored on
the real, corrected metric — a useful reminder that the local ranking is
currently unreliable (see `EXPERIMENTS.md`'s metric-correction section)
and shouldn't be trusted to make submit/don't-submit calls until a
corrected local scorer exists.

## Extended re-training (2026-09-16/17) — the biggest gain of the three

`train.py` in this directory was updated in place (epochs 5 -> 15,
`ReduceLROnPlateau` LR decay + early stopping, GPU), run concurrently
with experiments 002 and 004 on the one available GPU. Original 5-epoch
checkpoints preserved in `checkpoints_5epoch_backup/`. See
`EXPERIMENTS.md`'s "extended re-training" section for the full rationale,
setup, and how this compares to 002/004's results.

Ran the full 15 epochs — early stopping (patience 5) never formally
triggered, but came close: best epoch was 10 (valid_loss=0.3498), then a
similar mild-overfitting climb as experiment 002 for the remaining
epochs despite two LR decays (1e-3→5e-4 at epoch 8, →2.5e-4 at epoch 13),
ending at 0.3703 by epoch 14. The saved `best.pt` correctly holds epoch
10's weights. Total training time 12h57m25s (3-way GPU contention).

| | Original (5 epochs) | Extended (15 epochs, LR decay) |
|---|---:|---:|
| Local WP | 0.675941 | **0.680431** (+0.0045 — biggest gain of 002/004/006) |
| best valid_loss | 0.3598 | 0.3498 |

Re-exported and re-verified correct (max abs diff 1.311e-06),
re-benchmarked latency: **88.5% of budget uncorrected** — still clears
the safety bar, consistent with the original run's comfortable margin
(no platform-speedup dependence needed, same as before). Packaged and
verified: `../../submissions/006_wide_lstm_sinh_extended_2026-09-16.zip`.
**Submitted 2026-09-17 12:59 (ID `1Y0A760H`): platform WP = 0.6521**,
up from the original 5-epoch result (0.6502, +0.0019) — **now the
2nd-best platform score overall**, behind only 004's extended result
(0.6531). Runtime **36m46s (61.3% of budget)** — comfortably safe,
consistent with the re-benchmarked 88.5% uncorrected margin. Unlike
experiment 002's extended resubmission (same day, timed out at 69m59s
despite an identical architecture to its own safe original), this one
never needed to gamble on any platform speedup factor to fit — a good
illustration of why the safe-latency-margin discipline matters even when
retraining, not just when first designing an architecture.
