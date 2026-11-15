# 019: Loss / optimizer recipe on 004's architecture

Training-recipe experiment. **Architecture is 004's grouped-encoder LSTM,
unchanged** (`model.py`, `export_onnx.py`, `solution.py` are byte-copies of
004's: 233,194 params, 43 ONNX nodes, 95.6% of budget uncorrected, so any
winning checkpoint is directly submittable with no latency risk).
Baseline to beat: 004's 30-epoch run - valid_loss 0.3434, local WP 0.682425
(t0=0.677896, t1=0.686955), platform 0.6530.

## Motivation

1. Every experiment through 018 used Adam (lr 1e-3, no gradient clipping)
   and a flat 0.5 MSE + 0.5 per-sequence Pearson loss.
2. The per-sequence Pearson term matches the *old* metric's aggregation
   (average of per-sequence scores). The corrected 2026-09-15/16 metric
   pools all rows into one global correlation per target. See the caveat
   in `shared/losses.py::weighted_pearson_loss`. Effect on global WP never
   measured.
3. An idea from a different tabular, correlation-scored competition: SGD
   (not Adam/AdamW) generalized better, and a 0.6 MSE / 0.4 Pearson blend
   worked best. Transfer to LSTMs on 20,000-step sequences is unproven.

## What's new

- `eval_wp.py`: batched GPU implementation of the corrected global WP
  (pooled float64 moments over `need_prediction & is_scored` rows).
  **Validated 2026-10-07 against the real scorer: on 004's checkpoint it
  gives WP 0.682425 / t0 0.677896 / t1 0.686955 - identical to the
  step-by-step scorer's published numbers to 6 decimals - in 3m25s versus
  ~80 minutes.**
- `train.py` flags: `--pearson-scope {sequence,batch}` (batch = pool all
  scored rows in the batch into one correlation per target),
  `--mse-weight`, `--optimizer {adam,sgd}` (+ momentum/nesterov/
  weight-decay), `--clip-norm`, `--select-metric {wp,loss}`. With
  `--select-metric wp` (default) per-epoch WP drives best-checkpoint
  selection, `ReduceLROnPlateau` and early stopping. Pre-clip gradient
  norm is logged every epoch as `gnorm` (useful for choosing SGD's LR).
- `dataset.py` additionally returns `is_scored`.

## Run plan (on server A; one dir per arm via `--checkpoint-dir`)

| Arm | Optimizer | Pearson scope | MSE weight | Select | Status |
|---|---|---|---|---|---|
| control | Adam 1e-3 | sequence | 0.5 | wp | **done**: best valid_wp 0.681582 (epoch 12, early-stopped epoch 17) |
| pooled50 | Adam 1e-3 | batch | 0.5 | wp | **done**: best 0.679449 (epoch 13, stopped epoch 18) |
| pooled60 | Adam 1e-3 | batch | 0.6 | wp | **done**: best 0.680039 (epoch 11, stopped epoch 16) |
| SGD trials | SGD nesterov 0.9, clip 1.0, lr in {0.03, 0.1, 0.3} | sequence (Stage 1 winner) | 0.5 | wp | **done** 2026-10-08 (5 epochs each): best 0.6698 / 0.6697 / 0.6660 for lr 0.1 / 0.03 / 0.3 - no better than Adam |
| SGD full | best SGD lr | | | wp | NOT run (5-epoch trials showed no advantage) |

The *control* reruns 004's recipe with WP-based selection (not valid_loss)
so differences between arms are attributable to the loss/optimizer, not to
the changed checkpoint-selection rule. It also doubles as a seed/nondeterminism
datapoint versus 004's published 0.682425.

## Reading results

Local-to-platform WP ranking has flipped by ~0.002 several times (004
15->30 epochs, 010, 017). Treat a gain as real only if it exceeds ~0.002
WP, or if the loss and optimizer changes point the same way. Batched
`valid_wp` is the working metric; confirm any winner with the real
`solution.py --validation` scorer (~80 min) before packaging.

## Results

### Stage 1 (Adam; loss only) - finished 2026-10-08

Best batched `valid_wp` (corrected global WP, eval_wp.py), single seed each:

| Arm | best valid_wp | at epoch | early-stopped |
|---|---|---|---|
| control (per-sequence Pearson, 0.5/0.5) | **0.681582** | 12 | 17 |
| pooled50 (batch-pooled Pearson, 0.5/0.5) | 0.679449 | 13 | 18 |
| pooled60 (batch-pooled Pearson, 0.6/0.4) | 0.680039 | 11 | 16 |

**Pooling the Pearson term did not help** - both pooled arms finished
0.0021 / 0.0016 *below* the control. For noise scale: the control (004's
exact recipe, only WP-based selection added) landed 0.0008 below 004's
published 0.682425, and epoch-to-epoch valid WP wobbles by ~0.002-0.003
(control epoch 4 dipped to 0.6186 before recovering), so selecting the
best epoch on valid WP is itself mildly optimistic. Read: no evidence the
per-sequence/global aggregation mismatch costs us anything; if anything
the per-sequence form is a slightly better training signal here. The
0.6/0.4 blend did not beat 0.5/0.5 within the pooled variants either
(+0.0006, inside noise).

Weights/logs for all three arms: `stage1/checkpoints_*` and
`logs/train_*.log` (copied back from server A, checksum-verified, then deleted
there). Not exported or scored with the real scorer since none beat the
baseline.

### Speculative platform submission zip: pooled60 (2026-10-08)

Built at the user's request purely to see how the platform ranks a
pooled-loss model (it does NOT beat the baseline locally). Exported from
`stage1/checkpoints_pooled60/best.pt` with 004's unchanged export path:
max diff vs. batched forward 4.768e-07, 43 nodes, **97.3% of budget
uncorrected** (004: 95.6%). Zip: `submissions/019_loss_optimizer_recipe_pooled60_2026-10-08.zip`
(1.7 MB, fresh-extraction check passed). Only the batched `valid_wp`
(0.680039) exists for it - it has not been run through the real
step-by-step scorer. **Platform result: 0.6534** (ID 0F941MW4), runtime **37m27s** (62.4% of
budget, implied speedup 1.56x, 627 MB). That is a tie with 004's platform
scores (0.6530 for 30 epochs, 0.6531 for 15 epochs), +0.0003/+0.0004 and
well inside noise, even though pooled60 scored 0.0024 *lower* than 004's
30-epoch run on local valid (0.680039 batched vs. 0.682425) - another
local-vs-platform ranking gap, this time in the model's favour. Read: the
loss variants move local valid WP by ~0.002 but platform WP not at all;
004-architecture platform scores sit at 0.6530-0.6534 regardless of
epochs (15/30) or loss (per-sequence/pooled).

### Stage 2 (SGD LR trials) - finished 2026-10-08

5 epochs each, SGD Nesterov 0.9, clip-norm 1.0, per-sequence loss 0.5/0.5,
3-epoch warmup. Best batched valid_wp (single seed):

| lr | valid_wp by epoch 0..4 | best |
|---|---|---|
| 0.03 | 0.643 0.661 0.667 0.666 0.670 | 0.669684 |
| 0.1 | 0.652 0.667 0.668 0.659 0.670 | **0.669809** |
| 0.3 | 0.650 0.663 0.663 0.659 0.666 | 0.665984 |

Adam control over the same epochs: 0.649 0.665 0.670 0.672 0.619 (best
0.6717). SGD is not faster or better at 5 epochs; lr 0.3 is clearly too
high, 0.03 and 0.1 are equivalent. A short constant-LR trial cannot rule
out a gain after 30 epochs with decay, but nothing here suggests one, so
the full SGD run was not started. Weights/logs: `stage2/`, `logs/`
(copied from server A, checksum-verified, deleted there).
