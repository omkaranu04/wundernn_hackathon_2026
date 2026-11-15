# 029: Weight sharpness, a future-seeing teacher, and a wider lean model

Follow-ups 4-6 requested 2026-10-11, all on experiment 028's lean model with
the recipe that worked (scored-probability row weights in both loss terms,
EMA of the weights with decay 0.9998, standard 0.5 MSE + 0.5 per-sequence
Pearson loss, Adam 1e-3, 3-epoch warmup, plateau decay, WP selection on the
true valid mask). **Reference: 028 arm A = raw 0.685089 / EMA 0.686746.**
Mask labels are the original v1 file, so results compare directly with 028.

| arm | change vs 028-A | question | status |
|---|---|---|---|
| `teacher_bi` | `--bidirectional`: the LSTM also sees the FUTURE (625,342 params) | How far above 0.687 does hindsight get? = the ceiling for distillation. Cannot be shipped. | launched 2026-10-11 (server A) |
| `power2` | `--weight-power 2`: row weight = p^2 | Do sharper weights (more focus on confident rows) help? | launched 2026-10-11 (server A) |
| `floor10` | `--weight-floor 0.1`: row weight = 0.1 + 0.9p | Does keeping some weight on unscored rows help? | started 2026-10-11 13:59 on server A (taken from the queue) |
| `h160` | `--hidden-size 160` (371,390 params) | Does spending the lean model's latency headroom on width help? | started 2026-10-11 15:10 on server B, then MOVED 15:16 to the local 1080 Ti (user allowed one job there; 1.8 s/batch local vs 2.5 on server B's shared GPU; server B copy killed after ~7 min). Local run: torch 2.14 (remote runs use 2.2), weights in `results/checkpoints_h160/`, log `logs/train_h160_local.log` |
| `A_v2labels` | identical to 028-A but with the v2 mask labels (`025.../mask_out_v2_h96/`, held-out AUC 0.959 / AP 0.756 vs v1 0.952 / 0.728) | Do sharper scored-probabilities help? | started 2026-10-11 15:02 on server B (GPU 0) |
| `D_aux_trainvalid` | 028-B's recipe (weights + auxiliary head + EMA) on train+valid, fixed 14-epoch schedule (as 028-C) | The final-candidate combination of the two best 028 results (B 0.6583, C 0.6580). No local score possible; platform only. Conceptually 028 arm D; run from this directory because the queue script lives here. | started 2026-10-11 15:08 on server B (GPU 0) |

Checks before launch (random-init): the default model is strictly causal
(altering rows >= 60 changes earlier predictions by exactly 0) while the
teacher is not (7.7e-03); `export_onnx.py` refuses bidirectional checkpoints;
the hidden-160 model exports correctly (1.5e-08) and benchmarks at exactly
012's latency (ratio 1.000, alternating run), i.e. it uses up the lean
model's ~11% headroom and no more. All four configurations smoke-tested.

`queue_job.sh NAME MIN_FREE_MIB -- <args>` waits for free GPU memory and
launches; the same entry runs on both servers (shared home) and an atomic
mkdir lock in `locks/` makes each job start exactly once.

## Results

(pending)

## Interim readings (2026-10-11 18:35, all arms still training)

- **`teacher_bi` reaches valid WP 0.983 raw / 0.987 EMA by epoch 3** (vs
  ~0.687 for causal models). A model that sees the future does not predict the
  targets, it reads them off: the targets are almost a deterministic function
  of FUTURE input features. So this is not a usable "ceiling", and distilling
  from this teacher would be nearly the same as training on the labels.
  A teacher would need a restricted look-ahead to be informative; the more
  useful follow-up is to find WHICH future quantity the targets are (horizon,
  features) and use intermediate horizons as auxiliary targets.
- EMA valid WP at equal epochs vs 028-A (0.6793 @3, 0.6838 @5, 0.6847 @6):
  `power2` 0.6744 @3, 0.6808 @5, 0.6819 @6 (behind at every epoch, -0.002 to
  -0.005); `floor10` 0.6813 @3, 0.6839 @5, 0.6805 @6 (even, noisier EMA);
  `A_v2labels` 0.6775 @3; `h160` 0.6764 @1 (028-A: 0.6739). Too early to call
  anything except that sharper weights are not helping so far.
- Speeds on the shared GPUs: ~40 min/epoch (power2, floor10), ~50 (A_v2labels),
  ~44 (D, 14 epochs, expected done ~01:30 on 2026-10-12), ~95 (h160, local).

## Results (2026-10-12 01:15) - power2, floor10, D finished

Reference 028-A: raw 0.685089 / EMA 0.686746 (028-B: 0.684377 / 0.687037).

| arm | best raw WP (epoch) | best EMA WP (epoch) | vs 028-A EMA | stopped |
|---|---|---|---|---|
| `power2` (weights p^2) | 0.683174 (10) | 0.684285 (11) | **-0.0025** | early stop, epoch 16 |
| `floor10` (weights 0.1+0.9p) | **0.686718** (10) | **0.687892** (13) | **+0.0011** | early stop, epoch 18 |
| `D_aux_trainvalid` | no local score possible | - | - | fixed 14 epochs, done 00:18 |
| `A_v2labels` (still running, epoch 16) | 0.685118 (12) | 0.687082 (12) | +0.0003 so far | LR already at 2.5e-4 |
| `h160` (still running, epoch 6, local) | 0.682271 (5) | 0.684227 (5) | 028-A at epoch 5: 0.683759 | - |
| `teacher_bi` (still running, epoch 15) | 0.996737 | 0.997071 | not comparable | - |

Readings:
- **Sharper weights hurt** (-0.0025 EMA, -0.0019 raw, and behind 028-A at
  every epoch). Closed.
- **Softer weights are the best local score so far** (0.687892 EMA; raw
  0.686718 is also the best raw score of any run), but +0.0011 over 028-A is
  inside single-seed noise. Direction is consistent with power2: moving weight
  TOWARD unscored rows helps a little, away from them hurts. Worth a platform
  test; a floor of 0.2-0.3 is the natural next probe.
- **Sharper mask labels make no visible difference** (+0.0003 so far), like
  the auxiliary head. The label quality is not the bottleneck.
- `teacher_bi` climbed to 0.997: the targets are essentially a deterministic
  function of future inputs (see interim readings above).

### Submission zips (2026-10-12)

| zip | weights | export vs batched forward | local WP |
|---|---|---|---|
| `029_D_aux_trainvalid_ema_2026-10-12.zip` | D `ema.pt` (final, epoch 13) | 5.36e-07 | none possible (valid is in its training set) |
| `029_floor10_ema_2026-10-12.zip` | floor10 `best_ema.pt` (epoch 13) | 5.96e-07 | 0.687892 (batched) = **0.687892 real step-by-step scorer** on the extracted zip (t0 0.684014, t1 0.691771; `logs/real_score_floor10_ema.log`) |

Both 1.85 MB, 46 nodes, outputs `[prediction, next_h, next_c]` - the same
graph as 028's zips (expect ~41 min on the platform), fresh-extraction checks
passed, different weight files. Weights and logs for power2, floor10 and D
are in `results/checkpoints_*` and `logs/` (copied from the shared server
home, 16 checksums verified, deleted there).

### Platform results (submitted 2026-10-12) - new best, both

| zip | platform | runtime | local WP (batched) |
|---|---|---|---|
| D-EMA (028-B's recipe on train+valid) | **0.6596** (ID 2287KP72) - best so far | 41m51s, 627 MB | not measurable |
| floor10-EMA (row weights 0.1+0.9p) | **0.6588** (ID 2A3MHMA1) | 41m36s, 627 MB | 0.687892 |

Previous best: 028-B 0.6583, 028-C 0.6580, 028-A 0.6574.
- **Softer weights: +0.0014 on the platform** (floor10 vs 028-A, the same run
  except the floor), matching the local +0.0011. Each alone is inside noise;
  agreeing in size and sign they are credible, like the original weighting.
- **Train+valid: +0.0013** (D vs 028-B), after +0.0006 for C vs A. Two
  independent pairs, same sign: the extra 18% of sequences (with their TRUE
  scored mask) is a small real gain.
- The two changes are independent, so the obvious next step is to stack them
  and to find where the floor stops helping: experiment 030.

### Final state of the remaining arms (2026-10-12)

- `A_v2labels` finished (early stop, epoch 17): raw 0.685118 / EMA 0.687082,
  +0.0003 over 028-A. Sharper labels do not help; v1 labels stay the default.
- `teacher_bi` stopped by hand at epoch 18 of 30 (EMA 0.9971, flat): see the
  interim readings. Weights kept in `results/checkpoints_teacher_bi/`.
- `h160` still training on the local 1080 Ti (EMA 0.684227 at epoch 5).
- Everything is copied back (28 checksums verified) and deleted from the
  shared server home.
- `h160` at epoch 12 (2026-10-12 10:00): best EMA 0.687112 (epoch 10) vs 028-A's 0.686541 at the same epoch and 0.686746 best: +0.0004, noise, at ~12% more latency. Not a lead.
- **`h160` finished 2026-10-12** (early stop at epoch 15, local 1080 Ti): raw 0.685501 / EMA 0.687112, +0.0004 over 028-A at ~12% more latency. Width is not a lead. Weights in `results/checkpoints_h160/`.
