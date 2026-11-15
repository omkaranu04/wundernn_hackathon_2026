# 032: architecture additions and structural diversity (push for the top 8)

Started 2026-11-02, after the leaderboard check: #1 0.67586, #8 0.67213, #20 0.66938; our best was the 031
look-ahead pair at 0.66610 (outside the top 20). Experiment 031 had driven the label side to a peak (look-ahead
window 12-25 rows, look-ahead weight 0.5, recipe A; see the 031 README, "2026-11-02 platform"). This experiment
tries the model side, with ideas we had never tested:

| idea | why | runs |
|---|---|---|
| GRU-core member (`--arch lean_gru`, `LeanGRUModel`) | different structures averaged gave the rival +0.002 (seeds of one structure +0.0003); our pairs only mix labels so far | `R12_LA12_GRU` (server A), `R12_LA25_GRU` (local 1080 Ti) |
| valid regime oversampled (`--extra-repeat 3`) | rival: train and valid are different regimes (AUC 0.993) and train data does not transfer to test | `R12_LA12_V3` (server A) |
| residual MLP stage after the LSTM (`leanx_r`) | rival: "a second nonlinear stage and depth made trainable by residual connections" helped | `R13_LA12_R` (server B) |
| instrument-1 price + volume back (`leanx_i`) | dropped in 027 when the model was far weaker; public lead-lag hint (r = 0.116) | `R13_LA12_I` (local) |
| position in the sequence (`leanx_p`) | `step_in_seq` is passed to predict(); never used | `R13_LA12_P` (server B) |
| all three (`leanx_irp`) | | `R13_LA12_IRP` (server B) |

All students: LA12 targets (0.5 x TV12 + 0.5 x la12 teacher), R3h128_A's recipe (floor 0.1, aux 0.2, mass, 24 epochs,
LR halved at 13/18/22, seed 0, width 128) - so each pairs with `031_R6_LA12_A` (0.66554) and the seed replicates
`R11_LA12_A2/A3` (031) that measure how lucky 0.66554 was.

## Code (shared with 031: the running jobs import it from there)

The code stays in `experiments/031_seed_ensemble_distillation/` because jobs on server A/server B (shared home) run from
that directory:
- `model.py`: `LeanGRUModel` (arch `lean_gru`), `LeanXModel` (arch `leanx_<i|r|p>`; streaming with `step0` equals
  the full pass to 1e-8; residual output layer zero-initialised).
- `train.py`: `--arch` choices, `--extra-repeat K`.
- `export_slim.py`: `build_multi` takes GRU members (`_gru_onnx`, PyTorch r,z,n -> ONNX z,r,h, `linear_before_reset=1`;
  `--multi SPEC...` exports a mixed LSTM + GRU ensemble, exact to 5e-7); `build_x` / `verify_x` for LeanX (exact to
  2e-8; a `pos` graph input when position is used).
- `solution_slim.py`: feeds `pos` from a precomputed table (step_in_seq / 20000, sin, cos) when the graph has it.
- `run_student.sh` (remote) / `run_local_student.sh` (local 1080 Ti) with env `ARCH` / `EXTRA_ARGS`.
- `numba_engine.py`: fused numba ensemble engine - exact, but slower than ONNX int8 on our CPU; parked (below).

Packaging (this folder, logs in `logs/`): `build_gru.sh`, `build_local_gru.sh`, `build_round12.sh`,
`build_round13.sh` -> `submissions/032_*.zip`.

## Runtime facts (same core, per step)

| graph | us/step | platform (est.) |
|---|---|---|
| plain width-128 student, int8 | 64-67 | 28 min |
| merged pair (block-diagonal), int8 | 95-98 | 41-42 min |
| merged triple, int8 | 136 | ~60 min - too slow |
| mixed pair: LSTM int8 + GRU fp32 (separate chains) | 115 | ~47-49 min |
| leanx_irp, int8, pos from a table | 84 | ~37 min |
| numba engine fp32, 1 / 2 members | 59 / 109 | slower than ONNX int8 for pairs: parked |

So with our tools the ensemble is capped at a pair; a GRU member costs ~15% more than an LSTM one.

## Status

- 2026-11-02 13:20: all eight runs training (R12 x3, R13 x4, plus 031's replicates). Expected: `R12_LA12_GRU` + mixed
  pair ~20:00, `R12_LA25_GRU` + mixed pair ~21:30-01:30 (local, three jobs share the GPU), R13 ~06:00-09:30 on 11-03,
  `R12_LA12_V3` ~05:30.

**2026-11-03 submission set (user: every zip should aim to beat the best, 0.66610):** the three diagnostic singles were
swapped for pairs (pairs reliably add ~+0.0006 over their members' mean; no single has beaten 0.66610):
`032_mixpair_LA12_A_LA25_GRU` (~49 min), `032_mixpair_LA25_A_LA12_GRU` (~50 min), `031_pair_R11_LA12_A2_R11_LA25_A2`
(the best pair's recipe with NEW seeds, ~42 min), `031_pair_LA12_A_R11_LA25_A2` (~41 min), and the one single with a
plausible jump, `032_R13_LA12_IRP` (~35 min). Diagnostics (R11_LA12_A2, GRU single, V3, R13 I/R/P) are later.

### 2026-11-03 platform results

| zip | platform | runtime |
|---|---|---|
| `032_mixpair_LA12_A_LA25_GRU` (LSTM LA12 + GRU LA25) | **0.66635 NEW BEST** (25NJMD2B) | 51m16s |
| `032_mixpair_LA25_A_LA12_GRU` (LSTM LA25 + GRU LA12) | 0.66596 (1CFT7C4V) | 49m52s |
| `032_R13_LA12_IRP` (i1 + residual + position, single) | 0.66558 (1Q7RSALK) | 33m6s |
| `031_pair_R11_LA12_A2_R11_LA25_A2` (best LSTM-pair recipe, new seeds) | 0.66543 (1DABLMAI) | 41m32s |
| `031_pair_LA12_A_R11_LA25_A2` | 0.66500 (2BCMT47S) | 42m47s |

- LSTM+LSTM pairs of the LA12/LA25 recipe: 0.66610 / 0.66543 / 0.66500 (mean 0.66551) - the 0.66610 was a good draw
  (pair seed spread ~0.001). Mixed LSTM+GRU pairs: 0.66635 / 0.66596 (mean 0.66616) -> **mixing structures is worth
  ~+0.0006 over same-structure pairs**, the same direction as the rival's +0.0017.
- LeanX IRP single 0.66558 = LA12_A 0.66554: the three additions together are neutral (as a single).
- Runtime: the mixed pairs ran 49m52s-51m16s (estimate 49-50) - legal but only ~9 min under the limit; the fp32 GRU
  chain is the slow part. A cheaper "different structure" partner (int8 LSTM-based LeanX on other labels) would cut
  that to ~44 min.

**Round 14 (2026-11-03 10:30):** replicates of the best mixed-pair recipe with NEW GRU seeds - `R14_LA25_GRU_s1`
(local) and `R14_LA25_GRU_s2` (server B), paired with LSTM LA12 members (`R6_LA12_A`, `R11_LA12_A2`, `R11_LA12_A3`) - and
`R14_LA25_IRP` (LeanX on the LA25 targets, server B). `export_slim.py --multi` now also takes LeanX members (i1 encoders,
shared `pos` input, residual stage; exact to 6e-7), but an LSTM + LeanX pair is NOT cheaper: 127.6 us/step (~54 min),
because every separate ONNX recurrent chain has a fixed cost - the GRU mixed pair (~50 min) stays the pair format.
`build_round14.sh` packages everything.

**Round 15 (2026-11-03 11:00): cheaper mixed pairs as insurance** (the private set is a different hidden dataset; a
~51-min pair has ~15% headroom if it is larger). LSTM LA12_A + GRU of width 96 (local, `R15_LA25_GRU96`) / 64 (server B,
`R15_LA25_GRU64`) on the LA25 targets; measured mixed-pair graphs: width 128 117.1 us (51m16s on the platform), 96 107.8 us
(~47 min), 64 99.8 us (~43.5 min). `build_round15.sh` packages both pairs.

**Round 16 (2026-11-03 20:30):** `R16_LA50_GRU` (GRU on the LA50 look-ahead labels, local) -> `032_mixpair_LA12_A_LA50_GRU`
(structure AND a more distant label window than LA25). `build_round16.sh`.

### 2026-11-03 evening: tick units (CPU analysis) and the joint pair (round 17)

- `analysis/scripts/tick_unit_probe.py` (`analysis/results/tick_unit_probe.txt`): every sequence's tick unit (0.1254 /
  0.2413 / 0.4685 / 0.6569, labelled from its targets) is readable from simple statistics of the FIRST 100 INPUT ROWS
  with **99.9% held-out accuracy** (gradient boosting, 2,000 train -> all 1,873 valid sequences; majority 37%).
  Predictions start at step 99, so the unit is known before the first prediction - and the networks can learn it.
- `analysis/scripts/per_unit_wp.py` (`per_unit_wp.txt`): honest S1_a100 (train-only) on valid scored rows - unit
  0.6569 carries **47.7% of the metric weight at WP 0.440**, vs 0.80 / 0.78 / 0.73 for the others (exactly the
  rival's numbers). Perfect predictions on that unit alone would give 0.907. That is where the remaining points are.
- `per_unit_calibration.py`: per-unit output scale/offset fitted on half of valid, scored on the other half (running).
- Round 17: `--arch joint_pair` (`JointPairModel`: lean LSTM `a` + lean GRU `b`, trained together; loss = 0.5 x
  L(mean, mean target) + 0.5 x mean of L(a, LA12 target), L(b, LA25 target)); `dataset.py` second soft-target file,
  `train.py --soft-targets-b / --extra-soft-targets-b / --joint-lambda`, `split_joint.py` (members exact), exports as
  a normal mixed pair. `R17_JOINT_L05` on server B (`run_joint.sh`, ~10:00 on 11-04), `build_round17.sh`.
- `per_unit_calibration.py` (`analysis/results/per_unit_calibration.txt`): per-unit output scale fitted on half of
  valid -> held-out **-0.000006**; scale + offset **+0.00015** with offsets that disagree between folds (noise). The
  model is already calibrated across units (the rival: per-unit handling -0.0002..+0.0001). The large unit is weak
  because it is intrinsically hard (coarse lattice: mostly 0 / one tick), not mis-scaled. Tick routing: not pursued.

### 2026-11-03 night: genuinely new ideas instead of more seeds (user's push)

- Batch-POOLED Pearson loss (train on the metric's global aggregation): already tested in experiment 019 (pooled50 /
  pooled60 were 0.0021 / 0.0016 BELOW the per-sequence control). Not repeated.
- **Distributional lattice head** (`--arch lean_dist`, `LeanDistModel`): per target, 81 logits over values on
  [-2, 2]; point prediction = the |y|-weighted mean sum p|c|c / (sum p|c| + 1e-3) - exactly the maximiser of a
  |y|-weighted Pearson. `--dist-ce W` adds a cross-entropy on the TRUE labels (HL-Gauss, sigma 0.05) next to the usual
  distillation loss on the point output (the distribution learns the real mostly-0/1-tick outcomes of the large tick
  unit: 47.7% of the weight at WP 0.44). Streaming exact; `export_slim.py build_dist` exact (5e-7), 20 nodes, ZIP OK,
  per-step cost ~ a plain student.
  - honest tests (S1_a100's recipe, train-only T10 targets, valid each epoch; baseline 0.689331): `H_DIST_CE05`
    (`--dist-ce 0.5`, server B) and `H_DIST_CE0` (head only, server A), `run_honest.sh`;
  - platform version `R18_LA12_DIST` (LA12 targets, R3 A recipe, `--dist-ce 0.5`, local), `build_round18.sh`.

### 2026-11-04 morning: the distributional head works in the honest test

| epoch | S1_a100 baseline (EMA) | `H_DIST_CE05` (head + CE on true labels) | `H_DIST_CE0` (head only) |
|---|---|---|---|
| 17 | 0.688992 | 0.689340 | 0.688823 |
| 19 | 0.689092 | 0.689987 | - |
| 21 | 0.689079 | **0.690328** (+0.0012, still rising) | - |
| final | 0.689331 | running (epoch 23/30) | running |

The gain comes from the cross-entropy on the TRUE labels (the distribution learns the real lattice outcomes), not from
the |y|-weighted output alone. +0.0012 is below the 0.002 local->platform reliability level but the largest new-idea
lead since look-ahead labels. Packaged overnight: `032_R18_LA12_DIST` (platform version, ~33 min), `032_R14_LA25_IRP`,
`032_mixpair_LA12_A_LA50_GRU` (~51 min). Started `R19_LA25_DIST` (local, LA25 targets) as a distributional pair member;
`build_round19.sh`.

### 2026-11-04 platform results

| zip | platform | runtime | read |
|---|---|---|---|
| `032_mixpair_LA12_A_LA50_GRU` (LSTM LA12 + GRU LA50) | **0.66638 NEW BEST** (25NEF50F) | 48m37s | ties 0.66635 |
| `032_mixpair_LA12_A_LA25_GRUs1` (best recipe, GRU seed 1) | 0.66575 (04Y0FT5E) | 47m31s | recipe mean 0.66605 over 2 GRU seeds |
| `032_R18_LA12_DIST` (distributional head + CE) | 0.66561 (0IKCDTQB) | 30m57s | LA12 0.66554: **+0.0001** |
| `032_mixpair_LA12_A_LA25_GRU96` (narrow GRU) | 0.66542 (0KDA9NK3) | 44m57s | -0.0006 vs the width-128 recipe mean |

- The distributional head's honest local gain (+0.0014 at epoch 23, H_DIST_CE05 0.690503) did NOT carry to the
  platform on top of look-ahead labels (+0.0001). Likely overlap: both inject true-label information (the honest test's
  T10 targets have no look-ahead; R18's LA12 targets do).
- Mixed-pair GRU seed spread: 0.66635 vs 0.66575 (0.0006). Narrow GRU (96) costs ~0.0006 for ~5 min.
- Platform runtimes ran ~5% FASTER today (mixed pairs 47.5-48.6 min vs 49.9-51.3 on 11-03).

## 2026-11-04: plan execution (score-improvement plan, workstreams 1, 2, 4)

- Stopped: `R19_LA25_DIST` (distributional head: no platform gain) and the honest tests (`H_DIST_CE05` final EMA
  0.690470 at epoch 24, `H_DIST_CE0` 0.688902 at epoch 19).
- **WS2 cross-fitted look-ahead labels:** `run_la_cf.sh GPU FOLD` - five la12 teachers, teacher F trained on every
  sequence of train AND valid except fold F (index % 5), scored on valid fold F (server A GPU 0/1: folds 0/1; server B:
  folds 2/3 on GPU 0, 4 on GPU 1; ~27 min/epoch -> ~18:30). `run_cf_student.sh` (server B) then builds OOF soft targets
  (`teacher_preds.py --oof`: each sequence labelled only by the teacher that never saw it), blends 0.5 TV12 + 0.5
  LA12cf and trains `R20_LA12CF_A` (R3 A recipe) - pairs with R6_LA12_A (0.66554).
- **WS4 schedule check:** `R21_LA12_E32` (local): LA12 targets, 32 epochs, LR halved at 17/24/29.
- **WS1 fusion GRU member** (`--arch fusion_gru`, `FusionGRUModel`, 123k params like the rival's 124k): per feature
  [x, x - x_prev, x - EMA_0.7, x - EMA_0.97, cos x] over ALL 112 inputs -> Linear 64 + ReLU -> ONE GRU layer (128) ->
  head on [h, embedding]. Training computes the EMAs with an exact causal conv (`causal_ema`); streaming == full pass
  (7e-8). `export_slim.py build_fusion`: the EMA features are linear in [x, state, cos x], so they are folded into ONE
  embedding MatMul (22 nodes, exact 1.5e-7); `--combine` merges member graphs into one (prediction = mean, exact).
  LSTM LA12 + fusion pair: 118 us/step = the current mixed pair (~50 min). Training `R22_LA50_FUSION` (local) and
  `R22_LA25_FUSION` (server B GPU 0), ~23:30.
- `build_plan1104.sh` packages all four as singles + pairs (cross-fit / E32 student + GRU LA50; fusion + LSTM LA12_A).
- **Joint pair `032_jointpair_LA12_LA25_L05`: 0.66601** (0FD6M9SQ, 49m7s) = the independently trained pairs of the same
  split (0.66635 / 0.66575, mean 0.66605). Joint training neither helps nor hurts; not pursued further.

### 2026-11-05 00:30 status
- Packaged: `032_fxpair_LA12_A_LA25_FUSION` / `032_fxpair_LA12_A_LA50_FUSION` (LSTM LA12_A + fusion GRU; combine exact,
  ~50 min, same as the GRU mixed pairs), fusion singles (~27 min), `031_W256_LA12_A` (~38 min).
- Cross-fitted la12 teachers: held-out valid-fold EMA 0.7407 / 0.7366 / 0.7490 / 0.7358 / 0.7462 (folds 0-4); OOF soft
  targets built 21:03. The student `R20_LA12CF_A` died silently at epoch 6 (no traceback; server B is shared) - log kept as
  `train_R20_LA12CF_A_died_epoch6.log`; restarted from scratch on server B GPU 1 at 00:30 (~3 batch/s -> ~07:30).
- `R21_LA12_E32` at epoch 25/32 (local, ~03:30).

### 2026-11-05 platform: everything lands at ~0.6664 (plateau)

| zip | platform | runtime | vs |
|---|---|---|---|
| `032_mixpair_LA12_E32_LA50_GRU` (32-epoch LSTM) | 0.66641 (2II6EQPX), best by 0.00003 | 47m36s | same pair, 24 epochs: 0.66638 |
| `032_fxpair_LA12_A_LA50_FUSION` | 0.66640 (1F9XSABM) | **45m47s** | plain GRU LA50: 0.66638 |
| `032_fxpair_LA12_A_LA25_FUSION` | 0.66632 (2CEQF056) | 48m11s | plain GRU LA25 pairs 0.66635 / 0.66575 |
| `031_W256_LA12_A` | 0.66573 (1IOVH965) | 37m42s | LA12 0.66554 |

- Fusion GRU = plain GRU as a partner (no gain from the rival's member design), but its pair is a little faster.
- 32 epochs = 24 epochs. Width 256 on LA12 +0.0002.
- Seven LSTM+GRU-type pairs since 11-03: 0.66575-0.66641; the recipe family has saturated at ~0.6663 +- 0.0003.
- **Cross-fitted pair `032_mixpair_LA12CF_A_LA50_GRU`: 0.66612** (034E9WYJ, 49m57s) vs the same pair with in-sample LA12
  labels 0.66638 (24 epochs) / 0.66641 (32 epochs): **-0.0003**, inside pair seed noise but not a gain. Cross-fitted
  look-ahead labels (rival: +0.0007) do not help us - like the causal out-of-fold test (-0.0008). Closed.
