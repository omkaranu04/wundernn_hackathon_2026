# 031: seed-ensemble distillation

Requested 2026-10-12 after 030 showed that **platform seed noise (0.0013, two
seeds of the same recipe: 0.6605 / 0.6592) is as large as every single-change
gain since 028**. An ensemble of seeds averages that noise away but cannot be
submitted (two lean models already exceed the time budget). Distillation
keeps the submission at ONE lean model: train N seeds, average their
predictions, and train a student of the same size to match that average.

Unlike 029's `teacher_bi` (which saw the future and simply read the targets
off, WP 0.997), these teachers are causal: their average is a smoother,
lower-variance estimate of the same thing the student must predict.

## Plan

1. **Teachers: 10 seeds** (`T_seed10` .. `T_seed19`) of 029's floor10 recipe -
   lean model, scored-probability weights with floor 0.1, EMA 0.9998 - on
   train.parquet only, **fixed 14-epoch schedule, no validation pass**
   (`--no-validation --epochs 14 --lr-milestones 10`, final `ema.pt`).
   valid.parquet is never touched, so every valid number below is honest.
2. **Ceiling:** `teacher_preds.py --eval` on valid: WP of each teacher and of
   the mean of the first k, k = 1..10. This is the most a student can inherit.
3. **Soft targets:** `teacher_preds.py --out` writes the 10-teacher mean for
   every row of train (and of valid, where the teachers are out-of-sample).
4. **Student S1 (locally scored):** floor10's recipe + `--soft-targets
   --distill-weight A`, with validation, compared against the same recipe
   without distillation: 029 floor10 seed 0 (raw 0.686718 / EMA 0.687892) and
   030 `floor10_seed1`. Try A = 0.5 first, then 1.0 / 0.25 depending on it.
5. **Student S2 (platform):** 030-E's recipe (floor 0.1 + auxiliary head +
   train+valid, fixed 14 epochs) + distillation, soft targets for valid from
   the same train-only teachers. Compare with E's 0.6605 / 0.6592.

Known weakness: the teachers' predictions on TRAIN are in-sample (each
teacher was fitted to those labels). Standard distillation does exactly this
and averaging 10 seeds removes the seed-specific part of the memorisation,
but not the shared part. If S1 does not beat its baseline, cross-fitted
teachers (each sequence labelled only by teachers that did not train on it)
are the fix to try.

## Code (030's, plus)

- `train.py`: `--soft-targets PATH`, `--extra-soft-targets PATH`,
  `--distill-weight A`: loss = (1-A) * L(pred, y) + A * L(pred, teacher mean),
  L = the usual 0.5 MSE + 0.5 per-sequence Pearson in both terms. The row
  weights (scored-probability x |clip(y)|) always come from the TRUE y, so
  the emphasis still follows the metric. A = 0 is exactly 030's training.
  Unit-checked: the term equals the hand-written formula; A=0 path unchanged.
- `dataset.py`: optional `soft_targets_path` -> `soft_targets` per item.
- `teacher_preds.py`: mean prediction of any set of checkpoints over a parquet
  file (`--out`, float16 (n_seq, 20000, 2)) and/or per-model + cumulative
  ensemble WP on a file with `is_scored` (`--eval`).
- `queue_job.sh` (now picks the GPU with the fewest of our own jobs),
  `launch_teachers.sh`.

## Status

- 2026-10-12 11:15: 10 teacher seeds queued on server A and server B (shared home,
  each starts once). `T_seed17` started at once on server B; the rest wait for
  GPU memory (both servers are full; 030's five runs hold part of server A).
- 2026-10-12 11:17: **pilot ceiling test** on valid with five models we
  already have (029 floor10, 028-B, 028-A, 029 A_v2labels, 029 power2; all
  train-only, but different recipes and each selected on valid, so slightly
  optimistic): `logs/pilot_ceiling_existing5.log`.

- 2026-10-12 16:20: `T_seed17` finished (server B, ~5 h). All other teachers had
  started on server A, which was then running 11 of our jobs at 2 s/batch
  (epoch 3 of 14 after 5 h). `T_seed12` and `T_seed15` were restarted from
  scratch on server B (1.5 batch/s); `T_seed16` and `T_seed19` were also moved
  but ran out of memory when another user's job took server B's GPU 1 at the
  same moment, and are requeued through `queue_job.sh` on both servers.
  `T_seed10/11/13/14/18` continue on server A.

## Results

### Pilot ceiling on valid, five existing models (2026-10-12)

`logs/pilot_ceiling_existing5.log` (batched WP, full valid; each model's
checkpoint was selected on valid, so slightly optimistic):

| k | model added | its single WP | mean of first k |
|---|---|---|---|
| 1 | 029 floor10 | 0.687892 | 0.687892 |
| 2 | 028-B (aux head) | 0.687037 | 0.689052 |
| 3 | 028-A | 0.686746 | 0.689135 |
| 4 | 029 A_v2labels | 0.687083 | 0.689288 |
| 5 | 029 power2 (weak) | 0.684285 | 0.688920 |

**The ensemble gain is small and saturates at once:** +0.0012 over the best
single with 2 models, +0.0014 with 4 (+0.0021 over the average single), and a
weak member takes it back. So same-family lean models are nearly the same
function, and the student's ceiling is ~+0.002 locally - the size of seed
noise. 10 same-recipe seeds will not add much over 4-5; DIVERSITY of the
teachers matters more than their number (023 measured +0.004 for an ensemble
of different architectures; a rival's public notes for this competition say
the same: "same-loss pairs added nothing, loss diversity was the gain").

### int8 quantisation (2026-10-12; idea from a rival's public notes)

`onnxruntime.quantization.quantize_dynamic` on the exported graph, LSTM
weights only (`op_types_to_quantize=["LSTM"]`, 46 nodes kept).

Accuracy, `onnx_subset_wp.py` (row-by-row ONNX on every 25th valid sequence,
75 sequences, 029 floor10): fp32 WP 0.703486; int8 LSTM 0.703236 (-0.00025);
int8 everything 0.703242. Mean |prediction difference| 0.02, max 1.0.

Latency, alternating benchmark on a BUSY machine (noisy, +-10%; this CPU has
AVX2 but no VNNI), relative to the current fp32 hidden-128 graph:

| graph (random weights, 46 nodes) | fp32 | int8 LSTM |
|---|---|---|
| hidden 128 | 1.00 | 0.84 (trained E model: 0.92) |
| hidden 192 | 1.37 | 0.79 |
| hidden 256 | 1.49 | 1.14 |

Reading: int8 costs ~0.0003 WP and makes WIDE LSTMs much cheaper. A hidden-256
LSTM with block-diagonal weights is exactly two independent hidden-128
models, so a 2-model ensemble in ONE graph might cost ~1.15x today's latency
(~47 min) instead of 2x. Must be re-measured on a quiet machine and on the
platform before trusting it.

### Post-hoc output transforms (no retraining)

`posthoc_probe.py` on 029 floor10: causal smoothing, sharpening, power
transforms, rescaling and cross-target mixing of the predictions. On the
FULL valid set (`logs/posthoc_probe_floor10.log`, raw 0.687892) every variant
is below the raw output: smoothing a=0.9 -0.0001, a=0.75 -0.0006; sharpening
-0.0017; power 0.85 -0.0005, power 1.15 -0.0009; scale x0.5 -0.00001, x1.5
-0.0017; cross-target mixing -0.0012 or worse. The output is already well
shaped; nothing to gain here. Closed.

### Clean-teacher ceiling, first five seeds (2026-10-13 00:45)

`logs/ceiling_first5_teachers.log` on the shared server home. These teachers
never saw valid (fixed schedule, no selection), so the numbers are honest:

| k | teacher | single WP | mean of first k |
|---|---|---|---|
| 1 | T_seed12 | 0.686946 | 0.686946 |
| 2 | T_seed15 | 0.685473 | 0.687994 |
| 3 | T_seed16 | 0.687075 | 0.688865 |
| 4 | T_seed17 | 0.686180 | 0.689175 |
| 5 | T_seed19 | 0.686723 | 0.689510 |

- **Local seed noise, measured for the first time:** five seeds of one recipe
  span 0.6855-0.6871 (mean 0.68648, sd ~0.0006). Single-run differences below
  ~0.0015 are noise; 029/030's floor and dropout probes all sit inside it.
- An unselected fixed-schedule run averages 0.6865, vs 0.6879 for floor10
  seed 0 with best-epoch selection on valid: selection flatters by ~0.001.
- **Five seeds: +0.0030 over the average single, still rising** (+0.0003
  from the 5th). Two seeds: +0.0018. Better than the mixed pilot.
- T_seed10/11/13/14/18 finish ~03:00; then the 10-seed ceiling, the soft
  targets and the students.

### Two models in ONE graph (2026-10-13) - submittable now

`merge_models.py`: K lean models -> one lean model with K times the widths and
block-diagonal weights; output = mean of the members, exact (max difference
2e-07 to 5e-07). Same 46-node graph, unchanged `export_onnx.py` /
`solution.py`. `quantize_onnx.py`: int8 LSTM weights, single file.

Latency, alternating benchmark on a quiet machine (7 / 5 repeats, tight):

| graph | % of budget uncorrected | vs single fp32 |
|---|---|---|
| single lean fp32 (today's submissions, ~41 min on the platform) | 104-106% | 1.00 |
| single lean int8 | - | ~0.87 (subset run: 91 vs 105 us/step) |
| merged pair fp32 | 183.7% | 1.77 (would time out) |
| **merged pair int8** | **123.0%** | **1.16** (-> ~48 min if the platform's int8 behaves like ours) |

Accuracy of int8, row-by-row ONNX on 75 valid sequences: merged local pair
0.703488 fp32 -> 0.703307 int8; single E 0.722775 -> 0.722949; merged E
0.720956 -> 0.721053. So +-0.0002: noise.

Zips (`submissions/`), both solution.py + one self-contained `checkpoints/model.onnx`:

| zip | content | purpose |
|---|---|---|
| `031_E_seed0_int8_2026-10-13.zip` (0.24 MB) | 030-E seed 0 (platform 0.6605, 41m2s as fp32) with int8 LSTM | measures the PLATFORM's int8 speed on a known model; expect ~0.6605, ~36 min |
| `031_E_pair_merged_int8_2026-10-13.zip` (0.48 MB) | 030-E seed 0 + seed 1 merged, int8 | the ensemble; expect ~0.661-0.662 (two-seed gain +0.0015-0.0018 over the 0.65985 average), ~48 min IF int8 is as fast on the platform |

The merged zip relies on int8 for its time budget (fp32 would need ~73 min),
so submit the single int8 zip first and read its runtime. Real step-by-step
scorer on a locally scorable merged pair (029 floor10 + 028-B, int8; batched
fp32 mean = 0.689052) started 00:55:
`logs/real_score_merged_local_floor10_Baux_int8.log`.

### Platform results (submitted 2026-10-13) - new best 0.6621

| zip | platform | runtime | reference |
|---|---|---|---|
| `031_E_seed0_int8` | **0.6605** (ID 1552I7BF) | **35m2s**, 628 MB | same weights in fp32: 0.6605, 41m2s |
| `031_E_pair_merged_int8` | **0.6621** (ID 1N20IE18) - best so far | **47m17s**, 625 MB | members alone: 0.6605 and 0.6592 |

- **int8 is free on the platform:** identical score to four decimals and
  0.854x the runtime (local estimate was ~0.87x). Every future single-model
  zip should ship int8: 6 minutes of headroom for nothing.
- **The merged pair gained +0.0016 over its better member and +0.0022 over
  the members' average (0.65985)** - local predicted +0.0015 to +0.0018 for two
  seeds. Runtime 47m17s = 1.15x the fp32 single, exactly the local 1.16x.
- This is the first gain since 028 that is clearly larger than seed noise
  (0.0013), and it is the kind that should hold on any test set: averaging
  two seeds reduces variance rather than fitting anything.

### Unattended pipeline (started 2026-10-13 01:56 on server B)

`run_after_teachers.sh` (log `logs/pipeline.log` on the shared server home)
waits for all ten teachers, then: (1) 10-teacher ceiling on valid plus soft
targets for valid (`soft/valid_T10.npy`, `logs/ceiling_10_teachers.log`);
(2) soft targets for train (`soft/train_T10.npy`); (3) queues four students
through `queue_job.sh` on server B:

| student | recipe | scored |
|---|---|---|
| `S2_a50_seed0`, `S2_a50_seed1` | 030-E (floor 0.1 + aux + train+valid, fixed 14 epochs) + `--distill-weight 0.5` | platform only; the pair is the merge candidate |
| `S1_a50` | 029 floor10 (with validation) + `--distill-weight 0.5` | locally, vs 0.68648 average clean seed / 0.687892 floor10 seed 0 |
| `S1_a100` | same, `--distill-weight 1.0` (teacher targets only) | locally |

Also measured 2026-10-13: three models in one int8 graph cost 1.56x a single
fp32 model (164.9% of budget, ~64 min) and four cost 2.10x, so a PAIR is the
largest merge that fits at hidden size 128.

### Ten-teacher ceiling and soft targets (2026-10-13 03:15-04:50, unattended)

`logs/ceiling_10_teachers.log`: singles 0.685473-0.687291 (mean 0.686488,
sd 0.0007); mean of the first k: 0.685813, 0.688225, 0.689021, 0.689042,
0.689443, 0.689410, 0.689604, 0.689678, 0.689823, **0.689934** at k=10.
So the ten-seed ensemble is **+0.0034 over the average single**, with most of
it by k=5 (+0.0030) and ~+0.0001 per extra seed after that. Soft targets:
`soft/train_T10.npy` (849 MB), `soft/valid_T10.npy` (150 MB) on the server.

Real step-by-step scorer on the locally scorable merged pair (029 floor10 +
028-B, int8, the staged zip): **0.688896** (t0 0.684550, t1 0.693243) vs
0.689052 for the fp32 batched mean - the merged int8 graph is confirmed end
to end on full valid (int8 cost -0.00016).

### Students (started 04:50-04:57 on server B; status 2026-10-13 09:35)

All four share server B with other users (~1.1 batch/s each, ~47 min per epoch).

| student | epoch | batched valid WP so far (raw / EMA) | same-epoch reference |
|---|---|---|---|
| `S1_a100` (teacher targets only) | 6 | epoch 4: 0.686544 / **0.687029**; epoch 5: 0.685487 / 0.686922 | floor10 seed 0 at epoch 4: 0.680019 / 0.672672; 028-A: 0.679625 / 0.682059 |
| `S1_a50` (half and half) | 5 | epoch 4: 0.685381 / 0.685343 | same |
| `S2_a50_seed0`, `S2_a50_seed1` | 6 of 14 | no validation | expected done ~16:00 |

Early reading: after 5 epochs the pure-teacher student is already where an
ordinary run peaks after 11-13 (0.6865-0.6879), and it is ahead of the
half-and-half one. Too early for the final number, but learning is much
faster on the smoother targets.

Because a100 leads, **`S2_a100_seed0` / `S2_a100_seed1`** (same as the S2 runs
but `--distill-weight 1.0`) were started 09:40 on server A GPU 0 (1.1 batch/s
there too; expected done ~20:30), so both S2 pairs exist when S1 settles which
weight is better. (A test showed server A is no faster than server B right now, so
the running students were not moved.)

### Three more merged pairs from existing train+valid models (2026-10-13 10:30)

Requested by the user ("we have slots"). Same construction as the 0.6621 zip
(`merge_models.py` -> `export_onnx.py` -> `quantize_onnx.py`), members are the
four train+valid models we have. All: merge exact to 4e-07, export check
passed, 46 nodes, state (2,1,256), 0.48-0.49 MB, fresh-extraction check passed,
latency 1.004x the already-scored pair (120% of budget -> ~47 min).

| zip (`submissions/`) | members (platform scores alone) | int8 vs fp32 on 75 valid sequences |
|---|---|---|
| `031_pair_E0_D_merged_int8_2026-10-13.zip` | 030-E seed 0 (0.6605) + 029-D (0.6596) | 0.721586 -> 0.721646 |
| `031_pair_E1_D_merged_int8_2026-10-13.zip` | 030-E seed 1 (0.6592) + 029-D (0.6596) | 0.717123 -> 0.716923 |
| `031_pair_E0_C_merged_int8_2026-10-13.zip` | 030-E seed 0 (0.6605) + 028-C (0.6580) | 0.722739 -> 0.722932 |

(Subset WPs are inflated: all members trained on valid; only the int8-fp32
difference means anything.) Expectation: each ~0.661-0.662, i.e. the same as
the scored E0+E1 pair within seed noise; the three are three draws, not three
different ideas. No local score is possible for any of them.

### Platform results of the three extra pairs (submitted 2026-10-13) - new best 0.6626

| pair | platform | runtime | members alone | pair minus members' average |
|---|---|---|---|---|
| E0 + D | **0.6626** (ID 2MJE4430) - best so far | 50m21s | 0.6605, 0.6596 | +0.0026 |
| E0 + C | 0.6624 (ID 2AI2V07T) | 50m32s | 0.6605, 0.6580 | **+0.0032** |
| E0 + E1 (earlier) | 0.6621 (ID 1N20IE18) | 47m17s | 0.6605, 0.6592 | +0.0022 |
| E1 + D | 0.6617 (ID 1D0Z8X2I) | 50m7s | 0.6592, 0.6596 | +0.0023 |

- **Every pair beats every single model** (best single 0.6605); four pairs
  span 0.6617-0.6626, a tighter band than the singles (0.6580-0.6605).
- **Mixed-recipe pairs gain more than the same-recipe pair.** E0+C pairs the
  best single with the weakest and most different one (no auxiliary head, no
  floor) and still reaches 0.6624: +0.0032 over its members' average, vs
  +0.0022 for two seeds of one recipe. Diversity is worth about +0.001.
- **Runtime rose to ~50 min** for the identical graph shape that took 47m17s
  the night before (three were running at once). The platform's speed moves
  by ~6%; a graph at 50 min has less margin than it looked.

### Graph rewrite: where the time goes, and a 13-node export (2026-10-13 10:30-11:15)

`export_slim.py` (three builders, all verified exact to 3e-07..6e-07 against
PyTorch / the mean of the members) and `solution_slim.py` (takes the state
layout from the graph). Alternating benchmark, quiet machine, relative to the
standard fp32 single model (= 1.00, ~41 min on the platform):

| graph | nodes | fp32 | int8 |
|---|---|---|---|
| standard export, single | 46 | 1.00 | ~0.87 (platform: 0.854) |
| standard export, merged pair | 46 | 1.77 | 1.16 (platform: 1.15-1.23) |
| **`build`: slim, single** | 13 | 0.805 | **0.679** |
| **`build`: slim, merged pair (block-diagonal 256)** | 13 | - | **0.957** |
| `build`: slim, three members (block-diagonal 384) | 13 | - | 1.377 |
| `build`: slim, four members (512) | 13 | - | 1.920 |
| `build_multi`: K separate LSTM chains, K = 2 / 3 / 4 | 16-20 | 2.05 (K=3) | 1.158 / 1.614 / 2.086 |
| `build_batched`: LSTM cell from primitive ops, members as a batch, K = 1 / 2 / 3 / 4 / 6 / 8 | 36 (56 int8) | 0.88 / 1.20 / 1.55 / 1.80 / 2.43 / 2.98 | no gain (0.91 / 1.23 / - / 1.82 / 2.41 / 2.97) |

Cost model that fits these (units of the standard single):
fixed ~0.22 (Python + `session.run` + encoders + head) plus, per int8 `LSTM`
node, ~0.19 + 2.5e-6 * width^2 (128: 0.23, 256: 0.37, 384: 0.58, 512: 0.85).
- The 33 plumbing nodes of the standard export cost ~0.2 units (8 platform
  minutes); folding them into the weights is free and exact.
- An ONNX `LSTM` node has a large fixed per-call cost, so K separate chains
  lose to one wide block-diagonal LSTM, and a batched primitive-op cell loses
  to both (ORT's batched MatMul is slow and int8 does not help it).
- So the levers for more members are fewer LSTM NODES (one layer instead of
  two) and smaller WIDTH (narrower members), not cleverer graphs:
  3 x 96 (width 288) ~1.02, 4 x 80 (320) ~1.11, 3 x 112 (336) ~1.16; with ONE
  layer: 3 x 128 ~0.80, 4 x 128 ~1.07 (estimates from the cost model).

Zip built: `submissions/031_pair_E0_D_slim_int8_2026-10-13.zip` - the 0.6626
pair (E0 + D) in the 13-node graph with `solution_slim.py`. Checks: slim fp32
= standard merged fp32 exactly (0.721586 both, 75 valid sequences); slim int8
0.721626 (standard int8 0.721646); through `PredictionModel.predict` vs the
zip that scored 0.6626: max |difference| 0.018 over 2,802 predictions incl. a
sequence switch (two different int8 roundings of the same weights). Expected:
~0.6626 in ~40 min instead of ~50.

Probes started 10:45-10:50 to see what narrow / shallow distilled students
cost in accuracy (both `--distill-weight 1.0`, with validation; compare with
`S1_a100`, the hidden-128 two-layer student):
`P_h96_a100` (`--hidden-size 96`, server A GPU 1, ~8.5 batch/s) and
`P_L1_a100` (`--num-layers 1`, local 1080 Ti, ~2.1 batch/s; log
`logs/train_P_L1_a100_local.log`, weights `results/checkpoints_P_L1_a100/`).

**Platform result of the slim pair (2026-10-13): 0.6626 in 41m57s** (ID
12KIOI5F, 625 MB). Same score to four decimals as the 46-node zip of the same
two models (0.6626, 50m21s), 8.4 minutes faster (0.833x; local predicted
0.957 / 1.16 = 0.825x). The 13-node export + `solution_slim.py` is now the
default for every zip: a merged pair costs what a single fp32 model used to.

### Narrow students: probe reading and a three-member plan (2026-10-13 11:50)

- `P_h96_a100` (hidden 96, teacher targets only, train only) at epoch 9:
  raw 0.686851 / EMA **0.687458**, still rising. `S1_a100` (hidden 128, same
  recipe) at epoch 7: 0.687306 / 0.687725; `S1_a50`: 0.686914 (its LR already
  halved). So a distilled width-96 student is within ~0.0003 of a width-128
  one, and both are already above the average ordinary seed (0.686488).
  `P_L1_a100` (one layer) at epoch 1: 0.679807 (S1_a100 had 0.681449).
- Latency of block-diagonal slim int8 merges, relative to the slim pair that
  ran 41m57s on the platform: **3 x 96: 1.137 (~48 min)**; 4 x 80: 1.298 and
  3 x 112: 1.294 (~54 min, too tight); 4 x 96: 1.49.
- So the next candidate is a merged TRIPLE of width-96 students.
  `run_narrow_students.sh` (server A, log `logs/narrow_chain.log`) waits for the
  probe, then trains three train+valid width-96 students in turn on server A's
  idle GPU 1 (~8 batch/s alone, ~1.6 h each), deliberately with three recipes:
  `S2h96_A` (floor 0.1, aux, seed 0), `S2h96_B` (floor 0, aux, seed 1),
  `S2h96_C` (floor 0.3, no aux, seed 2). All `--distill-weight 1.0`.
  `merge_models.py` already handles K = 3.

### Status 2026-10-13 13:10

- **`P_h96_a100` (hidden 96, train only) reached raw 0.688928 / EMA 0.688854
  at epoch 20 and is still running.** A single width-96 distilled student is
  +0.0024 above the average ordinary seed (0.686488), above the best ordinary
  run (0.687892), and has kept ~70% of the ten-teacher gain (0.689934).
- `S1_a100` (hidden 128) epoch 9: 0.687872 / 0.688128, LR still 1e-3.
  `S1_a50`: 0.686913 / 0.687687. `P_L1_a100` (one layer) epoch 3: 0.684565
  (S1_a100 had 0.685409 at epoch 3).
- server A's GPU 0 emptied, so the sequential chain was cancelled and
  `S2h96_A/B/C` were started together on it at 13:05 (2.6 batch/s each,
  ~20 min per epoch, done ~17:50). `S2_a100_seed0/1` share that GPU (epoch 6
  of 14, done ~18:45); `S2_a50_seed0/1` on server B are at epoch 11 of 14
  (done ~15:40).

### Local student results and the first student zips (2026-10-13 17:45)

Batched valid WP, train-only students (teachers' average single 0.686488,
ten-teacher ensemble 0.689934, best ordinary run 0.687892):

| student | best raw | best EMA | state |
|---|---|---|---|
| `P_h96_a100` (hidden 96, teacher targets only) | **0.689267** | **0.689213** | finished all 30 epochs |
| `S1_a100` (hidden 128, teacher targets only) | 0.689240 | 0.688992 | epoch 17, LR still 1e-3 |
| `S1_a50` (hidden 128, half true targets) | 0.687864 | 0.687687 | early stop, epoch 15 |
| `P_L1_a100` (ONE layer, hidden 128) | 0.687732 | 0.687596 | epoch 11, running (local) |

- **Distillation works:** one model at 0.6892, i.e. +0.0027 over the average
  ordinary seed and ~80% of the ten-teacher ensemble's gain (+0.0034).
- **Teacher targets only (a100) beat half-and-half (a50) by ~0.0013-0.0015.**
- **Width 96 loses nothing** (0.689213 vs 0.688992 so far for width 128), and
  students keep improving for 25-30 epochs where ordinary runs peak at 11-13
  and then overfit - so the fixed 14-epoch S2 schedule is probably too SHORT
  for students (next round: 24+ epochs).
- A one-layer student is already level with ordinary two-layer runs.

`S2_a50_seed0/1` (train+valid, a50) finished 15:34 / 15:43; weights copied
back (checksums verified). Zips, both the 13-node int8 graph with
`solution_slim.py`:

| zip (`submissions/`) | content | latency vs the slim E0+D pair (41m57s) | purpose |
|---|---|---|---|
| `031_S2a50_pair_slim_int8_2026-10-13.zip` | the two a50 students merged | 0.976 (~41 min) | first distilled pair; compare with 0.6626 (E0+D) / 0.6621 (E0+E1) |
| `031_S2a50_seed0_single_slim_int8_2026-10-13.zip` | one a50 student | 0.703 (~30 min) | controlled test: one distilled student vs the same recipe undistilled (E0 0.6605, E1 0.6592) |

Checks: merge exact (4e-07), slim export exact (7e-07), int8 vs fp32 on 75
valid sequences 0.717216 -> 0.716874, `PredictionModel.predict` run over 1,200
steps for each zip (shape, dtype, finite, None handling). No local score is
possible (students trained on valid). Note they are the WEAKER variant (a50);
the a100 pair and the width-96 triple finish tonight.

`031_S2a50_pair_slim_int8_2026-10-13.zip` was submitted 2026-10-13 ~17:45
(result pending); the single-student zip and tonight's zips go in tomorrow.

`build_tonight.sh` (local, unattended, log `logs/build_tonight.log`) waits for
`S2h96_A/B/C` and then `S2_a100_seed0/1` to finish on server A, copies them back
with checksum verification, and runs merge -> `export_slim.py` ->
`quantize_onnx.py` -> `onnx_subset_wp.py` -> `make_zip.py` (new: builds the zip,
extracts it into a fresh folder and runs `PredictionModel.predict` over 1,200
rows). Expected zips, dated by build day:
`031_S2h96_triple_slim_int8_<date>.zip`, `031_S2a100_pair_slim_int8_<date>.zip`.
A dry run of a three-member merge + slim export on existing models was exact
(4e-07 / 6e-07).

### Platform result of the first distilled pair (2026-10-13): 0.6620 - no gain

`031_S2a50_pair_slim_int8`: **0.6620** in 41m17s (ID 09550JTJ). The same
recipe WITHOUT distillation, as a pair, scored 0.6621 (E0+E1); the other
undistilled pairs 0.6617-0.6626. So half-weight distillation added nothing
measurable at the pair level. Reasons, in order of likely size:

1. **Students of one teacher are near-copies of each other.** Each is pulled
   toward the same ten-teacher mean, so averaging two of them cancels much
   less than averaging two independent models. Distillation moves the gain
   from "ensemble of members" into "each member"; a pair of students is not
   (student gain) + (pair gain). Locally: a50 student ~0.6877 vs an ordinary
   pair 0.6880 - the same thing.
2. a50 is the weaker variant (a100 is +0.0013 locally) and 14 epochs is short
   for a student (they improve until epoch 25-30).
3. **Design flaw for tonight's a100 runs, found now:** with
   `--distill-weight 1.0` the student never sees true labels, and the teachers
   were trained on train only. So the S2 a100 students (pair and width-96
   triple) get validation FEATURES but not validation LABELS - they lose the
   +0.0011..0.0016 that train+valid gave every ordinary model. Expect them
   near 0.661, not above 0.6626.

Fix = round 2: teachers trained on train+valid (reuse E0, E1, D, C and add
six more with mixed recipes), soft targets from those ten, students with
teacher targets only, width 96, ~26 epochs.

### Round 2 started (2026-10-13 18:34, server B, unattended)

`run_round2.sh` (log `logs/round2.log` on the shared server home; started
through `r2_restart.sh`):

1. Teachers = the four TRAIN+VALID models we already have, copied to
   `tv_teachers/`: 030-E seed 0, 030-E seed 1, 029-D, 028-C (three recipes;
   platform 0.6605 / 0.6592 / 0.6596 / 0.6580 alone, pairs up to 0.6626).
2. Soft targets from their mean: `soft/valid_TV4.npy`, `soft/train_TV4.npy`.
   The validation LABELS now reach the students through the teachers.
3. Students, teacher targets only (`--distill-weight 1.0`), train+valid,
   fixed 24 epochs with LR halved at 13 / 18 / 22 (students keep improving to
   epoch 25-30): `R2h96_A` (floor 0.1, aux, seed 0), `R2h96_B` (floor 0, aux,
   seed 1), `R2h96_C` (floor 0.3, no aux, seed 2) on GPU 0 for a merged
   triple; `R2h128_A` (floor 0.1, aux), `R2h128_B` (floor 0, no aux) on GPU 1
   for a pair. Expected to finish around 06:00-10:00 on 2026-10-14.

A first attempt at 18:32 trained six MORE train+valid teachers first
(TV20-25); it was stopped after two minutes because that would have delayed
the students by ~10 hours for an expected ~+0.0005. Those teachers can be
trained later for a round 3 or as extra pair members.

No local score is possible for anything in round 2 (teachers and students
all saw valid); only the platform can judge it. Round 1's locally scored
students (0.6892 for one model) are the evidence that the method works.

### Status 2026-10-13 21:35

- **Round-1 width-96 triple packaged unattended:**
  `submissions/031_S2h96_triple_slim_int8_2026-10-13.zip` (0.47 MB, 13 nodes,
  state width 288). Merge exact (4e-07), slim export exact (7e-07), int8 vs
  fp32 on 75 valid sequences 0.706705 -> 0.707117, latency 1.144x the slim
  E0+D pair (~48 min), 1,101 predictions verified from the extracted zip.
- Its 75-sequence WP (0.7067) sits with models that never saw valid labels
  (0.7035) and far below valid-trained ones (0.7216): confirmation that
  teacher-only students of train-only teachers do not get the validation
  labels. Flip side: their valid score is HONEST, so the triple can be scored
  locally: `logs/honest_eval_S2h96_triple.log` (running on CPU).
- Round-1 a100 pair: last epoch running, zip expected ~22:10.
- Round 2 (server B): soft targets done 18:58; `R2h96_A/B/C` at epoch 7 of 24,
  2.5 batch/s (done ~03:10); `R2h128_A/B` at epoch 4 of 24, 1.2 batch/s
  (done ~11:30 unless that GPU frees).
- Local: `S1_a100` 0.689240 / 0.689246 at epoch 27; one-layer `P_L1_a100`
  0.688481 / 0.688067 at epoch 18 - a ONE-layer student is only ~0.001 below
  a two-layer one, and one-layer members need one LSTM node instead of two.

### Honest local score of the round-1 narrow students and their triple (2026-10-13 22:00)

Round-1 teacher-only students never received a validation label (their
teachers were train-only), so their valid score is honest
(`logs/honest_eval_S2h96_triple.log`, batched, full valid):

| model | single WP | mean of first k |
|---|---|---|
| `S2h96_A` (floor 0.1, aux; 14 epochs) | 0.688575 | 0.688575 |
| `S2h96_B` (floor 0, aux) | 0.689192 | 0.689142 |
| `S2h96_C` (floor 0.3, no aux) | 0.688197 | **0.689020** (the triple) |
| `P_h96_a100` (30 epochs, selected on valid) | 0.689214 | 0.689225 |

**Averaging students gains nothing:** the triple (0.689020) is below its best
member (0.689192); four students give 0.689225. Each student already sits
within ~0.0007 of the ten-teacher mean it was trained on (0.689934), so there
is nothing left for an average of students to cancel. Consequences:

- A merged pair/triple of students of ONE teacher set is a waste of latency:
  submit a single student (~30 min) instead.
- The ceiling is the TEACHER ENSEMBLE. Ten same-recipe seeds = +0.0034 over a
  single; to go further the teachers must be more numerous and above all more
  DIVERSE (other architectures / losses), or the base model must improve.
- Expected platform level of round-1 students: ~0.689 - 0.029 = ~0.660 (they
  lack the validation labels). Round 2's teachers are four train+valid
  models whose own pair already scores 0.6626; a round-2 student should land
  around 0.662-0.663, and its triple no higher than one student.

Useful by-product: this is a clean LOCAL protocol for developing the method
(train-only teachers -> students -> honest valid score) before spending
train+valid runs and platform slots.

### Base-model test started overnight (server A, 2026-10-13 22:05)

`--pearson-seq-weight mass` (new flag, default `equal` = unchanged): average
the per-sequence Pearson terms by each sequence's weight mass
sum(row_weight * |clip y|) - its share of the pooled metric - instead of
equally. Unit-checked against the hand-written formula. `M_seed10/11/12` use
exactly the ten clean teachers' protocol and seeds 10/11/12 (references:
0.685813 / 0.686888 / 0.686946; all ten: mean 0.686488, sd 0.0007), so three
runs give a paired, low-noise answer. `launch_massweight.sh`; ~1.05 batch/s,
done ~08:00; score with `teacher_preds.py --eval`.

### Status 2026-10-14 02:10

- Round-1 teacher-only pair packaged 21:53:
  `submissions/031_S2a100_pair_slim_int8_2026-10-13.zip` (merge 4e-07, slim
  8e-07, int8 0.707361 -> 0.707733 on 75 sequences, 1,101 predictions
  verified). Honest local score of its members: `logs/honest_eval_S2a100_pair.log`.
- Final local scores of the train-only students (batched, selected on valid):
  `S1_a100` 0.689406 / **0.689331** (30 epochs); `P_h96_a100` 0.689267 /
  0.689213; one-layer `P_L1_a100` 0.688563 / 0.688402; `S1_a50` 0.687864 /
  0.687687. Ordinary seed: 0.686488.
- Round 2 (server B): `R2h96_A/B/C` epoch 21 of 24 (done ~03:10);
  `R2h128_A/B` epoch 12 of 24 (done ~08:20). `build_round2.sh` packages the
  triple and the pair; `build_r2_single.sh` then packages `R2h96_B` alone
  (`031_R2h96_B_single_slim_int8_<date>.zip`).
- Mass-weighted loss test (server A): `M_seed10/11/12` epoch 7 of 14, done ~06:45.

### Honest local scores of the round-1 full-width students (2026-10-14 02:50)

`logs/honest_eval_S2a100_pair.log` (batched, full valid): `S2_a100_seed0`
0.688950, `S2_a100_seed1` 0.689020, their pair **0.689203** (+0.0002 over the
members' average - again almost nothing from averaging students); adding
`S1_a100` (0.689331): 0.689411; adding the one-layer `P_L1_a100` (0.688402):
0.689298.

Single-model zip of the best ten-seed student:
`submissions/031_S1a100_single_slim_int8_2026-10-14.zip` (`S1_a100`
best_ema, train only, 30 epochs, local 0.689331; slim export exact 7e-07,
int8 0.707329 -> 0.707183 on 75 sequences, 71 us/step locally = ~29 min).

Background jobs as of 2026-10-14 03:10 (nothing needs attending):
local `build_round2.sh` (round-2 triple ~03:30, round-2 pair ~08:45) and
`build_r2_single.sh` (single `R2h96_B` after the triple); server B the five
round-2 students; server A `M_seed10/11/12` and `score_massweight.sh`, which
scores them against `T_seed10/11/12` into `logs/eval_massweight.log` (~07:00).

### Overnight results (2026-10-14)

**Mass-weighted Pearson term (`--pearson-seq-weight mass`) - first base-model
gain since 028.** Paired with the equal-weighted teachers of the same seeds,
identical protocol (`logs/eval_massweight.log`):

| seed | equal (T) | mass (M) | difference |
|---|---|---|---|
| 10 | 0.685813 | 0.687967 | +0.0022 |
| 11 | 0.686888 | 0.688099 | +0.0012 |
| 12 | 0.686946 | 0.686755 | -0.0002 |
| mean | 0.686549 | **0.687607** | **+0.0011** |

All ten equal-weighted seeds: mean 0.686488, sd 0.0007, so three mass-weighted
seeds averaging 0.6876 sit ~2.7 standard errors above. Their 3-model ensemble
is 0.690063 vs 0.689021 for the three equal-weighted ones (+0.0010). Likely
real, ~+0.001; worth adopting for every future teacher and student, and one
more confirmation (2-3 seeds of the train+valid recipe) on the platform.

**Round-2 zips, all packaged unattended (all checks passed):**

| zip (`submissions/`) | members | 75-sequence WP (in-sample, valid seen via teachers) | latency |
|---|---|---|---|
| `031_R2h96_B_single_slim_int8_2026-10-14.zip` | `R2h96_B` (width 96, floor 0, aux) | 0.711981 | 67.6 us/step (~28 min) |
| `031_R2h96_triple_slim_int8_2026-10-14.zip` | `R2h96_A/B/C` | 0.712114 | 118 us/step (~48 min) |
| `031_R2h128_pair_slim_int8_2026-10-14.zip` | `R2h128_A/B` | 0.714086 | 102 us/step (~42 min) |

On those 75 sequences the triple equals its single member again (0.7121 vs
0.7120). In-sample numbers cannot rank them against the E0+D pair (0.7216,
which trained on the valid labels directly).

### Platform results of round 2 (submitted 2026-10-14) - new best 0.6632

| zip | platform | runtime | what it is |
|---|---|---|---|
| `031_R2h128_pair_slim_int8` | **0.6632** (ID 1Y8KUPS8) - best so far | 39m2s | two width-128 students of the four train+valid models, different recipes |
| `031_R2h96_triple_slim_int8` | 0.6625 (1VTSH8TV) | 45m17s | three width-96 students |
| `031_R2h96_B_single_slim_int8` | **0.6623** (1J3TYZOM) | **23m41s** | ONE width-96 student |
| `031_S1a100_single_slim_int8` | 0.6597 (05OWVLW6) | 26m56s | the ten-seed student (train-only teachers; local 0.689331) |

- **Distillation works on the platform once the teachers carry the
  validation labels.** One width-96 student (0.6623, 24 minutes) beats every
  single model we ever had (best 0.6605) by +0.0018 and matches the merged
  pairs of its own teachers (0.6617-0.6626) at half their runtime.
- **The triple of students equals one student** (0.6625 vs 0.6623), as the
  honest local check predicted. Merging students of one teacher set buys
  nothing; do not spend latency on it.
- The width-128 pair (0.6632) is +0.0007..0.0009 above the width-96 triple and
  single: its two members use more different recipes (floor 0.1 + aux vs
  floor 0, no aux) and are wider. Inside seed noise (0.0013), but it is the
  best score so far.
- **The ten-seed student scored exactly as predicted:** 0.6597 = local
  0.689331 - 0.0296 (the usual offset). Train-only teachers cap the student.
- So the student's ceiling is its TEACHERS: the next gains come from a larger,
  more diverse, better (mass-weighted) train+valid teacher pool.

### Round 3 started (2026-10-14 14:29, unattended)

Teachers: eight NEW train+valid models with the mass-weighted Pearson term
(`--pearson-seq-weight mass`, +0.0011 locally) and mixed recipes -
`TVm20/21` floor 0.1 + aux, `TVm22` floor 0 + aux, `TVm23` floor 0, `TVm24`
floor 0.3 + aux, `TVm25` floor 0.3, `TVm26` floor 0.1, `TVm27` floor 0 + aux
(seeds 20-27; six on server B via `run_round3.sh`, two on server A via
`launch_round3_server A.sh`). Soft targets from all twelve train+valid teachers
(the four old ones + eight new). Students: `R3h128_A` (floor 0.1, aux),
`R3h128_B` (floor 0, no aux), `R3h128_C` (floor 0.3, aux), width 128,
teacher targets only, mass-weighted, 24 epochs. Local `build_round3.sh`
packages the A+B pair and singles of A and C when they finish.
Expected: teachers ~00:15, students ~13:00 on 2026-10-15.

`submissions/031_R2h128_A_single_slim_int8_2026-10-14.zip` (built 14:45): member A
of the 0.6632 pair alone (width 128, floor 0.1, aux). Slim export exact
(5e-07); on 75 sequences fp32 0.712429, int8 0.713792 (int8 rounding moves
this one more than usual: mean |diff| 0.023), the pair 0.714086; 71 us/step
(~29 min). Purpose: tells whether the pair's 0.6632 came from pairing two
DIFFERENT-recipe students (single ~0.6623, like the width-96 one) or from
width 128 (single ~0.663).

**Result 2026-10-14: member A alone = 0.6628 in 26m1s (ID 2A8C1AV4).** Width-96
single 0.6623, width-128 single 0.6628, width-128 pair 0.6632: the pair adds
only ~+0.0004 over its own member (noise level), and width 128 is ~+0.0005
over 96. A single full-width student is a ~26-minute model within noise of
the best score - the cheapest finalist with a large runtime margin.

### Status 2026-10-15 10:00 and today's zips

Round-3 teachers `TVm20..27` all finished by 00:28 (copied back, 8 x
checksums verified); soft targets from twelve train+valid teachers at 02:26;
students running on server B: `R3h128_A` epoch 16 of 24 (3.1 batch/s, ~11:50),
`R3h128_B/C` epoch 14 of 24 (1.7 batch/s, ~14:45). `build_r3_singleA.sh`
packages A alone as soon as it finishes; `build_round3.sh` the A+B pair and
singles of A and C after that.

Built from the new mass-weighted train+valid teachers (all verified: exact
merges/exports, int8 within 0.00015 of fp32 on 75 sequences, latency of the
pairs 0.998x the 41m57s slim pair):

| zip (`submissions/`) | content | why |
|---|---|---|
| `031_TVm20_23_pair_slim_int8_2026-10-15.zip` | TVm20 (floor 0.1, aux) + TVm23 (floor 0, no aux) | the E0+D (0.6626) design with mass-weighted members |
| `031_TVm21_25_pair_slim_int8_2026-10-15.zip` | TVm21 (floor 0.1, aux) + TVm25 (floor 0.3, no aux) | second draw, most different recipes |
| `031_TVm20_single_slim_int8_2026-10-15.zip` | TVm20 alone | E0's recipe + mass weighting: platform test of the +0.0011 (E0 scored 0.6605) |
| `031_R2A_TVm23_pair_slim_int8_2026-10-15.zip` | round-2 student A (0.6628 alone) + TVm23 | a student plus a model that is not one of its teachers |

### Platform results 2026-10-15 - no improvement

| zip | platform | runtime | reference |
|---|---|---|---|
| `031_TVm20_23_pair` | 0.6622 (2EVD6NHZ) | 40m11s | E0+D 0.6626, E0+E1 0.6621 |
| `031_R2A_TVm23_pair` | 0.6621 (1HRY6EIC) | 40m1s | student A alone 0.6628 |
| `031_TVm21_25_pair` | 0.6616 (28Q0XAPH) | 40m22s | |
| `031_TVm20_single` | 0.6610 (20A8QZW3) | 26m42s | E0 (same recipe, no mass weighting) 0.6605 |

- **Mass weighting is not visible on the platform:** single +0.0005 vs E0,
  pairs at the same level as the old ones - all inside seed noise (0.0013).
  A local +0.0011 is below what the public score can resolve.
- **Adding an independent model to a student lowers it** (0.6621 vs 0.6628):
  the pair gives the single model half the weight, undoing the student's
  averaging.
- Everything in this family now lands at 0.6616-0.6632. The public score
  cannot separate variants whose local difference is ~0.001; only a change
  worth >= ~0.003 locally will show. More recombinations of the same models
  are not worth slots.

### NEW BEST 2026-10-15: round-3 single student = 0.6642

`031_R3h128_A_single_slim_int8_2026-10-15.zip`: **0.6642** (ID 15NFZS2X),
27m22s, 623 MB. One width-128 student (floor 0.1, aux, mass-weighted,
teacher targets only, 24 epochs) of TWELVE train+valid teachers: the four old
ones (E0, E1, D, C) + the eight mass-weighted TVm20-27 with mixed recipes.

- +0.0014 over the same student design with four teachers (round-2 single
  0.6628), +0.0010 over the previous best (round-2 pair 0.6632), +0.0037
  over the best single non-distilled model (0.6605) - in ONE 27-minute model.
- The same eight teachers as merged PAIRS only reached 0.6616-0.6622 this
  morning; distilled together into one student they lift it well above. The
  morning's "plateau" verdict was about recombining a few models; it does not
  hold for the teacher-pool lever, which keeps paying.
- So: (1) the next round should grow the train+valid teacher pool again
  (more models, more diverse: recipes AND architectures), (2) keep single
  students as the submission form (cheapest, and merging students adds
  nothing), (3) the round-3 pair A+B and single C (~15:15) will tell whether
  this 0.6642 is the recipe or a lucky draw.

### Round 4 started (2026-10-25 20:20, unattended)

Gap 2026-10-15 -> 10-25: nothing ran (round-3 pair A+B and single C were
packaged on 10-15 and sent to the user's machine on 10-25; results pending).

Teacher diversity by ARCHITECTURE, not just recipe. New code: `model.py`
`NonlinearGroupedEncoderRNNModel` (experiment 012's 5-way nonlinear expansion
of all 112 inputs, LSTM or GRU core, optional aux head) and `build_model()`;
`train.py --arch {lean,nl_lstm,nl_gru}` (recorded in config.json);
`teacher_preds.py` builds any of them. Smoke-tested on CPU (train + mixed
teacher scoring). Parameters: lean 248,639, nl_lstm 269,951, nl_gru 207,487.

Ten new train+valid teachers (mass-weighted, 14 epochs, mixed recipes; six on
server B via `run_round4.sh`, four on server A via `launch_round4_server A.sh`):
`TV4_nll_a/b/c` (012-LSTM), `TV4_nlg_a/b/c` (012-GRU), `TV4_h160_a/b` (lean,
hidden 160), `TV4_L1_a/b` (lean, one layer); seeds 40-49; floors 0 / 0.1 /
0.3, with and without the aux head. Then soft targets from all TWENTY-TWO
train+valid teachers (`soft/*_TV22.npy`) and three lean width-128 single
students `R4s_A/B/C` (teacher targets only, mass-weighted, 28 epochs, LR
halved at 15 / 21 / 25). Local `build_round4.sh` packages each student as a
single zip (`031_R4s_<X>_single_slim_int8_<date>.zip`) as soon as it finishes.
Speeds at start: 1.2-2.8 s/batch (both servers shared with other users), so
teachers ~1-1.5 days, students ~1 day after that.

Also started (CPU): `analysis/scripts/target_horizon_probe.py` -> which
future quantity / horizon the targets are (step 1 of the auxiliary-targets idea).

### Round-3 replicate results (submitted 2026-10-25; the platform now shows 5 decimals)

| zip | platform | runtime |
|---|---|---|
| `031_R3h128_A_single` (10-15) | 0.6642 (15NFZS2X) | 27m22s |
| `031_R3h128_pair_AB` | 0.66387 (1XM0Y13Z); a second copy (22RJS5VM) was still scoring | 40m46s |
| `031_R3h128_C_single` | **0.66352 twice** (1LD02FKW, 0CA5N7ZM) | 27m46s / 28m36s |

- **The platform score is deterministic:** the same zip scored 0.66352 both
  times. All "noise" we see is between models (seeds, recipes), none is
  measurement - resubmitting a zip gives no new information.
- Round-3 students: A 0.6642, C 0.66352, pair A+B 0.66387 -> average ~0.6639,
  vs round 2's 0.6623 / 0.6628 / pair 0.6632 (~0.6628): **twelve teachers
  instead of four = ~+0.001**, and student A's 0.6642 was partly a good draw
  (students of one teacher pool spread by ~0.0007).
- Again the pair of students does not beat its better member.

### Target-horizon probe (2026-10-25, CPU; `analysis/results/target_horizon_probe.txt`)

300 train sequences: a least-squares fit of each target on ALL 112 future
changes x[t+k] - x[t] explains almost nothing - R^2 0.011 at k=1 rising
slowly to 0.046 at k=400; the best single feature (i0 price levels) never
exceeds |corr| 0.105. So the targets are NOT a simple (linear) function of
future input changes at any horizon up to 400 steps, even though a
bidirectional LSTM reads them off at WP 0.997: the relation is nonlinear
and/or longer-range (the inputs are rank-transformed, per a rival's notes).
There is no obvious hand-made auxiliary horizon target; that idea would need
a learned one (e.g. predicting future FEATURES as an auxiliary task) and is
not a cheap next step. Parked.

### Ideas from the 2026-10-25 web search (to discuss 2026-10-26)

1. Tick-aware output head: a rival's notes (krish-rRay23/wunderTrader) say the
   targets sit on a lattice of two tick sizes (~0.657, ~0.469), 22.6% exact
   zeros, 3-tick moves carry 28% of the metric weight -> predict a
   distribution over tick buckets, output the metric-optimal value. Check the
   lattice on our data first (CPU, minutes).
2. Out-of-fold (cross-fitted) teachers for distillation (arXiv 2104.09732):
   soft targets for a sequence only from teachers that did not train on it.
3. Teacher annealing / end on true labels (BAM!; a rival's log reports
   +0.0008 from head-only fine-tuning on scored rows).
4. Small inputs: position in sequence, first differences (rival log).
5. Limited-look-ahead privileged teacher - not recommended.

### 2026-10-26: tick-lattice check, teacher annealing, round-4 status

**Targets are discrete** (`analysis/scripts/target_lattice_probe.py`,
`analysis/results/target_lattice_probe.txt`, 3.98M rows of 200 train
sequences): only ~200 distinct values per target; 22% exact zeros; the most
frequent values are 0, -0.656868 (17%), -0.468536 (11%), -1.970603 (=3 x
0.656868, 6%), -1.405608 (=3 x 0.468536, 6%), +0.656868 (3.7%), +0.468536
(3.3%), -0.241330, -0.125433, ... (top 12 ~ 78% of rows). Strongly SKEWED:
negative ticks are ~5x more frequent than positive ones. t0 and t1 are never
both zero (corr -0.747). Metric weight by |y| ~ k x 0.657: k=0 2%, k=1 41-42%,
k=2 21-22%, k=3 26-28%, k=4 5%. The rival's lattice claim holds; a
categorical head over the ~40 most common values is feasible (idea #3).

**Teacher annealing** (idea #2): `train.py --distill-anneal-start E
--distill-final A_f` moves the distill weight linearly from --distill-weight to
A_f between epoch E and the last epoch (smoke-tested: 1.0 -> 0.667 -> 0.333 -> 0).
`R4s_A_anneal` = `R4s_A` exactly (22-teacher soft targets, floor 0.1, aux,
seed 0, 28 epochs) but annealed to true labels from epoch 14; trains on the
LOCAL 1080 Ti (log `logs/train_R4s_A_anneal_local.log`); `build_r4_anneal.sh`
packages it when done. Paired platform comparison with `R4s_A`.

Round 4: soft targets from 22 teachers done 11:05; `R4s_A` 2.9 batch/s (~19:30),
`R4s_B/C` 1.7 batch/s (~01:30).

2026-10-26 18:20 - at the user's request nothing of ours runs on the local
1080 Ti any more: the annealed student `R4s_A_anneal` was stopped there at
epoch 7 (log kept as `logs/train_R4s_A_anneal_local_STOPPED.log`, partial
weights deleted) and restarted from scratch on server A GPU 1
(`launch_r4_anneal_remote.sh`, 2.35 batch/s, done ~05:00 on 10-27).
`build_r4_anneal.sh` now waits for it on the server like `build_round4.sh`.
(Local CPU is still used for packaging zips: export, int8, checks.)

### Round 4 finished (2026-10-27 07:26) - four single-student zips

All built unattended (exact slim export, int8 vs fp32 checked on 75 valid
sequences, 1,101 predictions verified from each extracted zip):
`031_R4s_A_single_slim_int8_2026-10-27.zip` (floor 0.1, aux, seed 0),
`031_R4s_B_single_slim_int8_2026-10-27.zip` (floor 0, seed 1),
`031_R4s_C_single_slim_int8_2026-10-27.zip` (floor 0.3, aux, seed 2),
`031_R4s_A_anneal_single_slim_int8_2026-10-27.zip` (= A, annealed to true
labels from epoch 14; trained on server A). Teachers: twenty-two train+valid
models of four families. Each ~27 min on the platform. Not yet submitted.

### Round-4 platform results (2026-10-27) - no gain, and annealing fails badly

| zip | platform | runtime |
|---|---|---|
| `031_R4s_B_single` (floor 0, seed 1) | 0.66341 (265B77FD) | 26m57s |
| `031_R4s_A_single` (floor 0.1, aux, seed 0) | 0.66257 (1DWR2KLG) | 27m32s |
| `031_R4s_C_single` (floor 0.3, aux, seed 2) | 0.66250 (0SH7WSBK) | 27m7s |
| `031_R4s_A_anneal_single` int8 | **0.61641** (1GN5OKST) | 27m41s |
| `031_R4s_A_anneal_single` fp32 | **0.61613** (0SZNEUD6) | 33m57s |

- **Twenty-two teachers of four families did NOT beat twelve lean ones.**
  Round-4 students average 0.66283 (0.6625-0.6634) vs round 3's ~0.6639
  (A 0.6642, C 0.66352, pair 0.66387): about -0.001, just beyond the
  ~0.0009 spread of students within a round. Candidate causes, not yet
  separated: (a) the new families teach functions the lean student cannot
  represent, or are weaker, and equal weighting dilutes the twelve good
  lean teachers; (b) the student schedule changed (28 epochs, LR halved at
  15/21/25, vs 24 epochs at 13/18/22).
- **Teacher annealing is catastrophic: -0.046** (0.6164 vs 0.6626 for the
  same student without it; int8 and fp32 equal, so not a quantisation
  problem). Fourteen more epochs on hard labels (train+valid) after the
  student was already strong overfit badly - consistent with ordinary runs
  peaking at epoch 11-13 and declining after, and with its inflated in-sample
  valid score (0.759 vs ~0.716 for the others). Never anneal a full student
  to hard labels; at most a short, low-LR final epoch, if ever. Closed.
- Best remains round-3 student A, 0.6642.

### Round-5 tests started (2026-10-28, user go-ahead) - `run_round5_tests.sh`

1. **Diagnostic student `R5d_B_TV12`** (server B GPU 0): round-4 student B's
   exact recipe (28 epochs, LR halved 15/21/25, floor 0, no aux, seed 1,
   mass-weighted) on round 3's TWELVE-teacher soft targets. Near 0.664 -> the
   round-4 teacher families hurt; near 0.663 -> the schedule did. Packaged by
   local `build_r5d.sh`.
2. **Out-of-fold teachers, honest local test.** New code: `train.py
   --exclude-fold F --n-folds 5` (drops sequences with index % 5 == F) and
   `teacher_preds.py --oof` with models given as `CKPT:WEIGHTS@F` (a
   sequence's soft target = mean of only the teachers that never saw it;
   checked: five copies of one model at folds 0-4 reproduce that model to
   6e-05). Ten train-only lean teachers `F{0..4}_s{50..59}` (T_seed recipe,
   two per fold; 3 on server B, 7 on server A); the pipeline (server A,
   `logs/round5_tests.log`) then scores the ten on valid (ceiling, plain mean),
   writes `soft/train_OOF10.npy` and trains `S_OOF` with S1_a100's recipe.
   Baseline: S1_a100 0.689331 (ten in-sample teachers on all of train).

### Round-5 tests, interim (2026-10-29 10:25)

- **Diagnostic student packaged:** `031_R5d_B_TV12_single_slim_int8_2026-10-29.zip`
  (round-4 recipe B on round-3's twelve-teacher targets; int8 0.715919 ->
  0.715376 on 75 sequences; not yet submitted). Compare with R4s_B 0.66341.
- **Out-of-fold teachers:** ten train-only teachers on 80% of train each;
  singles 0.685163-0.686921, ten-model mean on valid **0.689630** vs 0.689934
  for the ten ordinary teachers on all of train (-0.0003: the 20% less data).
- **Out-of-fold student `S_OOF`** (S1_a100's recipe on out-of-fold targets),
  epoch 15: EMA 0.687858, still at full LR. S1_a100 at the same epochs was
  ahead (0.688128 by epoch 8, 0.688992 by epoch 17; best 0.689331). So far
  out-of-fold targets are NOT better; final number pending.

### Round-5 teachers started (2026-10-29 10:50, user: "start whatever you can")

`run_round5_teachers.sh` on server B: eight MORE lean train+valid teachers
`TVm60-67` (mass-weighted; floors 0 / 0.1 / 0.2 / 0.3, with and without the
aux head; seeds 60-67), then soft targets from all TWENTY lean train+valid
teachers (E0, E1, D, C, TVm20-27, TVm60-67) -> `soft/*_TV20L.npy`. Reason:
4 -> 12 lean teachers raised the student (~0.6628 -> ~0.6639) while round 4's
other architectures did not. Students are launched by hand once the
diagnostic zip (`031_R5d_B_TV12`) has chosen the schedule (24 vs 28 epochs).

### Diagnostic result (2026-10-29): the round-4 drop came from the SCHEDULE

`031_R5d_B_TV12_single_slim_int8`: **0.66234** (07VDLQEG, 28m1s).

| student | teachers | schedule | recipe | platform |
|---|---|---|---|---|
| R3 A | 12 lean | 24 ep, LR /2 at 13/18/22 | floor 0.1, aux | 0.6642 |
| R3 C | 12 lean | 24 ep | floor 0.3, aux | 0.66352 |
| R4 A | 22 (4 families) | 28 ep, LR /2 at 15/21/25 | floor 0.1, aux | 0.66257 |
| R4 C | 22 | 28 ep | floor 0.3, aux | 0.66250 |
| R4 B | 22 | 28 ep | floor 0, no aux | 0.66341 |
| **R5d B** | **12 lean** | **28 ep** | floor 0, no aux | **0.66234** |

With the 28-epoch schedule, twelve teachers did NOT beat twenty-two (0.66234
vs 0.66341, inside student noise ~0.0009); the same recipes A and C lost
0.0010-0.0016 when the schedule changed. So the longer schedule (or its later
LR drops) is what hurt, not the bigger teacher pool. Round 5 therefore uses
round 3's 24-epoch schedule with a 20-lean-teacher pool:
`run_round5_students.sh` (server B) starts `R5s_A` (floor 0.1, aux, seed 0),
`R5s_C` (0.3, aux, 2), `R5s_A2` (0.1, aux, 3), `R5s_D` (0.2, aux, 4) as soon as
the soft targets exist; local `build_round5.sh` packages each one.

### 2026-10-29 - width test W256_A (local 1080 Ti, user-allowed one-off)

User asked why we do not spend more of the 60-minute budget on a heavier model. Single-thread int8 slim latency of
untrained exports, alternating with R3h128_A's int8 graph (27m22s on the platform): **hidden 192 = 1.16x (~32 min),
hidden 256 = 1.35x (~37 min)**. Launched `run_width_local.sh`: R3 A's recipe and TV12 soft targets unchanged, hidden 256,
24 epochs (~1.1 s/batch, ~58 min/epoch locally -> done ~11:30 on 10-30). Prior evidence against width: 029 h160 +0.0004
(noise); the best train-only student (0.689331) sits only ~0.0006 below its teacher ensemble (0.689934) locally. It trains
on train+valid, so there is no honest local score: the platform submission IS the test (vs R3 A 0.6642, noise ~0.001).

### 2026-10-30 - S_OOF final (out-of-fold teacher targets): NOT adopted

`S_OOF` (train-only student on 5-fold out-of-fold soft targets `soft/train_OOF10.npy`, 30 epochs, server A) finished:
**best EMA 0.688101 / raw 0.688201** vs S1_a100 (same recipe, in-sample T10 targets) **0.689331** -> **-0.0012**.
Out-of-fold targets cost accuracy (their teacher ensemble was also weaker, 0.689630 vs 0.689934). Closed; round 5 keeps
in-sample train+valid teachers.
User chose to submit it anyway as a platform test of the out-of-fold idea (compare with the train-only S1_a100 single,
platform 0.6597): `031_S_OOF_single_slim_int8_2026-10-30.zip` (best_ema.pt, slim int8, 0.24 MB, ZIP OK; sent to ASUS,
MD5 e183dd81... matches). Only a platform score clearly above 0.6597 would reopen out-of-fold teachers.

### Ideas from the 2026-10-30 web search

Main source: a rival's public summary (sunArise blog, jy447.github.io, 2026-10-24/25; public 0.664539 = rank 66 of
5,334 then - level with our 0.6642, so the board is tightly packed and +0.001 is worth many places). Their ledger:
+0.0125 from LABELS, +0.0040 from combining models, ~0 from architecture or features.
1. **Valid-only fine-tune stage** (their biggest step, +0.0071 public): after pretraining, fine-tune ALL parameters on
   the 1,873 valid sequences only (scored rows weight 1, unscored 0.1; target 0.2 x true + 0.8 x cross-fitted soft
   label; 6 epochs, batch 32, lr 2e-4, 1e-4 looked better; two seeds averaged in WEIGHT space). Their claim: train and
   valid are different market regimes (AUC 0.993 separable) and train-regime data "does not transfer to test". Ours
   already weights valid rows by the true mask inside TV training, but has no valid-only stage. Honest local test:
   S1_a100 (train-only) fine-tuned on 4/5 of valid with T10 soft targets (teachers never saw valid), scored on the 5th.
2. **Look-ahead teacher** (+0.0012 public): causal GRU + gated dilated convolutions over the next 25/50 rows
   (valid 0.81/0.89); students distil it together with the causal teachers. Our 029 bidirectional teacher saw the
   whole future (0.997, labels ~= truth); a SHORT window is the difference.
3. **Different-structure ensemble members** (+0.002 public; same-structure seeds only +0.0003, matching our pairs):
   e.g. our lean LSTM student + a small GRU/fusion member at 0.25-0.5 weight. Needs a latency check (40-min cap).
Smaller: tick-unit routing +0.0007 (four tick units, fixed per sequence; the large unit 0.657 carries 47.7% of the
metric weight at corr 0.44 vs 0.70); weight soups +0.0002-0.0005. Negative for them: bid/ask mirror and instrument
swap augmentation, SWA, LwF, minGRU/linear recurrences, per-unit scaling. Other public repos (wunderTrader 0.6517,
MarkBrezina log ~0.646 local) add nothing new beyond head-only scored fine-tuning (+0.0008) already noted.

**S_OOF platform (2026-10-30): 0.65893** (2BFF1YN5, 25m52s) vs the train-only ten-seed student S1_a100's 0.6597
(-0.0008; locally -0.0012). Out-of-fold teacher targets do not help on the hidden data either. Closed for good.

### 2026-10-30 - valid-only fine-tune stage: honest local test started (server A, `run_validft.sh`, log `logs/validft.log`)

`train.py` new switches: `--init-from`, `--train-true-mask`, `--blend-targets`, `--valid-fold`, `--extra-exclude-fold`,
`--weight-decay`, `--clip-grad`; `--lr-milestones` now also drives the LR when validation is on. Held-out fold = valid
index % 5 == 0 (375 sequences), never trained on. Fine-tune = rival's recipe: valid folds 1-4 only, true mask (scored 1,
unscored 0.1), target 0.2 x y + 0.8 x T10 soft (train-only teachers -> honest), 6 epochs, AdamW wd 1e-4, clip 1.0, no
warmup, EMA 0.98; settings a = batch 32 lr 2e-4, b = batch 32 lr 1e-4, c = batch 8 lr 1e-4.
- A: from S1_a100 (train-only; **fold-0 start 0.688557**) -> FT_S1_a/b/c (~25 min each, GPU 0).
- B: control `TVf0` = R3 A recipe on train + valid folds 1-4 (scored on fold 0 each epoch), then FT_TVf0_a/b/c. B is the
  real question (our students already train on valid); ~16 h on the shared GPU 1 -> ~06:00 on 10-31.

**Part A result (2026-10-30 11:50): the valid-only fine-tune does NOT help our student.** Fold-0 WP from S1_a100's
0.688557: every setting dips at epoch 0-1 and recovers to just below the start after 6 epochs - a (b32, lr 2e-4)
0.688269 (EMA best 0.688240), b (b32, 1e-4) 0.687976, c (b8, 1e-4) 0.688468 (EMA 0.688426). Even a model that NEVER saw
valid gains nothing from 1,498 valid sequences with honest labels. Most likely reason: the rival's +0.007 came from
weighting by the true scoring mask for the first time - we already get that through the scored-probability weights
(026/028) and the true-mask weights on valid in train+valid training. Part B (TVf0) keeps running as the honest
fold-0 score of the train+valid recipe, but its fine-tunes are expected to show the same.

### W256_A platform (2026-10-30): **0.66460 - NEW BEST** (21YIT9UZ, 39m6s)

Same recipe, teachers and seed as R3h128_A (0.6642, 27m22s); only the width changed (128 -> 256): **+0.0004**. That is
inside the spread between students of one teacher pool (~0.0007), so width is not yet proven - but it is the first
wider model that did not lose. Runtime 39m6s vs the 37 min estimated (1.43x R3 A's platform time vs 1.35x locally):
at our 40-minute ceiling, so nothing wider; an h256 finalist carries more timeout risk than an h128 one.

### 2026-10-30 evening - push for the last 16 days: look-ahead teachers (round 6)

- `TVf0` (fine-tune control) stopped at epoch 11 to free server A GPU 1 (part A already answered the question).
- `model.py`: `LookaheadLeanModel` (`--arch la25 | la50`): the lean causal model + a branch reading rows t..t+K
  ([encoded x, encoded diff] -> 64 -> gated dilated forward convs, kernel 2, dilations 1,2,4,8,10 / 1,2,4,8,16,19 ->
  zero-initialised projection added to the LSTM output). Checked: rows <= t+K move the output at t, rows > t+K do not
  (float noise only). Teacher only, never exported.
- `run_lookahead.sh GPU ARCH`: teacher recipe (true labels, floor 0.1, aux 0.2, mass, 14 epochs, LR halved at 10) on
  train + valid folds 1-4, fold 0 scored every epoch (rival: ~0.81 for K=25, ~0.89 for K=50). la25 on server A GPU 1
  (~21 min/epoch -> ~22:30), la50 on server B GPU 1 (~34 min/epoch -> ~01:30).
- `run_la_students.sh GPU K` (queued behind each teacher): its soft targets -> `blend_soft.py` 0.5 x TV20L + 0.5 x LA
  (the rival's weighting) -> student `R6_LA<K>_A` = R5s_A's recipe exactly on the blend (paired with R5s_A).
  `build_round6.sh` packages both (~14:00-16:00 on 10-31).

### Round 5 platform (2026-10-30): the 20-teacher pool is WORSE - back to round 3's twelve teachers

| recipe | TV12 (R3, 24 ep) | TV22 (R4, 28 ep) | TV20L (R5, 24 ep) |
|---|---|---|---|
| A (floor 0.1, aux) | **0.6642** | 0.66257 | 0.66314 (06UAUZ51, 27m47s) |
| C (floor 0.3, aux) | **0.66352** | 0.66250 | 0.66221 (0F2GCR40, 27m2s) |

All four paired comparisons favour the twelve-teacher targets by 0.0010-0.0016; only the single 28-epoch diagnostic
(recipe B) pointed the other way, inside noise. So the pool size/composition, not just the schedule, cost round 4.
Why is open (more, smoother teachers regress the targets toward the mean? the TVm60-67 recipe?) - but TV12 is the
platform-best target set and everything new now uses it:
- `W256_R5A` (h256 on TV20L) stopped at epoch 3; replaced by **`W256_C`** = R3h128_C recipe at h256 on TV12 (server A
  GPU 0, `run_w256_c.sh`; pairs with R3 C 0.66352) and **`W256_A2`** = W256_A's recipe, seed 3 (server B GPU 0,
  `run_w256_a2.sh`; seed replicate of 0.66460). `build_w256c.sh` packages both (~15:00 / ~20:30 on 10-31).
- Look-ahead students now blend 0.5 x **TV12** + 0.5 x LA and use R3h128_A's recipe (pair with 0.6642).
- R5s_A2 / R5s_D zips exist (TV20L pool) - low value now.

**R3h128_B single (2026-10-30): 0.66281** (27NXZTZ9, 26m27s). Recipe B (floor 0, no aux) on TV12, 24 epochs. Recipe B
does NOT follow the pool pattern: TV12 24 ep 0.66281, TV12 28 ep 0.66234, TV22 28 ep 0.66341. Means over recipes:
TV12 (A/B/C) 0.66351, TV22 (A/B/C) 0.66283, TV20L (A/C) 0.66268 -> TV12 still best on average (~+0.0007), but the
pool effect is smaller and less uniform than the A/C pairs suggested. Recipe A (floor 0.1 + aux) is the best recipe in
two of three pools: new students use it (look-ahead students do; W256_C is the recipe-C replicate).

### Round 6 look-ahead results so far (2026-10-31 morning)

Teachers (held-out valid fold 0, EMA, final epoch 13): **la25 0.775507, la50 0.822316** (whole valid, mostly
in-sample: 0.786927 / 0.832038); rival reported ~0.81 / ~0.89 on their easier subset. Students `R6_LA25_A` (done
07:47) and `R6_LA50_A` (05:53): R3h128_A's recipe on 0.5 x TV12 + 0.5 x LA targets; zips
`031_R6_LA25_A_single_slim_int8_2026-10-31.zip`, `031_R6_LA50_A_single_slim_int8_2026-10-31.zip` (ZIP OK, int8 vs fp32
mean diff 0.012 / 0.013). Platform pair: R3h128_A 0.6642.

2026-10-31 afternoon: `W256_A2` (seed 3; finished 14:24) and `W256_C` (recipe C; 15:10) packaged (ZIP OK, int8 ~95
us/step locally = W256_A's speed, ~39 min on the platform). server B then idle -> speculatively started (before the R6
platform scores) **`W256_LA50_A`** (GPU 0) and **`W256_LA25_A`** (GPU 1) = W256_A's recipe on the 0.5 x TV12 + 0.5 x LA
targets (`run_w256_la.sh GPU K`; ~45-50 min/epoch -> ~12:00-13:00 on 11-01; `build_w256la.sh` packages them).
Stop whichever look-ahead variant loses on the platform.

### Round 6 platform (2026-10-31): LOOK-AHEAD LABELS WORK - new best 0.66544

| zip | platform | runtime | paired with |
|---|---|---|---|
| `R6_LA25_A` (0.5 TV12 + 0.5 la25) | **0.66544** (1EFLZV48) NEW BEST | 28m21s | R3h128_A 0.6642: **+0.0012** |
| `R6_LA50_A` (0.5 TV12 + 0.5 la50) | 0.66503 (2EM19HMY) | 27m43s | R3h128_A 0.6642: +0.0008 |
| `W256_A2` (TV12, recipe A, seed 3) | 0.66426 (1HM1M1NV) | 39m46s | W256_A 0.66460 (seed spread 0.0003) |
| `W256_C` (TV12, recipe C) | 0.66426 (23AEVMP0) | 39m51s | R3h128_C 0.66352: +0.0007 |

- Look-ahead labels: +0.0008 / +0.0012 at NO runtime cost - exactly the rival's +0.0012. The shorter window (25) is
  at least as good as 50.
- Width 256: three TV12 students 0.66460 / 0.66426 / 0.66426 (mean 0.66437) vs width 128 A/C 0.6642 / 0.66352:
  **~+0.0005**, at 39m46-39m51 (our ceiling). Real but small; look-ahead is worth twice as much for free.

**Round 7 (started 2026-10-31 20:05, server A, `run_r7.sh`; packaged by `build_round7.sh`)** - push look-ahead, all
R3h128_A's recipe at width 128:
- `R7_LAmix_A`: 0.5 TV12 + 0.25 LA25 + 0.25 LA50 (two look-ahead teachers averaged, same look-ahead weight) - GPU 0
- `R7_LA25w75_A`: 0.25 TV12 + 0.75 LA25 (more look-ahead weight) - GPU 1
- `la12` teacher (dilations 1,2,4,5: rows t..t+12, checked) -> `R6_LA12_A` (0.5 TV12 + 0.5 LA12) - GPU 0
Students ~35 min/epoch -> ~11:00 on 11-01; la12 teacher ~03:30, its student ~18:00 on 11-01. Still on server B:
`W256_LA50_A`, `W256_LA25_A` (~12:00-13:00 on 11-01).

### 2026-10-31 night - round 8: weight soup + more look-ahead teachers (server B)

- `train.py --save-state / --resume-state` (full state: weights, Adam, EMA, epoch) and `soup.py` (average branches'
  EMA weights -> checkpoint dir for export_slim). CPU smoke test: branch resumes at the right epoch/LR, branches
  diverge, soup = exact mean, exports to the 13-node slim graph.
- `run_soup.sh 0`: trunk = R6_LA25_A's recipe (0.5 TV12 + 0.5 LA25) epochs 0-12, then branches seeds 1/2/3 for
  epochs 13-23 -> `031_R8_SOUP3_LA25_A` (soup, one-model runtime) and control `031_R8_SOUPb1_LA25_A` (branch 1 alone).
- `run_la25_seeds.sh 1`: la25 teachers seeds 71 (GPU 1) / 72 (GPU 0) -> soft = mean of THREE la25 teachers ->
  0.5 TV12 + 0.5 LA25x3 -> `R8_LA25x3_A` (pairs with R6_LA25_A 0.66544).
- `build_round8.sh` packages all. server B is crowded (our 5 jobs + another user's 29 GB job on GPU 1): W256_LA25/50
  slowed to 1.5-1.8 s/batch -> ~19:00-21:00 on 11-01; soup and R8 ~20:00-23:00 on 11-01.

2026-10-31 21:00: built `031_R6_pair_LA25_LA50_slim_int8_2026-10-31.zip` for the day's last slot = merged pair of the two
look-ahead students (merge exact to 5e-7; 100.5 us/step locally = R3 pair AB's speed, which ran 40m46s). Members learned
from DIFFERENT look-ahead teachers, so they should be more diverse than our earlier seed pairs (pairs so far scored
~member mean + 0.0004 -> expect ~0.6655-0.6660 vs LA25's 0.66544).

**R6 pair LA25+LA50: 0.66592 - NEW BEST** (0IVH8F8M, **42m31s**). Members 0.66544 / 0.66503 (mean 0.66524): the pair is
**+0.0007 over its members' mean and +0.0005 over the better member**. Our earlier pairs of same-teacher seeds gave
~+0.0004 over the mean and never beat the better member - students that learned from DIFFERENT label sources are more
diverse (the rival's +0.002 from different structures is the same effect). At the same ~40 min, a diverse pair
(0.66592) beats every width-256 single (0.66426-0.66460): spend the runtime budget on diverse pairs, not width.
Runtime caveat: 42m31s vs R3 pair AB's 40m46s for the same graph size - above our 40-min ceiling; still 17 min
under the limit, but a pair finalist carries more timeout risk than a 28-min single.
- 2026-10-31 23:00: W256_LA50_A stopped at epoch 7 (the pair result: diverse pairs beat width at equal runtime; server B overloaded). W256_LA25_A continues.

### 2026-11-01 morning

Packaged overnight: `R7_LAmix_A` (06:13), `R6_LA12_A` (09:02; la12 teacher held-out 0.741101). la25 seed teachers
s71 / s72 held-out 0.772800 / 0.776322 (s70: 0.775507) -> `R8_LA25x3_A` training since 08:46. Soup: trunk done 03:36,
branches at epoch 22/24. `build_pairs.sh RUN1 RUN2 ...` (local CPU, seconds per pair) built candidate pairs
`031_pair_LA25_A_LA12_A`, `031_pair_LA50_A_LA12_A`, `031_pair_LA25_A_LAmix_A` (all ZIP OK, merge exact to 5e-7).

**R7_LAmix_A 0.66497** (02JVZKQH, 26m2s) and **R6_LA12_A 0.66554** (0LQ89L21, 27m37s; best single). Look-ahead singles
by window: 12 rows 0.66554, 25 rows 0.66544, 50 rows 0.66503; mixing the 25- and 50-row teachers into ONE target
0.66497 - no gain from averaging look-ahead teachers inside the labels (contrast: averaging two STUDENTS of different
labels, the LA25+LA50 pair, gave +0.0007). Shorter windows are at least as good although their teachers are weaker
(held-out 0.741 / 0.776 / 0.822): the benefit is not "better labels = more accurate teacher".

**Round 9 (2026-11-01 11:08, server A; `run_r9.sh`, packaged by `build_round9.sh`)**: la6 teacher (`--arch la6`,
dilations 1,2,3: rows t..t+6, checked) -> `R6_LA6_A` (~01:30 on 11-02), and `R9_LA12_C` = R3h128_C's recipe on the
LA12 targets (restarted on GPU 0 after an OOM on GPU 1, which another user fills with ~43 GB; `run_r9_la12c.sh`,
~21:00). Purpose: more distinct width-128 members for pairs. `R7_LA25w75_A` packaged 10:56.

**Pair LA25+LA12 0.66610 - NEW BEST** (0G9BQSE6, 40m36s): members 0.66544 / 0.66554 (mean 0.66549) -> +0.0006 over the
mean, +0.0006 over the better member (LA25+LA50 pair: +0.0007 / +0.0005). The diverse-pair gain replicates.
**R7_LA25w75_A 0.66383** (0TIU1KSH, 28m41s): 75% look-ahead weight is -0.0016 vs 50% (R6_LA25_A 0.66544) - too much.
The optimum may be BELOW 0.5: next test 0.3 on the LA12 targets.
- Round 10 queued (server B GPU 0 after the soup): R10_LA12w30_A = 0.7 TV12 + 0.3 LA12 (run_r10.sh, build_round10.sh).

**Soup (R8_SOUP3_LA25_A): 0.66528** (2630P72E, 28m4s) vs R6_LA25_A (same targets and recipe, a full single run)
0.66544: -0.0002, inside seed noise. Averaging three late branches gives no visible gain on the platform (the rival
reported +0.0002-0.0005, below our noise floor). Not pursued further; the branch-1 control zip stays unsubmitted.
Diverse PAIRS (+0.0006-0.0007, twice) remain the only ensembling that pays.

### 2026-11-02 morning - round 9-10 + W256_LA25 + R8 zips ready; round 11 started

Packaged overnight (all ZIP OK): `R9_LA12_C` (21:07), `R10_LA12w30_A` (21:40), `R6_LA6_A` (02:22; la6 teacher held-out
0.717459), `W256_LA25_A` (03:58; 94 us/step ~ 40 min), `R8_LA25x3_A` (05:09). All five sent for submission.
Round 11 (server B GPU 0, `run_student.sh GPU NAME TARGETS FLOOR SEED [HIDDEN]`, packaged by `build_round11.sh`): seed
replicates `R11_LA12_A2` / `R11_LA25_A2` (seed 3) of the two best pair members - finalists should be chosen on recipe
averages, not single draws (the private leaderboard is a different hidden set).

### 2026-11-02 - leaderboard check and round 12; inference-engine test (parked)

Leaderboard (user screenshot, 11-02): #1 0.67586, #8 0.67213, #20 0.66938; our best 0.66610 is outside the top 20
(gap +0.0033 to #20, +0.0060 to #8).
- Round 12 (server A, `run_student.sh`, env ARCH / EXTRA_ARGS): `R12_LA12_GRU` = `--arch lean_gru` (new `LeanGRUModel`:
  lean encoder + head, GRU core) on the LA12 targets - a structurally different pair member (needs a GRU export path
  before it can ship); `R12_LA12_V3` = LA12 targets with `--extra-repeat 3` (valid sequences shown 3x per epoch: the
  rival found train and valid are different regimes and train data does not transfer to test). `build_round12.sh`
  packages V3. (A sed edit corrupted run_student.sh's last line and both first launches exited at once; rewritten.)
- `numba_engine.py` (pack / check): fused numba kernels for K lean LSTM / GRU members in one call, exact vs torch
  (4e-7). Same core, per step: ONNX int8 single 64 us, merged pair 95 us, merged TRIPLE 136 us (~60 min on the
  platform: too slow); numba fp32 single 59 us, two members 109 us. ONNX int8 is already efficient (each extra member
  ~30 us); fp32 numba moves 4x more weight bytes. Parked: with our tools the ensemble is capped at a PAIR.
- `export_slim.py --multi SPEC...` now also takes GRU members (`_gru_onnx`: PyTorch r,z,n -> ONNX z,r,h,
  `linear_before_reset=1`): an LSTM + GRU graph matches the mean of the members to 5e-7. Per step (same core): mixed
  LSTM-int8 + GRU-fp32 114.7 us vs merged LSTM pair 97.9 us -> ~47-49 min on the platform (pairs ran 40.6-42.5 min).
  `build_gru.sh` packages `R12_LA12_GRU` alone and as `031_mixpair_LA25_A_LA12_GRU` when it finishes (~20:00).
- 2026-11-02 12:20: user allowed the LOCAL 1080 Ti for a few days. `run_local_student.sh` (same recipe family,
  checkpoints straight into results/). First job: `R12_LA25_GRU` (lean_gru on the LA25 targets, 2.55 batch/s ->
  ~20:30); `build_local_gru.sh` packages it alone and as `031_mixpair_LA12_A_LA25_GRU` (LSTM LA12 + GRU LA25:
  different structure AND labels).

### 2026-11-02 platform: the look-ahead recipe is at a peak

| zip | platform | runtime | vs |
|---|---|---|---|
| `W256_LA25_A` | **0.66580** (1PXXR5G4) best single | 39m51s | LA25 width 128 0.66544: +0.0004 (width again ~+0.0004) |
| `R8_LA25x3_A` (three la25 teachers) | 0.66512 (1RBKYSRK) | 29m6s | one la25 teacher 0.66544: -0.0003 (noise; no gain) |
| `R9_LA12_C` (recipe C) | 0.66427 (217F6024) | 28m51s | LA12 recipe A 0.66554: -0.0013 |
| `R10_LA12w30_A` (look-ahead weight 0.3) | 0.66396 (0A07W1YS) | 28m7s | LA12 at 0.5 0.66554: -0.0016 |
| `R6_LA6_A` (6-row window) | 0.66362 (1KQBWPMF) | 28m21s | LA12 0.66554: -0.0019 |

Window: 6 rows 0.66362 / 12 rows 0.66554 / 25 rows 0.66544 / 50 rows 0.66503 -> peak at 12-25. Look-ahead weight: 0.3
0.66396 / 0.5 0.66554 / 0.75 0.66383 (LA25) -> 0.5. Recipe A > C again. Averaging look-ahead teachers (LAmix, LA25x3)
never helps. Caveat: every variant lands 0.0013-0.0019 BELOW LA12_A / LA25_A, so those two may be lucky draws -
the seed replicates R11_LA12_A2 / R11_LA25_A2 (tonight) tell how much.
- Local second job (2026-11-02 13:00): `R11_LA12_A3` (LA12 targets, recipe A, seed 4), packaged by `build_local_r11.sh`.

### 2026-11-02 afternoon - web search (top-8 push) and round 13: three untried architecture additions

Web search (Wunder Predictorium / RNN challenge winners, rival repos, Russian sources): no public winner write-ups,
no new rival posts since 10-25. The rival's full log adds: helpful = "a second nonlinear stage and depth made
trainable by residual connections"; neutral = capacity beyond 2x256, memory beyond ~500 rows, soups.
`model.py` `LeanXModel` (`--arch leanx_<flags>`): i = instrument-1 price+volume back in (dropped in 027 when the
model was much weaker; public lead-lag hint r=0.116), r = residual MLP after the LSTM (zero-init output), p = position
in the sequence [s, sin, cos] from step_in_seq (never tried). Streaming in halves == full pass. `export_slim.py
build_x` + `verify_x` (exact, 2e-8); `solution_slim.py` feeds `pos` from a precomputed table when the graph has it.
leanx_irp int8: 83.9 us/step (~37 min) vs 67 for the plain student. Students on the LA12 targets, recipe A, seed 0:
`R13_LA12_R`, `R13_LA12_P` (server B GPU 1), `R13_LA12_IRP` (server B GPU 0), `R13_LA12_I` (local; OOM on server B GPU 1).
`build_round13.sh` packages all four (~06:00-09:00 on 11-03).

**2026-11-02: rounds 12 and 13 (GRU members, valid oversampling, LeanX additions) are now EXPERIMENT 032**
(`experiments/032_architecture_diversity/`, zips `032_*`); the code stays here because the running jobs use it.
031 keeps the label work and the seed replicates R11.

**2026-11-03: R11_LA12_A2 / R11_LA25_A2 finished (weights saved 04:13-04:14, on schedule) but their training logs
were truncated afterwards**: run_student.sh was rewritten on 11-02 while these two bash processes were still running
it; after training, bash resumed reading the changed file mid-line, ran a fragment whose redirect truncated
logs/train_<N>.log, and exited 127 ("0.2: command not found"). Weights intact (24 epochs by timing); packaged by hand.
Lesson: never edit a shell script that running jobs were started from - write a new file name instead.
- 2026-11-03 10:00: all GPUs idle -> `W256_LA12_A` (width 256 on the LA12 targets, recipe A; finalist candidate next to W256_LA25_A 0.66580) on server A GPU 0, `build_w256la12.sh`.
