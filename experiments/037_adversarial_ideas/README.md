# 037: the three ideas from the generator/discriminator debate (2026-11-06)

The user asked for a 30-minute generator vs discriminator debate (two agents, 11:47-12:28). Nine candidates were
proposed; four were killed (teacher averaging: already lost as LA25x3/LAmix; compute routing by unit; strided
members: a stale row costs -0.004; event-clock inputs: the fusion GRU already had them). These three survived. Code
lives in `experiments/031_seed_ensemble_distillation/`.

## 1. IOBinding predict loop (`solution_iob.py`)

All buffers (features, two sets of recurrent state, prediction) are allocated once and bound; each row writes into the
bound features buffer and runs one of two pre-built bindings (state A -> B, B -> A); a new sequence zeroes the buffer
the next run reads. LeanX graphs with a `pos` input fall back to the plain path.

Local test on the 0.66658 graph (12,000 valid rows, 4 sequences, 3 switches, one pinned core, alternating):
**bit-identical outputs (max diff 0, None pattern identical); 130 -> 103 us/step (-21%).** The generator's faster
"IOB2" (raw `session._sess` binding) is ALSO exact - its 1.74 diff came from a test bug (a new model per row) - but
only ~2% faster and uses a private API, so it is not used.

Zip: `037_IOB_mixpair_fullLSTM_LA12_leanGRU_LA50_slim_int8_2026-11-06` = the 0.66658 model (same md5,
88917dbc...) + `solution_iob.py` (`make_zip.py --solution`). It must score exactly 0.66658; the runtime shows the
real saving (expected 41-46 min vs 50m47s). Only then do finalists switch to it.

## 2. Segment-end labels (`seg_soft.py`, run `R28_FULL_LSTM_SEG12`)

Targets change only on rows where an i0 price column changes (39 exceptions in 744k scored valid rows). So
y(t) = y(te), te = min(last row before the next i0 price change, t + 12), and the CAUSAL teacher pool read at te is a
look-ahead label that never sees the move settling the target; no new teacher. Honest check (train-only T10 teachers,
valid fold 0), 0.6569 unit: row t 0.454, seg12 0.512, seg25 0.548 (honest LA12 0.551); fine units barely move.
Discriminator's fixes, both applied: (a) te is not a stopping time, so on the 0.6569 unit the label's |y|-weighted mean
drifts (-0.760 -> -0.785): each unit's mean drift (seg - TV12, per target) is fitted on TRAIN and subtracted
(`soft/seg_drift_TV12_SEG12.json`); (b) look-ahead weight kept moderate (75% hurt before): label =
0.5 TV12 + 0.25 seg12 + 0.25 LA12 = 0.25 TV12 + 0.25 seg12 + 0.5 TV12_LA12.
Student: all-feature LSTM, R23 recipe, seed 0, on server B GPU 0 after the 034 seed replicate (`run_seg_student.sh`);
labels built locally (~5-6 h CPU), copied with checksums by `copy_seg_labels.sh`.
Gate: single >= 0.66637 (0.66597 + 0.0004) before a pair slot. Estimate +0.0002, P(> +0.001) ~12%.

## 3. Per-unit look-ahead window (`unit_window_soft.py`, run `R28_FULL_LSTM_LAU`)

LSTM label = TV12_LA25 on 0.6569-unit sequences (4,557 train / 697 valid), TV12_LA12 elsewhere; the GRU partner keeps
LA50. Per-unit honest teacher WP on 0.6569: LA12 0.545 -> LA25 0.602. Student on server B GPU 1 after the 034 teacher
pipeline (`run_lau_student.sh`). Estimate +0.0002.

`build_037.sh` (local) packages singles + mixpairs with the lean GRU LA50 (R16): `037_full_LSTM_{SEG12,LAU}_single`,
`037_mixpair_fullLSTM_{SEG12,LAU}_leanGRU_LA50`.

Discriminator's bottom line: even if all three work, ~+0.0005 in total - the top-8 gap (+0.006) needs something else.

## Status

- 2026-11-06 13:45: IOB zip built and verified; LAU labels on the servers (checksums match); SEG12 labels building
  (pass 1 at 4,000/10,607); both students queued on server B.

## Platform results (2026-11-07, submitted together at 10:30)

| zip | platform | runtime |
|---|---|---|
| `037_IOB_mixpair_fullLSTM_LA12_leanGRU_LA50` | 0.66658 (identical, as designed) | **61m22s - SLOWER than the plain 50m47s** |
| `037_mixpair_fullLSTM_s_SEG12_leanGRU_LA50` | 0.66619 (-0.00039) | 50m26s |

- **IOBinding is slower on the platform** despite -21% locally; the three other zips of the same graph shape submitted
  with it ran 50-53 min, so this is not platform load. Never ship IOBinding; never trust a new inference mechanism's
  local speed-up without a platform run. (The numba zips' fallback path is this IOBinding code.)
- Segment-end labels: -0.00039 (within pair noise ~0.0006, but no gain). Per-unit window (LAU) result pending.
