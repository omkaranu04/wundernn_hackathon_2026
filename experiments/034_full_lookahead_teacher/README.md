# 034: all-feature look-ahead teacher + seed replicate (2026-11-06)

**Starting point:** 033's all-feature LSTM LA12 + lean GRU LA50 = **0.66658** (new best, 50m47s); the all-feature
LSTM single 0.66597 (lean 0.66554). Both gains are within seed noise on their own, but they agree.

## Runs (server B; code in `experiments/031_seed_ensemble_distillation/`)

| run | GPU | what | gate / compare with |
|---|---|---|---|
| `R25_FULL_LSTM_LA12_s1` | 0 | `run_student.sh` with ARCH=full_lstm, TV12_LA12 targets, **seed 1**: an exact replicate of 033's LSTM member | its mixpair vs 0.66658: a second sample of the recipe for choosing the final submission |
| `LA_fla12` | 1 | **all-feature look-ahead teacher** (`--arch fla12`, `FullLookaheadModel`: FullFeatureModel + the gated forward convs over the next 12 rows; 373k params). Recipe, split (valid fold 0 held out), seed 70 identical to `LA_la12` | held-out ema_wp vs **0.741101** (LA_la12) |
| `R25_FULL_LSTM_FLA12` | 1 (after the teacher) | its predictions blended 50/50 with TV12 -> `soft/*_TV12_FLA12.npy`; all-feature LSTM student, R23 recipe (seed 0) | single vs 0.66597, mixpair with lean GRU LA50 vs 0.66658 |

`run_fla.sh GPU K` chains teacher -> soft targets -> student. `build_034.sh` (local, unattended) fetches each run with
checksums and packages: `034_mixpair_fullLSTM_LA12s1_leanGRU_LA50`, `034_full_LSTM_FLA12_single`,
`034_mixpair_fullLSTM_FLA12_leanGRU_LA50`. All mixpairs are the 0.66658 graph shape (expected ~50-51 min).

Why the teacher: look-ahead targets were the biggest gain of the last fortnight (TV12 0.6642 -> LA12 0.66554); a
student can only be as good as its targets, and the teacher was still lean (68 features, [x, cos x]).

## Timeline (17-18 min/epoch on server B)

- seed replicate: 24 epochs -> done ~18:00 11-06, mixpair packaged a few minutes later;
- teacher: 14 epochs -> ~15:30; soft targets ~16:30; student -> ~00:30 11-07; packaged ~00:45.

## Results

- Seed replicate `R25_FULL_LSTM_LA12_s1` done 18:10 11-06; `034_mixpair_fullLSTM_LA12s1_leanGRU_LA50` packaged 18:14.
- All-feature teacher `LA_fla12`: led the lean teacher by +0.0035 / +0.002 at epochs 0-1 and +0.0015 at epoch 7, then
  stalled; final best EMA held-out **0.739680 vs 0.741101 (-0.0014)**. More inputs help the causal student but not
  the look-ahead teacher (it already sees the next 12 rows). Student `R25_FULL_LSTM_FLA12` trained anyway (low
  expectation), done ~midnight 11-06.
