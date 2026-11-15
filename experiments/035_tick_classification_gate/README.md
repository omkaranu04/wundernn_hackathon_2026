# 035: zero-inflated tick classification gate for the 0.6569 unit (user proposal, 2026-11-06)

**Proposal (user):** the 0.6569 tick unit holds 47.7% of the metric weight at WP 0.44. Train a look-ahead
CLASSIFICATION teacher on those sequences only (y_bin = |y| > 0.1; focal BCE "FZLPR" + AUCM loss, AdamW), and gate the
students' soft targets there: `SoftTarget_new = P(tick) x mu` with mu = 0.5 TV12 + 0.5 LA_K. Other units unchanged;
the shipped graph is unchanged (same members, same runtime).

## Implementation (code in `experiments/031_seed_ensemble_distillation/`)

- `tick_units.py`: tick unit of every sequence from its targets -> `soft/units_{train,valid}.npy`.
  train: 0.1254 12.3%, 0.2413 11.9%, 0.4685 32.9%, **0.6569 43.0%**; valid: 21.2 / 18.5 / 23.0 / **37.2%**.
- `train_tick_cls.py`: la12 architecture (lean LSTM + gated forward convs over rows t..t+12), 2 logits (t0, t1);
  loss = focal BCE (alpha 0.25 on the tick class, gamma 2) + AUCM (LibAUC's min-max square-margin loss, imratio 0.70,
  a/b minimised, alpha maximised and clamped >= 0); AdamW lr 1e-3 wd 1e-2, 12 epochs, LR halved at 9, EMA 0.9998,
  row weights 0.1 + 0.9 p_scored. Data = 0.6569 sequences of train (4,557) + valid folds 1-4 (560); valid fold 0's 137
  are held out (held-out AUC on scored rows each epoch). Same split and seed as the la12 regression teacher.
- `gate_soft.py write|eval`: gating, and the honest gate test.
- `run_tickcls.sh` (server A GPU 0): classifier -> P(tick) on valid -> gate test -> P(tick) on train -> gated
  `soft/*_TV12_LA12G.npy`, `soft/*_TV12_LA50G.npy` -> student `R26_FULL_LSTM_LA12G` (all-feature LSTM, R23 recipe).
- `run_tickcls_gru.sh` (server B GPU 0, after the 034 seed replicate): `R26_GRU_LA50G` (lean GRU, R16 recipe).
- `build_035.sh` (local): `035_full_LSTM_LA12G_single`, `035_mixpair_fullLSTM_LA12G_leanGRU_LA50` (one change vs the
  0.66658 best), `035_pair_fullLSTM_LA12G_GRU_LA50G` (both members gated). Runtime = the best pair's (~50-51 min).

## Two corrections to the proposal (measured before any training)

1. **Class balance is the other way round.** In the 0.6569 unit, ticks are the MAJORITY: 70.2% of train rows, 65.3%
   of valid scored rows; flat rows are 30-35%. Focal alpha 0.25 on the tick class therefore up-weights the minority
   flat class, which is what focal alpha is for, so the parameters stay as proposed. Also, flat rows have ZERO weight
   in the metric (w = |y|): the gate can only help by re-ranking the magnitudes on rows that DO tick.
2. **Scale across units.** mu is already an expectation (it includes the no-tick case), so P(tick) x mu shrinks the
   0.6569 sequences relative to the other units, and the metric is ONE correlation pooled over all sequences. Test with
   an information-free gate (P = 0.5 everywhere) on valid fold 0: raw gating **-0.0389** global WP; gating with each
   sequence/target rescaled to mu's RMS ("seq" mode) exactly 0. The pipeline evaluates both modes and uses the better.

## Gate test (decides whether students are trained)

`gate_soft.py eval soft/valid_LA12.npy soft/valid_PTICK.npy`: valid fold 0 scored rows (744k; 382k in the 0.6569
unit), mu = the la12 teacher alone (it never saw fold 0; TV12 did, so the TV12 blend is reported separately as
in-sample). Students start only if the better mode does not lower the labels' honest WP.

## Status

- 2026-11-06 11:23: classifier training on server A GPU 0 (shared with another user's job).
- 2026-11-06 14:02: **closed by the gate test.** Classifier held-out AUC 0.832 (t0) / 0.852 (t1). Gating the honest
  la12 teacher's labels on valid fold 0: global WP 0.741101 -> raw 0.714780 (-0.0263), seq-rescaled 0.722274
  (-0.0188); on the 0.6569 unit itself 0.5449 -> 0.4756 / 0.4871. In-sample TV12_LA12: -0.0285 / -0.0217. Multiplying
  by P(tick) destroys magnitude information the regression teachers already carry (mu is already an expectation over
  tick / no tick); the classifier's 0.83 AUC is not enough to sharpen it. No students trained, nothing packaged.
