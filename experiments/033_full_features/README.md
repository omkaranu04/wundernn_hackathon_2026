# 033: all 112 features with a [x, cos x, sin x] expansion (user request, 2026-11-05)

**Hypothesis (user):** the lean model drops instrument 1's 44 price and volume columns (experiment 027, based on
021/022 when the model was much weaker), and its expansion has only [x, cos x]; we may be losing information.

**Change (one thing vs the best pair):** the best pair (031/032: LSTM on 0.5 TV12 + 0.5 LA12 targets + GRU on
0.5 TV12 + 0.5 LA50 targets; 0.66638 / 0.66641) rebuilt with `FullFeatureModel` (`--arch full_lstm` / `full_gru`):
- inputs: ALL 112 features;
- expansion: [x, cos x, sin x] (sin added: cos alone cannot tell x from -x; together they give the full phase);
- group encoders (Linear + ReLU): i0 price 66->24, i0 volume 66->24, **i1 price 66->24, i1 volume 66->24**, trade (both
  instruments) 48->12; the 8 additional columns expanded (24) pass straight on -> RNN input 132 (lean: 76);
- everything else identical: 2 x 128 core, head 64, aux head, R3 A recipe (floor 0.1, aux 0.2, mass, EMA 0.9998,
  24 epochs, LR halved at 13/18/22, seed 0), train + valid.
Parameters: LSTM member 281,775 (lean ~250k), GRU member 215,215.

## Runs

| run | where | what | compare with |
|---|---|---|---|
| `R23_FULL_LSTM_LA12` | server B GPU 0 | all-feature LSTM on LA12 targets | LA12 lean 0.66554 |
| `R23_FULL_GRU_LA50` | server B GPU 1 (server A attempt OOM: other users' memory) | all-feature GRU on LA50 targets | - |
| `H_FULL_LSTM` | local 1080 Ti | HONEST: train-only, T10 targets, S1_a100 recipe, valid each epoch | S1_a100 0.689331 |

Code lives in `experiments/031_seed_ensemble_distillation/` (shared by the running jobs): `model.py`
`FullFeatureModel` / `expand3`; `train.py` arch choices; `export_slim.py build_multi` now takes all-feature members
(a sin block in the shared expansion, i1 encoders folded into the single encoder MatMul; exact: pair 1.9e-8, single
1.1e-8; lean pair unchanged 7.2e-7); `run_local_honest.sh` (honest test on the local GPU).

## Runtime (measured, same core)

All-feature LSTM + all-feature GRU pair: **1.125x** the current best pair's per-step time -> ~55 min on the platform
(the best pair ran 48m37s): legal but only ~5 min of margin. Hence three packages (`build_033.sh`):
`033_fullpair_LSTM_LA12_GRU_LA50` (the exact request, ~55 min), `033_mixpair_fullLSTM_LA12_leanGRU_LA50` (all-feature
LSTM + the existing lean GRU LA50, ~51 min) and the two singles (`033_full_LSTM_LA12_single`, ~31 min, the cleanest
test; `033_full_GRU_LA50_single`).

## Status

- 2026-11-05 11:10: all three runs training. GRU ~22:00, honest test ~01:00, LSTM ~05:30 on 11-06 (server B GPU 0 is
  shared, ~37 min/epoch); `build_033.sh` packages when both members are done.

## Part 2 (user, 2026-11-05 11:30): the SECOND-best pair (LSTM + fusion GRU, 0.66640, 45m47s) with all features + one more non-linearity

- Member A = the all-feature LSTM `R23_FULL_LSTM_LA12` above (already [x, cos x, sin x] on all 112 inputs, LA12 labels).
- Member B = `R24_LA50_FUSIONSIN` (`--arch fusion_gru_sin`, 130,179 params): the fusion GRU (already all 112 features)
  with a SIXTH channel per feature, sin x: [x, x - x_prev, x - EMA_0.7, x - EMA_0.97, cos x, sin x]; LA50 labels, R3 A
  recipe, local 1080 Ti. Streaming exact (5.6e-8); `build_fusion` folds it into 23 nodes (exact 1.2e-7).
- `export_slim.py --combine` now also takes build_multi graphs (3-D `features`: a Reshape from the shared input).
  Pair measured at 1.097x the second-best pair -> **~50 min** on the platform.
- `build_033b.sh` packages `033_fxpair_fullLSTM_LA12_FUSIONSIN_LA50` and the fusion-sin single.

## Platform results (2026-11-06, all five submitted at once)

| zip | platform | runtime | compare with |
|---|---|---|---|
| `033_mixpair_fullLSTM_LA12_leanGRU_LA50` | **0.66658 (NEW BEST)** | 50m47s | lean pair 0.66638 / E32 pair 0.66641 (+0.00017..0.00020) |
| `033_full_LSTM_LA12_single` | **0.66597 (best single)** | 30m27s | lean LA12 single 0.66554 (+0.00043); W256 LA25 0.66580 |
| `033_full_GRU_LA50_single` | 0.66485 | 33m7s | (lean GRU LA50 never scored alone) |
| `033_fxpair_fullLSTM_LA12_FUSIONSIN_LA50` | 0.66583 | 53m17s | fusion pair 0.66640: the sin fusion member hurts |
| `033_FUSIONSIN_LA50_single` | 0.66339 | 27m1s | weak alone |

Reading: the all-feature LSTM is better on the platform in both the single (+0.0004) and the pair (+0.0002), although
the honest local test said "no gain" (0.689199 vs 0.689331, T10 targets, train only). Both gains are within seed noise
on their own (singles ~±0.0008, pairs ~±0.0006) but point the same way. The all-feature GRU and the sin fusion GRU are
not better partners than the lean GRU LA50. Follow-ups in 034 (seed replicate + all-feature look-ahead teacher).
