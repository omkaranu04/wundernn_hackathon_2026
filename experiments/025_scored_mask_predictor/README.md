# 025: Scored-row mask predictor (labels for train.parquet)

Goal: the metric scores only a hidden ~10% subset of rows (`is_scored`), and
that subset is a much more predictable regime (012: WP 0.670 on scored vs
0.468 on unscored rows; see EXPERIMENTS.md "Scored-row structure"). Only
valid.parquet carries `is_scored` (also `valid_mask.parquet`); train.parquet
has none. This experiment learns the mask from valid and uses it to label
every row of train.parquet with a soft scored-probability, so a later
training run can weight its loss toward likely-scored rows. The predictor is
used OFFLINE ONLY and never ships in a submission, so it is non-causal.

## Model (`mask_model.py`, `train_mask.py`)

Bidirectional 2-layer LSTM (hidden 96) over 2,000-step crops. Inputs: the 112
raw features, their step-to-step differences, and the normalized position in
the sequence. **Targets t0/t1 are never inputs.** Loss: BCE on
`need_prediction` rows only (the first 99 steps are never scored). Split by
sequence: **held-out = every 4th valid sequence (469 seqs, 9.33M rows)**,
the other 1,404 train it. Inference uses overlapping windows (stride 1000),
keeping each window's central part, so every row has >=500 steps of context
on both sides except at the sequence ends. 25 epochs, ~17 s each on an L40S.

## Held-out results (469 unseen valid sequences)

| | AUC | average precision | base rate |
|---|---|---|---|
| gradient-boosting probe (112 features + diffs, no sequence context, 50 seqs) | 0.859 | 0.42 | 0.085 |
| **bidirectional LSTM (this model)** | **0.952** | **0.728** | 0.092 |

Still improving slowly at epoch 25 (AP 0.51 at epoch 2 -> 0.73 at 25), so
more epochs or width would likely add a little; not pursued.

## Calibration

Raw probabilities were slightly over-confident (mean 0.103 vs. true 0.092;
the 0.3-0.7 bins ~5-6 points high). `calibrate.py` fits a 2-parameter Platt
map on the held-out sequences (a=0.999, b=-0.254, i.e. essentially a logit
shift); afterwards the reliability table matches to three decimals in every
bin (e.g. predicted 0.3913 vs actual 0.3917 in [0.30,0.50)). AUC/AP are
unchanged by a monotone map.

## Labels for train.parquet (`label_train.py`)

All 10,607 training sequences labelled in 3.5 min:
`mask_out/mask_probs_train.npy`, float16, shape (10607, 20000), row-group
order (= sequence order in train.parquet), calibrated probabilities.
**Mean probability on train `need_prediction` rows = 0.106** (valid's true
rate is 0.092): both ~10%, matching the "~10% scored" description; the gap
is either a genuine train/valid difference or mild predictor shift - flagged,
not resolved. Rows with `need_prediction == False` are stored but ignored.

## Files

`mask_out/` (weights, `calibration.json`, `mask_train_results.json`,
`mask_probs_train.npy` ~424 MB, gitignored), `train_mask.log`,
`label_train.log`. A copy of `mask_probs_train.npy` is kept on server A for
the planned weighted-loss training run (012 architecture, loss weights
p x |clipped target| in both the MSE and Pearson terms).

## Follow-up (2026-10-10): what makes a row "scored"?

**Causal mask predictor** (`train_mask.py --causal`: same model but a
unidirectional LSTM, so it sees only past and current rows - what a shipped
model could know). Same split, 25 epochs:

| mask predictor | held-out AUC | average precision |
|---|---|---|
| bidirectional (sees the future) | 0.952 | 0.728 |
| **causal (past + current only)** | **0.938** | **0.673** |

So the scored regime is almost entirely knowable at prediction time; the
future adds little (0.014 AUC). Outputs in `mask_out_causal/` (not used to
label train - the bidirectional, calibrated one in `mask_out/` is better).

**Rule probe** (`rule_probe.py`, 50 valid sequences to fit, 19 held out,
`rule_probe.log`): there is NO simple rule.
- Best single per-row statistic: the value of `a3`, AUC 0.655; then `a4`
  0.617, `i0_dp0` 0.613, `a2` 0.613, `a5` 0.593. By group, `additional` is the
  most informative (max 0.655), then i0 trade (0.613 / 0.565); price and volume
  levels are near chance (<= 0.558).
- "Any trade feature changed from the previous row" barely separates
  (P(scored) 0.088 vs 0.074); no on/off event flag explains scoring.
- A depth-4 decision tree reaches held-out AUC 0.744 on causal inputs and
  0.747 when also given the next row - no gain from the future at this depth.
  Top splits: `a3`, `i0_dp0`, `a5`.
- The gap from 0.74 (shallow tree) / 0.86 (boosted trees on one row) to 0.94
  (causal LSTM) says the regime is a diffuse function of recent history, not
  a threshold on one feature.
- Note `a3`: experiment 021 found it slightly HARMFUL for predicting the
  targets, yet it is the single best indicator of whether a row is scored.

## Follow-up (2026-10-11): can the TARGETS reveal the mask? No.

`target_mask_probe.py` (valid; targets used deliberately - this would only
ever produce LOSS WEIGHTS for training rows, never model inputs):
- The targets are piecewise constant: they differ from the previous row on
  only 18.1% of rows. Scoring is not tied to those changes
  (P(scored | changed) 0.059 vs 0.080 unchanged).
- Best single target statistic: "steps until the next change", AUC 0.574;
  |t|, |change|, zero-ness are all <= 0.522.
- Boosted trees on targets alone (statistics + raw window t-3..t+3): held-out
  AUC 0.743, AP 0.18 - far below the feature-based predictor (0.952 / 0.728).
- Added to the features in a one-row model (`target_plus_feature_probe.py`),
  target statistics lift AUC only from 0.877 to 0.886 (AP 0.431 -> 0.457).
So train cannot be labelled exactly from its targets; the feature-based
predictor stays the tool.

## v2 mask predictor (2026-10-11)

Same model as v1 (hidden 96, 2 layers, bidirectional), trained 60 epochs with
a cosine LR decay instead of 25 at a constant LR (`train_mask.py --epochs 60
--cosine`, output `mask_out_v2_h96/`). Full held-out set (469 sequences):

| | AUC | average precision |
|---|---|---|
| v1 (25 epochs) | 0.9521 | 0.7278 |
| **v2 (60 epochs, cosine)** | **0.9586** | **0.7555** |

Raw v2 probabilities are over-confident at the top (the [0.70,1.00) bin
predicts 0.882 vs 0.830 actual), so it is Platt-recalibrated before use, as
v1 was. A chain on server A (`relabel_v2_then_queue.sh`) calibrates it, labels
train.parquet (`mask_out_v2_h96/mask_probs_train.npy`) and queues a 029 run
(`A_v2labels`) that differs from 028-A only in the labels. A wider variant
(hidden 192, 3 layers) was still training (AUC 0.9617 / AP 0.7656 on the
60-sequence check at epoch 35 of 60, ~2.7 min per epoch on the shared GPU;
on that same check v2-h96 had 0.9597 / 0.7564 at epoch 35 and finished at
0.9611 / 0.7645, so the wider model is ahead by only ~0.002 AUC at equal
epochs while its train loss is much lower, 0.118 vs 0.136 - at best a small
further gain).

**v2 labels done (2026-10-11 14:00):** Platt a=0.884, b=-0.259 (v1: 0.999,
-0.254); mean calibrated probability on train's need-prediction rows 0.1105
(v1: 0.106; valid's true rate 0.102). File `mask_out_v2_h96/mask_probs_train.npy`
(424 MB) is on the shared remote home only so far; `A_v2labels` is queued on it.

**Wider v2 variant finished (2026-10-11 evening): no gain.** Hidden 192, 3
layers, 60 epochs cosine (`mask_out_v2_h192/`, on the shared remote home):
full held-out AUC 0.9579 / AP 0.7517, slightly BELOW v2-h96 (0.9586 / 0.7555)
despite a much lower train loss (0.097 vs 0.125) - it overfits the 1,404
training sequences. v2-h96 stays the best mask predictor; capacity is not the
limit, so further gains would need more labelled sequences or new inputs.
