# 028: Lean model + scored-probability weights, with weight averaging, an auxiliary scored-row head, and a train+valid arm

The user asked (2026-10-10) to try all five follow-up ideas after the 026/027
platform results (026-weighted 0.6559 best so far; 027-lean-control 0.6555;
pure Pearson lost). Three of them are training runs and live here; the other
two are quick checks (weight averaging on existing runs: below; what makes a
row scored: `../025_scored_mask_predictor/README.md`).

## What is combined

Base = experiment 027's lean model (i0 price + i0 volume + all trade +
additional = 68 features, `[x, cos(x)]`, grouped MLPs, 2-layer LSTM/128,
248,510 params, 46 ONNX nodes, ~10% faster than 012) with the STANDARD loss
form (0.5 weighted MSE + 0.5 per-sequence weighted Pearson), which beat pure
Pearson on the platform. One training script, four independent switches:

| switch | idea | cost at inference |
|---|---|---|
| `--mask-probs` | 026's scored-probability row weights (in both loss terms) | none |
| `--ema-decay 0.9998` | exponential moving average of the weights (~2-epoch horizon), scored on valid every epoch next to the raw weights; `best_ema.pt` saved | none (still one model) |
| `--aux-weight 0.2` | auxiliary head `Linear(128,1)` predicting "this row is scored" (target = the scored-probability), added as 0.2 x BCE | none: the export never touches the head (verified: 46 nodes, same outputs) |
| `--extra-train valid.parquet --no-validation` | add the 1,873 validation sequences to training, weighted by their TRUE `is_scored` flag; fixed schedule | none |

Verified before launch: the loss equals 026's verified loss exactly (0.0
difference, weighted and unweighted); the EMA tracks a parameter shift
correctly; the auxiliary head adds 129 params and leaves the exported graph
unchanged; all three configurations smoke-tested end to end.

## Arms (server A, launched 2026-10-10, all on GPU1, `--save-epochs` so any window of epochs can be averaged offline)

| arm | switches | status |
|---|---|---|
| **A** lean + weights + EMA | `--mask-probs --ema-decay 0.9998` | **done**: best raw 0.685089 (epoch 8), **best EMA 0.686746** (epoch 11), stopped epoch 16 |
| **B** A + auxiliary head | A + `--aux-weight 0.2` | **done**: best raw 0.684377 (epoch 12), **best EMA 0.687037** (epoch 10), stopped epoch 17 |
| **C** A trained on train+valid | A + `--extra-train valid --no-validation --epochs 14 --lr-milestones 10` | **done**: all 14 epochs ran; no local score possible (ships `ema.pt` / `last.pt`) |

- **Reference for A/B:** 027 lean control (same model, unweighted): batched
  0.683450, platform 0.6555. And 026: weighted 0.685846 vs control 0.684065
  (012 architecture). Differences under ~0.004 locally are unresolved; a
  platform submission is the real test.
- **Idea 2 (weight averaging) is tested inside every arm:** each epoch logs
  `valid_wp` (raw weights) and `ema_wp` (averaged weights) from the same run.
- **Arm C cannot be evaluated locally** - validation sequences are in its
  training set, so any valid WP would be contaminated. Its schedule is fixed
  from earlier weighted runs (warmup 3 epochs, 1e-3 through epoch 9, 5e-4 for
  epochs 10-13, stop; 026-weighted peaked at epochs 11-13 on that pattern).
  It ships `last.pt` / `ema.pt`; only the platform can score it.
- Arm B's log also reports `aux_auc` / `aux_ap`: how well the main model's
  own state predicts the true scored flag - a causal mask predictor for free.

## Quick check: weight averaging on finished runs (no training)

`avg_ckpt.py` averaged each finished run's `best.pt` and `last.pt` (5 epochs
apart, `last` taken after the LR decays and visibly degraded) and scored the
result (corrected WP, true mask):

| run | best.pt | last.pt | average of the two | avg - best |
|---|---|---|---|---|
| 026 weighted | 0.685846 | 0.679281 | 0.685605 | -0.0002 |
| 026 control | 0.684065 | 0.680854 | 0.684360 | +0.0003 |
| 027 control | 0.683450 | 0.680726 | 0.683624 | +0.0002 |
| 027 pure | 0.680912 | 0.677307 | 0.680888 | -0.0000 |

Reading: averaging the best checkpoint with a clearly worse one gives back
the best checkpoint's score (within 0.0003), every time. So this crude
two-point average is **not a gain over picking the best epoch**, but it gets
there WITHOUT picking - which matters because best-epoch selection on valid is
optimistic and arm C has no validation at all. The proper test (a running
average over neighbouring epochs) is the `ema_wp` column of arms A/B.

## Results (finished 2026-10-11; batched corrected global WP on valid, true mask, single seed)

| run | best raw-weights WP | best EMA-weights WP | EMA - raw |
|---|---|---|---|
| 027 lean control (no weights, no EMA) - reference | 0.683450 | - | - |
| 026 weighted (012 architecture, no EMA) - reference | 0.685846 | - | - |
| **A** lean + weights | 0.685089 (ep 8) | **0.686746** (ep 11) | +0.0017 |
| **B** A + auxiliary head | 0.684377 (ep 12) | **0.687037** (ep 10) | +0.0027 |

EMA WP by epoch, 9-14: A 0.6864 0.6865 0.6867 0.6867 0.6865 0.6861; B 0.6862
0.6870 0.6870 0.6869 0.6868 0.6866. Raw WP over the same epochs swings
between 0.6789 and 0.6851 (A) / 0.6772 and 0.6844 (B).

Readings:
1. **The scored-probability weights help on the lean model too.** Raw A vs the
   lean control: +0.0016, the same size as on the 012 architecture (+0.0018
   local, +0.0015 platform in 026). Three agreeing measurements now.
2. **Weight averaging (EMA) is a clear, cheap win locally.** +0.0017 (A) and
   +0.0027 (B) over the same run's best raw epoch, and - more important - the
   EMA curve is FLAT near its peak (0.6861-0.6870 for six epochs) while the
   raw weights swing by ~0.006. So the EMA number depends very little on
   which epoch is picked; the earlier two-checkpoint check understated this.
   These are the highest local scores so far (previous best 0.685846).
3. **The auxiliary head shows no clear effect on the targets.** B vs A: raw
   -0.0007, EMA +0.0003 - noise. (B's LR also decayed later, at epoch 15 vs
   11.) It does learn the regime: the main model's own state predicts the
   TRUE scored flag at AUC 0.925 / AP 0.63 causally.
4. Stacked: best EMA vs the plain lean control is +0.0033 (A) / +0.0036 (B) -
   still just under the 0.004 bar on one seed, but built from two effects
   that each replicated.
5. **Arm C (train+valid) cannot be scored locally**; only a platform
   submission can judge it. Its EMA weights are the natural candidate.

Not yet exported, real-scored or submitted. Weights (including every epoch,
for offline averaging) are in `results/checkpoints_{A_weighted,B_aux,C_trainvalid}/`,
logs in `logs/`; copied from server A, verified by checksum, deleted there.

## Submission zips (2026-10-11)

Built from the AVERAGED (EMA) weights, exported to `results/checkpoints_*/export_ema/`:

| zip | weights | export vs batched forward | local WP |
|---|---|---|---|
| `028_lean_weighted_A_ema_2026-10-11.zip` | A `best_ema.pt` (epoch 11) | 5.66e-07 | 0.686746 (batched and real scorer) |
| `028_lean_weighted_B_aux_ema_2026-10-11.zip` | B `best_ema.pt` (epoch 10) | 5.07e-07 | 0.687037 (batched and real scorer) |
| `028_lean_weighted_C_trainvalid_ema_2026-10-11.zip` | C `ema.pt` (final, epoch 13) | 4.77e-07 | none possible (valid is in its training set) |

All 1.85 MB, 46 ONNX nodes (B's auxiliary head is not in the graph), outputs
`[prediction, next_h, next_c]`, fresh-extraction checks passed, and the three
archives carry three different weight files. Trained-weights latency,
alternating run vs 012: 105.3% vs 118.2% uncorrected = 0.891x (~11% faster;
expect ~41-43 min on the platform, as 027's lean zips did).

**Real step-by-step scorer (finished 2026-10-11, `logs/real_score_{A_ema,B_aux_ema}.log`),
run on the extracted zips:** A-EMA **0.686746** (t0 0.682016, t1 0.691476, 1h14m);
B-EMA **0.687037** (t0 0.682200, t1 0.691873, 1h17m). Both equal the batched
evaluator to 6 decimals, so the export path and the zips are confirmed.

### Platform results (submitted 2026-10-11) - new best, all three

| zip | platform | runtime | local WP (batched) |
|---|---|---|---|
| A-EMA (lean + weights) | **0.6574** (ID 0H9Y4JPP) | 41m16s | 0.686746 |
| B-EMA (A + auxiliary head) | **0.6583** (ID 0L9LNMMN) - best so far | 40m41s | 0.687037 |
| C-EMA (A on train+valid) | **0.6580** (ID 0VDTAAM7) | 41m1s | not measurable |

Previous best: 026-weighted 0.6559, 012 0.6556, 027 lean control 0.6555.
- **All three land at 0.6574-0.6583, above everything before** (+0.0015 to
  +0.0024 over the previous best; +0.0019 to +0.0028 over the lean control,
  which is the same model without weights or averaging). Three independent
  training runs agreeing is much stronger evidence than any one of them.
- Local ranked B above A (+0.0003) and so did the platform (+0.0009); both
  gaps are inside noise, so the auxiliary head is still "no clear effect",
  but it is certainly not harmful.
- **Train+valid (C) scored +0.0006 over A** - inside noise, but it had no
  local evidence at all and held up, so the fixed schedule and unselected
  averaged weights are safe to use. Adding the validation data is not a
  large effect at this size (+18% sequences).
- Runtime ~41 min (68% of budget): ~5 minutes faster than 012-style models.
- The stack that produced this: lean inputs/`[x, cos(x)]` (027) +
  scored-probability loss weights (026) + EMA of the weights (028).

`analysis/scripts/bench_latency.py` is the alternating latency benchmark used
for these checks (now kept in the repo).

## Offline epoch averaging (2026-10-11, `avg_epochs.py`, no training)

Uniform averages of the saved per-epoch checkpoints, scored on valid:

| window (epochs) | A | B |
|---|---|---|
| best single epoch | 0.685089 | 0.684377 |
| 4-8 | 0.684407 | 0.684470 |
| 6-10 | 0.685262 | 0.685842 |
| 6-12 | 0.685768 | 0.686501 |
| 8-12 | 0.686236 | 0.686626 |
| 7-13 | 0.686405 | 0.686847 |
| 8-14 | 0.686559 | 0.686929 |
| 9-13 | 0.686177 | 0.686597 |
| 10-12 | 0.686423 | 0.686162 |
| 5-15 | 0.686203 | 0.686662 |
| 3-16 | 0.685996 | 0.686036 |
| **EMA (in the submitted zips)** | **0.686746** | **0.687037** |

Every window around the peak lands within ~0.001 of the EMA and none beats
it. Uniform averaging = EMA here; nothing more to gain from the choice of
averaging scheme. Both are ~+0.002 above the best single epoch.
