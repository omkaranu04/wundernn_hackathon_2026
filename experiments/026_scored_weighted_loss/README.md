# 026: 012 trained with a scored-probability-weighted loss

Hypothesis (from a competitor-analysis idea, checked on our data in
EXPERIMENTS.md "Scored-row structure" and experiment 025): the metric pools
only a hidden ~10% of rows (`is_scored`), a regime where 012 correlates far
better (WP 0.670 scored vs 0.468 unscored). The existing loss weights every
row equally, so ~90% of its weight goes to rows the metric ignores.

## What changes (and what does not)

**Unchanged:** architecture (012's `NonlinearGroupedEncoderLSTMModel`,
269,822 params; `model.py`, `export_onnx.py`, `solution.py` are byte-copies of
012's), loss form (0.5 weighted MSE + 0.5 per-sequence weighted Pearson),
Adam 1e-3, batch 4, 3-epoch warmup, ReduceLROnPlateau, early stopping
(patience 5), 30 epochs max, seed 0.

**Changed (the only thing):** each training row's contribution is multiplied
by its scored-probability p (from experiment 025's calibrated mask predictor,
`../025_scored_mask_predictor/mask_out/mask_probs_train.npy`), on top of the
existing |clipped target| weights, **in both terms**: the MSE sums AND the
weighted-Pearson means/variances/covariance. `weight-floor f` (default 0)
makes it `f + (1-f)*p`. Soft weights, no thresholding.

Verified numerically (`train.py::compute_loss`): with no weights it equals the
old loss to 1e-7; with 0/1 weights it equals the old loss restricted to the
selected rows exactly; gradients are finite for soft weights.

## Evaluation

Checkpoint selection, LR decay and early stopping use the corrected global WP
on valid.parquet with the TRUE scored mask (batched `eval_wp.py`, validated
against the real scorer). `valid_loss` is logged as the old unweighted loss,
comparable to earlier experiments. Compare to the SAME-CODE unweighted
control run alongside (identical selection rule), and to 012's published
numbers (local WP 0.682697, valid_loss 0.3438, platform 0.6556). Per the
brief, **differences smaller than ~0.004 are unresolved, not wins**.

## Runs (server A, 2026-10-09)

| arm | weights | status |
|---|---|---|
| weighted | p (floor 0) | **done**: best valid WP 0.685846 (epoch 11), early-stopped epoch 16 |
| control | none (all rows equal) | **done**: best valid WP 0.684065 (epoch 12), early-stopped epoch 17 |
| blend (planned) | 0.2 + 0.8p | not yet run; weighted did not underperform, so this is optional - see Results |

## Risks being watched

- Concentrating the loss shrinks the effective sample (~10-40% depending on
  the weight spread); watch train vs. valid curves for earlier overfitting.
- A weak mask predictor would add noise; mitigated by its strength (held-out
  AUC 0.952 / AP 0.728, calibrated) - mean p on train is 0.106 vs valid 0.092,
  an unresolved small shift.
- If the weighted run underperforms, rerun with the floor blend before
  concluding the idea fails.

## Results

Batched corrected global WP on valid (true scored mask), single seed each,
both selected on WP:

| | best valid WP | epoch | WP at epochs 11 / 12 / 13 | unweighted valid_loss at best | early-stopped |
|---|---|---|---|---|---|
| **weighted** (p) | **0.685846** | 11 | 0.6858 / 0.6853 / 0.6854 | 0.3634 | 16 |
| control (no weights) | 0.684065 | 12 | 0.6832 / 0.6841 / 0.6833 | 0.3439 | 17 |

- **Difference at best epochs: +0.0018** (weighted ahead). By the brief's
  bar (~0.004) this is **unresolved, not a win**. Averaged over the three
  best-plateau epochs the gap is +0.0020, so it is not a single-epoch
  spike, but it is the size of our known run-to-run noise (~0.002).
- Versus 012's published numbers (local WP 0.682697 selected on valid_loss):
  weighted +0.0032, control +0.0014 - the control also beats 012, showing how
  much WP-based selection plus seed noise alone can add.
- As expected the weighted model has a WORSE unweighted valid_loss (0.363 vs
  0.344): it is trained to ignore unscored rows, which is exactly what the
  metric does.
- Overfitting: both arms peak around epochs 11-13 and degrade after the LR
  decays at 14-15 (weighted 0.6854 -> 0.6793 by epoch 16; control 0.6841 ->
  0.6805), so the weighted arm does not overfit noticeably earlier than the
  control. (I had suspected otherwise mid-run; the control degrades too.)
- **Real step-by-step scorer** (`solution.py --validation`, exported ONNX, max
  diff vs batched forward 9.5e-07 / 7.2e-07): **weighted 0.685846** (t0 0.681520,
  t1 0.690171) vs **control 0.684065** (t0 0.680435, t1 0.687694) - identical to
  the batched evaluator to six decimals, so `eval_wp.py` is validated on
  012-architecture models too. Gap +0.001781. Logs:
  `logs/real_score_{weighted,control}.log`.

## Reading / what next

A real but small and unresolved gain. To separate it from noise: a second
seed of both arms (~2 runs, ~5-6 h each in parallel), and/or the planned
floor blend (0.2 + 0.8p). A platform submission of the weighted model would be
the decisive (if noisy, +-0.002) test, since local WP here is selected on the
same valid set it is scored on.


## Submission zips (2026-10-10)

`submissions/026_scored_weighted_loss_weighted_2026-10-10.zip` and
`submissions/026_scored_weighted_loss_control_2026-10-10.zip` (2.0 MB each;
012-architecture ONNX from `results/checkpoints_*`, fresh-extraction checks
passed). Trained-weights latency, alternating run vs 012: weighted 117.3% vs
012 116.2% (same graph, ~equal). Real-scorer WP 0.685846 / 0.684065. Platform
results (submitted 2026-10-10): **weighted 0.6559** (ID 22HKZ5TQ, 47m46s = 79.6% of budget) and **control 0.6544** (ID 2EV24GDL, 45m37s). The weighted-over-control gap is +0.0015 on the platform vs +0.0018 on the real scorer locally: the same direction and size, each below the 0.004 bar but mutually consistent. 0.6559 is the best platform score so far (012: 0.6556; a tie within noise). Same graph, different runtimes (47m46s vs 45m37s): platform load variance ~2 minutes.
