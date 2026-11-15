# 020: Recurrent-state drift / periodic-reset diagnostic (no training)

Idea tested (from another competition's write-up): zeroing the LSTM state
every N steps helped there once the metric became globally pooled, because
a stale carried state drifted. Our LSTMs carry state across all 20,000
steps. Two checks on ALREADY-TRAINED models (004 30-epoch, 012), using the
validated corrected-global-WP accumulator (019's `eval_wp.py`; full-state
baseline reproduced the known scorer values 0.682425 for 004 and 0.682697
for 012). Run 2026-10-08 on server A. `diag.py`, `results_*.json`, `diag_*.log`.

## Results (full valid.parquet, 1873 sequences)

Inference-time reset (state zeroed every N steps; models NOT trained for it):

| model | full state | reset 10000 | reset 5000 | reset 2000 |
|---|---|---|---|---|
| 004 | 0.682425 | 0.682093 (-0.0003) | 0.681682 (-0.0007) | 0.680125 (-0.0023) |
| 012 | 0.682697 | 0.682431 (-0.0003) | 0.682310 (-0.0004) | 0.681302 (-0.0014) |

WP by 2,000-step position bin, full state (004): 0.670 0.694 0.674 0.671
0.670 0.689 0.701 0.679 0.689 0.689 - **flat, no downward trend** (012 the
same). Later bins are not worse than early ones.

Cold-start cost, reset every 2000, WP by position within chunk (004):
[0,100) 0.649 | [100,300) 0.682 | [300,1000) 0.684 | [1000,2000) 0.680.
(012: 0.668 | 0.681 | 0.684 | 0.681.)

## Reading

- **No drift signature.** WP does not decay over the sequence, and every
  reset made WP monotonically worse the more often it was applied. No N
  matched the full-state baseline, let alone beat it.
- The reset models recover to full-state quality within ~100-300 steps
  (after the 10,000-step reset, bins 5-9 are within ~0.001-0.003 of full
  state, mostly 0.0001-0.0005). So the model's *useful* memory is short:
  state built over 10,000 steps predicts no better than state rebuilt in a
  few hundred. This is the opposite of "stale state hurts": the state is
  simply not carrying much long-range information.
- Caveat: the models were not trained with resets, so the cold-start cost
  (~0.03 WP in the first 100 steps of each chunk) is partly avoidable by
  retraining. But a retrain can only help if stale state was hurting, and
  these results show no sign it is. Idea judged not worth a training run.
- Side observation, UNTESTED: since long carried state seems to add little,
  training on short chunks (e.g. 1,000 steps, fresh state) might lose
  nothing and could train faster (shorter sequences, more parallelism).
  Only worth testing as a training-speed optimization, not an accuracy one.
