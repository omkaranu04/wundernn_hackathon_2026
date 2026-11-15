# 023: Ensemble ceiling test (teacher headroom for distillation)

Question: if we distill an ensemble of our trained models into one
budget-sized student, how much accuracy is on the table? The ensemble is
the teacher, so its WP is the ceiling for what distillation could transfer.
No training: `extract_preds.py` runs 20 trained models over valid.parquet
once (server A) and stores each model's raw predictions on the 3,528,273 scored
rows (`preds/`, ~600 MB, gitignored, regenerable); `ensemble_eval.py`
evaluates any combination offline with the corrected global WP. Sanity: each
single model's WP reproduces its real-scorer number (017 0.68293, 016
0.68275, 012 0.68270, 004 0.68243, 018 0.68189, 013 0.67877).
Run 2026-10-08. Output: `ensemble_eval.log`, `ensemble_results.json`.

## Results (corrected global WP, valid)

Best single model: **017 = 0.68293** (012 0.68270, 016 0.68275, 004 0.68243).

| ensemble | WP | vs best single |
|---|---|---|
| 012 + 017 (equal avg) | 0.68534 | +0.0024 |
| 012 + 004 | 0.68549 | +0.0026 |
| 012 + 016 | 0.68580 | +0.0029 |
| 004 + 016 (best pair, picked on valid) | 0.68609 | +0.0032 |
| 012 + 017 + 018 | 0.68656 | +0.0036 |
| nonlinear family (012, 017, 018, 015, 016) | 0.68705 | +0.0041 |
| grouped family (004, 010, 019 x3, 022 arm2) | 0.68549 | +0.0026 |
| raw-LSTM family (002, 003, 005, 006, 008, 009) | 0.68324 | +0.0003 |
| all 20 models, equal avg | 0.68670 | +0.0038 |
| **cross-fitted greedy selection** (pick on half the sequences, score the other half) | **0.68730** | **+0.0044** |
| cross-fitted ridge stacking over all 20 | 0.68709 | +0.0042 |

Cross-fitting makes the last two honest: members/weights are chosen on even
sequences and scored on odd ones (and vice versa), so they are not tuned on
the data they are scored on. They are the best estimate of the real ceiling:
**~0.687, about +0.004 over the best single model.**

## Reading

- **There is real headroom, but it is modest**: +0.003-0.004 WP. That clears
  the +0.003 bar set beforehand, narrowly.
- **Members agree very strongly** (t0 prediction correlation 0.965-0.979
  between any two key models), which is why the gain is small even though
  the models are architecturally different. Different families do add
  something: pairs mixing a raw-grouped model with a nonlinear one
  (004+016, 004+017) beat pairs within the nonlinear family (012+017).
- **The strongest teacher members** (chosen by both cross-fit halves
  repeatedly): 017, 016, 004, 018, plus 012. Weak models (008, 009, 003,
  005) add nothing and the raw-LSTM family on its own barely helps.
- **A distilled student will keep only part of this.** A single
  budget-sized student usually recovers a fraction of an ensemble's gain,
  so the realistic target is ~+0.001-0.003 local WP. Our local-to-platform
  noise is ~0.002 (018, pooled60), so even a successful distillation may
  not show up reliably on the platform.
