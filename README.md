# Alfa Connectome ML Competition (Wunder Fund): streaming RNN ensemble on one CPU core

Our solution and full research log for Wunder Fund's **Alfa Connectome** competition (2026).

The task was to predict two undisclosed indicators (`t0`, `t1`) of an instrument's future price movement from a
stream of 112 anonymised order-book, trade and auxiliary features. The model reads one row at a time, in order, and
carries state within each 20,000-row sequence. Constraints:

- **1 vCPU** and **60 minutes** for the whole test set (~39.4M rows, so about 90 µs per row);
- a 20 MB submission;
- `numpy`, `onnxruntime` and `pyarrow` only at inference.

The metric is a |y|-weighted Pearson correlation, with predictions clipped to ±2. A full restatement of the problem
is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

**Best public leaderboard score: 0.66754**, from a five-model LSTM/GRU ensemble that ran in 47 minutes. The same
ensemble on our int8 engine scored 0.66753 in 38½ minutes.

## Final approach

1. **Causal recurrent members.** Every member is an LSTM or GRU that runs one step per row.
   - Input features: `[x, cos x, sin x]` encodings, built from all 112 features ("full") or from the 68 informative
     ones ("lean").
   - Head: a small MLP.
   - Trained with a weighted-Pearson-aware loss, row weights based on the probability that a row is scored, an
     auxiliary head and EMA weights.
2. **Distilled, smoothed targets.** Members are not trained on raw labels. They are trained on a 50/50 blend of two
   soft-target sets (experiments 029-034):
   - the average prediction of 12 causal teachers trained on train+valid ("TV12");
   - the predictions of a non-causal **look-ahead teacher** that also sees the next K rows (K = 12, 25 or 50; the
     "LA12", "LA25" and "LA50" members).

   The look-ahead teacher is only used to make training targets. It is never shipped.
3. **Diverse ensemble.** The final ensembles mix an LSTM fine-tuned on LA12 targets, a lean GRU (LA50), a full LSTM
   (LA50) and one to three seeds of a full GRU (LA25), with a fixed weight per member.
   - Diversity of architecture, look-ahead window and seed added about +0.001 over the best single model.
   - Every member was trained on validation data too, so members were chosen by priors and by public-leaderboard
     paired comparisons, not by in-sample validation scores.
4. **A custom inference engine.** Fitting 4-6 recurrent members into 90 µs per row on one core needed a
   hand-written [numba](https://numba.pydata.org/) engine in place of ONNX Runtime
   ([experiments/039](experiments/039_engine_events_finetune) and [experiments/040](experiments/040_numba_quads)):
   - the whole ensemble runs in a single compiled call per row;
   - the matrix-vector products use int16 integer arithmetic (v5), with the `vpmaddwd` multiply-add instruction
     reached through a numba `@intrinsic`;
   - v7 uses int8 weights, with a per-row quantisation scale for activations;
   - each engine checks itself against a reference at start-up and falls back (v7 → v5 → v4 → ONNX) if the check
     fails.

   The engine work took the per-row cost from about 130 µs (ONNX) to under 60 µs on the platform. That is what made
   five- and six-member ensembles fit the time limit.

## Score progression (public leaderboard)

| stage | score |
|---|---|
| Organizer GRU baseline / first LSTM experiments | ~0.62-0.65 |
| Lean inputs + scored-row weights + EMA + auxiliary head (027-030) | 0.6605 |
| Seed-ensemble distillation into one student (031) | 0.6642 |
| Look-ahead-teacher targets, all-feature members, pairs on ONNX (032-034) | 0.66658 |
| numba engine → three and four members (039-040) | 0.66717 |
| integer engine v5 → full-GRU LA25 member, four members | 0.66735 |
| five members with two GRU seeds (v5) | **0.66754** |

All 136 scored submissions, with their runtimes, are listed in
[analysis/scripts/compute_timing_correlations.py](analysis/scripts/compute_timing_correlations.py).

## Repository layout

```
├── PROBLEM_STATEMENT.md     the competition: data, metric, constraints, rules
├── EXPERIMENTS.md           chronological lab notebook: every experiment, hypothesis and result
├── shared/                  code shared by all experiments (Parquet sequence reader, losses, plotting)
├── analysis/                exploratory analysis of the 112 features and targets; timing/score analysis
└── experiments/
    └── NNN_short_name/      one directory per idea; each README states the hypothesis, setup and outcome
```

Where to start:

- **Final pipeline:** [experiments/031_seed_ensemble_distillation](experiments/031_seed_ensemble_distillation).
  Despite its name, this directory holds the code for all later work:
  - training: `model.py`, `train.py`, `blend_soft.py`, `teacher_preds.py`;
  - export: `export_onnx.py`;
  - numba engines: `numba_engine*.py`;
  - submission entry points: `solution_numba*.py`;
  - packaging: `make_zip_numba.py`.
- **Engine development and the final ensembles:** [experiments/040_numba_quads/README.md](experiments/040_numba_quads/README.md).
- **Earlier ideas, most of which did not work:** experiments 001-030. They include attention, XGBoost heads,
  rolling normalisation, nonlinear encoders, sparse attention, a look-ahead classification gate and multi-head
  members, each with its result.

## Reproducing

The competition data and the organizer's starter pack are **not** included. Get them from the competition page and
place them at `wnn_connectome_starterpack/`, with `datasets/train.parquet` and `datasets/valid.parquet` inside, plus
the organizer's `utils.py`. `shared/data.py` imports `DataPoint` and the column constants from that `utils.py`.

The environment:
- for training: Python 3.12, `torch`, `numpy`, `pyarrow`, `onnx`, `onnxruntime`, `numba`, `tqdm`, `matplotlib`;
- at inference: `numpy`, `onnxruntime`, `numba`.

A typical cycle for one member:

```bash
cd experiments/031_seed_ensemble_distillation
python teacher_preds.py ...     # soft targets from trained teachers
python blend_soft.py ...        # 0.5 x TV12 + 0.5 x look-ahead blend
python train.py ...             # train a causal member on the blended targets
python make_zip_numba.py --engine 5 ...   # pack members into a numba submission zip (with ONNX fallback)
```

Arguments vary by experiment. Each experiment README records the exact settings used.

Not in this repository:
- model weights and checkpoints;
- packed `.npz` engine weights;
- prediction dumps;
- launcher shell scripts and run logs.

These are regenerable from the code, or specific to our machines.

## Lessons that may transfer

- **On a CPU-bound streaming task, engineering is modelling.** Every ensemble gain after mid-competition came from
  making members cheaper, so that more of them fit in the time budget.
- **Measure on the target platform.** Local speed-ups did not always transfer. IOBinding was faster locally and
  slower on the platform; the numba engines ran at about 0.6× of local time. Runtime for identical code drifted by
  about 10% from day to day.
- **Distilled targets beat raw labels.** Averaged causal teachers, and especially look-ahead teachers, gave
  smoother targets that students learned better from.
- **Seed noise is large.** Single-seed differences on the public leaderboard (~0.0002-0.0013) were often as large
  as the change being tested. Use paired comparisons, seed replicates and seed averaging.
- **In-sample selection misleads.** Once members are trained on validation data, validation scores reward
  memorisation. We chose members by priors and confirmed them on the public leaderboard.
- **Most training-recipe changes did not move the platform score** (loss variants, optimizers, dropping dead
  features). Target construction and ensemble diversity did.

## License and data

This repository contains our code and notes only. Competition data, organizer documents and the starter pack remain
the property of the organizers and are not redistributed here.
