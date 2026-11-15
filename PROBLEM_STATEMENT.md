# Alpha Connectome ML Competition — Problem Statement

Consolidated reference. Source: competition docs (`wnn_connectome_starterpack/docs/`, `README.md`, `METRIC.md`) and starter pack code (`utils.py`, `baseline/`).

---

## 1. The task

Predict `t0` and `t1` — two **undisclosed indicators of the future price movement** of a trading instrument `i0` — from a stream of orderbook and trade features for `i0` and a second instrument `i1` (context only, no targets of its own).

This is framed as a **streaming / online inference problem**, not a batch one:

- Data arrives as **sequences** of 20,000 consecutive rows, one row per timestep.
- Your model must process rows **one at a time, in order**, maintaining internal state (e.g. RNN hidden state) that is updated on **every row**, including rows where no prediction is required.
- State must be **reset whenever `seq_ix` changes** — sequences are independent, no information may cross between them.
- A prediction at step `n` may use **only** the current row and earlier rows of the *same* sequence. No look-ahead, no using future rows.
- `seq_ix` values are shuffled (not time-ordered) and must **never** be used as a feature or for ordering.

Sequence structure:

| Steps | `need_prediction` | Notes |
|---|---|---|
| 0 – 98 (99 rows) | `False` | Warm-up: gives the model context before its first prediction. State must still update. |
| 99 – 19,999 | `True` | Must return a finite `[t0, t1]` prediction on **every** one of these rows. |

Only a *subset* of the required rows are actually used for scoring (`is_scored`), but this mask is **hidden at test time** — your model must return a valid prediction on every required row regardless, "as if it were scored."

---

## 2. Dataset

Provided in `datasets/` as Apache Parquet files. **Each Parquet row group = exactly one sequence** (20,000 rows, single `seq_ix`) — use `ParquetFile.read_row_group(i)` to iterate without loading a whole file into memory.

| File | Sequences | Rows | Columns |
|---|---:|---:|---|
| `train.parquet` | 10,607 | 212,140,000 | 117 (see below) |
| `valid.parquet` | 1,873 | 37,460,000 | 118 (adds `is_scored`) |
| `valid_mask.parquet` | 1,873 | 37,460,000 | 3 (`seq_ix`, `step_in_seq`, `is_scored`) — same mask as `valid.parquet.is_scored`, as a standalone table for convenience |

**Test data** (never distributed) has the same shape as `valid.parquet`, but is fed to your `predict()` callback row-by-row by the platform; you never see `is_scored` or the targets.

### Column layout

```
train.parquet       seq_ix, step_in_seq, need_prediction,            <112 features>, t0, t1   (117 cols)
valid.parquet       seq_ix, step_in_seq, need_prediction, is_scored, <112 features>, t0, t1   (118 cols)
valid_mask.parquet  seq_ix, step_in_seq, is_scored                                             (3 cols)
```

| Column | Type | Meaning |
|---|---|---|
| `seq_ix` | int | Sequence id. **Not a feature.** Shuffled — a larger id does not mean "later." |
| `step_in_seq` | int | Row position within the sequence, 0…19,999 |
| `need_prediction` | bool | `False` on steps 0–98 (warm-up), `True` on 99–19,999 |
| `is_scored` | bool (valid only) | `True` on rows actually used for evaluation. Always implies `need_prediction` (it's a safeguard AND, never selects warm-up). Hidden at test time. |
| 112 feature columns | float32 | See below |
| `t0`, `t1` | float32 | Targets — only in `train`/`valid`, never given at test time |

### Features (112 total, float32)

Two instruments, `i0` (the one you predict) and `i1` (context only). Naming: `i<instrument>_<group><index>`, groups are `p` (price-like), `v` (volume-like), `dp` (trade price), `dv` (trade volume). This is **L11 order book data**: top-11 levels on each side (bid/ask) for price and volume, plus recent trade history, plus 8 extra columns.

**Column order is fixed and must not be re-sorted alphabetically** (`i0_p10` sorts before `i0_p2` alphabetically but is not adjacent to it in the real feature order):

| Group | Columns | Count | Position in `state` |
|---|---|---:|---|
| i0 bid price-like | `i0_p0` … `i0_p10` | 11 | 0–10 |
| i0 ask price-like | `i0_p11` … `i0_p21` | 11 | 11–21 |
| i0 bid volume-like | `i0_v0` … `i0_v10` | 11 | 22–32 |
| i0 ask volume-like | `i0_v11` … `i0_v21` | 11 | 33–43 |
| i0 trade price | `i0_dp0` … `i0_dp3` | 4 | 44–47 |
| i0 trade volume | `i0_dv0` … `i0_dv3` | 4 | 48–51 |
| i1 bid price-like | `i1_p0` … `i1_p10` | 11 | 52–62 |
| i1 ask price-like | `i1_p11` … `i1_p21` | 11 | 63–73 |
| i1 bid volume-like | `i1_v0` … `i1_v10` | 11 | 74–84 |
| i1 ask volume-like | `i1_v11` … `i1_v21` | 11 | 85–95 |
| i1 trade price | `i1_dp0` … `i1_dp3` | 4 | 96–99 |
| i1 trade volume | `i1_dv0` … `i1_dv3` | 4 | 100–103 |
| Additional | `a0` … `a7` | 8 | 104–111 |

**Important caveat:** the index within a group is just an *identifier*, not a depth level or price rank. `p0` is not necessarily the best bid, `p10` is not necessarily the deepest level — the guarantee is only that the same underlying component appears in the same column on every row. Don't assume ordering by price or distance-from-best-quote. Features and target semantics are otherwise **anonymized/undisclosed** by design, to keep the competition about modeling rather than domain-specific feature engineering.

### Targets

- `t0`, `t1`: float32, undisclosed nature, both indicators of `i0`'s *future* price movement. No targets exist for `i1`.
- Present only in `train.parquet` / `valid.parquet`.
- Your model returns them as a length-2 array, `[t0, t1]` order.

---

## 3. Evaluation metric — Global Weighted Pearson correlation (WP)

**⚠️ CORRECTED 2026-09-15/16.** The organizers announced a mistake in their
original WP calculation and re-evaluated every submission under a
corrected metric. The spec below is the corrected version; see
`EXPERIMENTS.md`'s "Metric correction" section for what changed, why it
matters, and the re-evaluated scores for all our submissions. The old
spec ID was `v2_wp_block_clip2_v1` (per-sequence-block averaged); the
corrected metric is described by the organizers simply as **"Global
Weighted Pearson correlation."** Higher is better, range `[-1, 1]`.

Calculate WP separately for `t0` and `t1` **over the entire evaluated
dataset** (not per-sequence), using all rows where `need_prediction AND
is_scored` is true. On those rows, both target and prediction values are
clipped to `[-2.0, 2.0]`:

```
y_i = clip(t_i, -2, 2),   ŷ_i = clip(p_i, -2, 2),   w_i = |y_i|
```

For each target, the weighted correlation is:

```
ρ_w = cov_w(y, ŷ) / sqrt(var_w(y) · var_w(ŷ))
```

**All selected moments (weights, means, covariance, variances) are pooled
across every sequence in the dataset into one single set of sums — scores
are NOT averaged per sequence/block anymore.** This is the core change
from the old metric.

The final score is the mean of the two (now-global) target correlations:

```
S = (ρ_t0,w + ρ_t1,w) / 2
```

**What this changes vs. the old (pre-correction) metric:**
- Old: compute one weighted-Pearson value *per 20,000-row sequence*, then
  average those per-sequence values equally across all eligible sequences
  (so a sequence with 100 scored rows counted exactly as much as one with
  10,000).
- New: pool every selected row from every sequence into one global
  calculation per target — a sequence's influence on the score is now
  proportional to how many (and how heavily-weighted) rows it contributes,
  not counted as "one vote" per sequence.
- The old metric's per-block **eligibility rule** (a sequence needed both a
  positive and a negative clipped target value to count at all, or was
  excluded entirely) does not appear in the corrected spec — there is no
  mention of block-level exclusion in the global formulation, since there
  are no longer per-block scores to exclude. Unconfirmed by an explicit
  organizer statement, but implied by the new spec's structure.
- Practical effect: sequences with more scored rows, or with larger
  |target| weights, now have proportionally more influence on the score
  than before. A model's per-sequence consistency matters less than its
  aggregate accuracy on the highest-weight rows across the whole dataset.

All required predictions (even on unscored rows) must still be finite and
correctly shaped — missing/invalid outputs are submission errors.

**Local reference implementation status:** the starter pack's `utils.py`
(`weighted_pearson`, `BlockAccumulator`, `ScorerStepByStep`) implements the
**old, pre-correction** per-block-averaged metric. It has **not** been
updated to match the corrected global-pooled formula as of this writing —
any local WP number computed with it is no longer a reliable estimate of
the real (corrected) platform score. This needs a new/updated local scorer
before local numbers can be trusted again for comparing future experiments
— see `EXPERIMENTS.md`.

**Baseline:** a stateful vanilla GRU (ONNX), 2 recurrent layers of hidden
size 128. Original (pre-correction) platform score: 0.5719 WP.
**Re-evaluated (corrected) platform score: 0.5935 WP.**

---

## 4. Expected output format / interface

Your `predict()` callback is driven **row by row** by the platform (or locally, by `ScorerStepByStep`). Each call receives a `DataPoint`:

```python
@dataclass(frozen=True)
class DataPoint:
    seq_ix: int            # sequence id (not a feature)
    step_in_seq: int        # 0 .. 19999
    need_prediction: bool   # False during warm-up (steps 0-98)
    state: np.ndarray        # 112 float32 features, fixed order (see table above)
```

Contract:

- Update internal state on **every** call, including warm-up rows.
- Detect a new sequence yourself (`seq_ix` changed) and **reset state** — there's no automatic reset.
- If `need_prediction` is `False` → return `None`.
- If `need_prediction` is `True` → return a **finite**, float32-compatible array of **shape `(2,)`**, `[t0, t1]`.
- Repeated runs on identical input must produce **identical** output (determinism is a hard requirement, checked during prize verification).

Minimal skeleton (from the submission guide):

```python
import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        self.seq_ix = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            # reset sequence state here

        # update model state using data_point.state (always)

        if not data_point.need_prediction:
            return None

        return np.zeros(2, dtype=np.float32)  # replace with real prediction
```

---

## 5. Deliverables / submission packaging

This is a **code competition**: you submit inference code + artifacts, not raw predictions.

- A single **`.zip`** archive.
- **`solution.py` at the root** of the zip, defining a no-argument `PredictionModel` class with `predict(self, data_point)`.
- Include any weights/model files/helper modules/configs alongside it; load them via **paths relative to `solution.py`** (works from any extraction location):

  ```
  submission.zip
  ├── solution.py
  ├── model.onnx
  ├── your_own_helpers/
  │   └── lib.py
  └── MODEL.md
  ```
- Package from inside the solution directory: `zip -r ../submission.zip .`
- Up to **5 submissions per day**.
- **Pre-upload checklist** (from the docs): `solution.py` at zip root + all required files present; correct output shape; correct warm-up handling and state resets; identical predictions across repeated runs; size/time limits respected.

No training code or checkpoint is required in the submission — only working inference code. (For prize verification, training code/report/etc. is required separately — see §7.)

---

## 6. Execution environment & resource constraints

Your submitted code runs standalone in an isolated container:

| Resource | Limit |
|---|---|
| CPU | 1 vCPU |
| RAM | 16 GB |
| GPU | **None** — CPU inference only |
| Inference time | **60 minutes** for the entire test set |
| Submission size | **20 MB** max (zipped) |
| Internet access | **Disabled** |
| Base image | `python:3.11-slim-bookworm` (Docker), Linux, local SSD |

Docker details that matter for reproducing the environment locally:

- ML library telemetry/network access is forced off via env vars (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `WANDB_DISABLED`, etc.), caches redirected under `/app/.cache`.
- The provided `requirements.txt` only pins `numpy==2.2.6`, `onnxruntime==1.23.2`, `pyarrow==19.0.1` — sufficient for the inference-only baseline. The actual scorer Dockerfile installs a broader "reasonable set of popular ML libs" (with a CPU-only PyTorch wheel index) — but the exact list is not given in the starter pack. **If you need a package not already in the image, you must request it from the organizers (Discord/email) — you cannot just `pip install` at inference time (no internet).**
- **Implication:** train with whatever tooling you want locally (PyTorch, JAX, etc.), but for **inference**, either (a) confirm your training framework is in the scoring image, or (b) export to a framework guaranteed to be present — the baseline's choice, **ONNX Runtime**, is the safe, portable option and is explicitly supported/threaded for the 1-vCPU limit.

Given 1,873 validation sequences × ~19,901 required predictions each (≈37.3M steps) must fit in 60 minutes, per-step latency budget is roughly **~100µs/step** at that scale — favors small/efficient models (the baseline's 128-hidden-unit 2-layer GRU is illustrative of the target scale) over large architectures.

---

## 7. Constraints on ML approach / fair play

- **No external data** — train only on the provided `train.parquet` (and `valid.parquet`, presumably for validation/model selection).
- **Pre-trained public models are allowed**, provided they are publicly available and **contain no external market data** baked into them (i.e., no smuggling in market knowledge via a pretrained finance model).
- Features are deliberately **anonymized** — no domain knowledge of what `a0`...`a7` or the exact target definitions mean is available or intended to be reverse-engineered as the "point" of the competition; the challenge is the general sequence-modeling/correlation task.
- **Determinism required**: identical inputs → identical outputs across runs (rules out unseeded stochastic inference, e.g. dropout left active, non-deterministic reductions that vary run to run).
- **No infra abuse**: don't attempt to crack the scoring system, leak data, or exploit the platform — disqualification, even if "impressive."
- **No sharing**: don't publish or privately share your code, models, or detailed strategy with other participants before the competition ends. General idea discussion is fine; full solutions are not.
- **Submission cadence**: max 5 submissions/day; one account per person/team (no in-platform teaming yet).
- Two leaderboards: **Public** (feedback on a sample of test data during the competition) and **Private** (hidden final dataset, determines actual final ranking). Near the end, you'll be asked to pick previously-submitted solutions to be re-scored on the final dataset.

---

## 8. Timeline & prizes

- **Start:** 2026-09-11
- **Final submission deadline:** 2026-11-15, 23:59 UTC
- **Winners announced:** 2026-12-01

Prize pool: **$13,600 USDT** across the top 8 on the Private Leaderboard:

| Place | Prize |
|---|---:|
| 1st | $5,000 |
| 2nd | $2,500 |
| 3rd | $1,700 |
| 4th | $1,300 |
| 5th | $1,000 |
| 6th | $800 |
| 7th | $700 |
| 8th | $600 |

**Prize verification** (top 8 only, within 7 days of preliminary results):
1. Inference code + model weights (the exact winning submission).
2. Full training code, with **fixed random seeds** (reproducibility is explicitly checked).
3. A 2–8 page technical report (approach, architecture, validation strategy).
4. A 30–60 minute code-review call with the organizers.
5. Grant of a non-exclusive license for internal (non-public) research use by the organizers.

Payout: USDT to your wallet within 14 days of verification; failed verification gets 5 days to fix issues before the prize passes to the next-ranked participant.

---

## 9. Practical takeaways for modeling

- Treat this as an **online sequence regression** problem with two correlated real-valued targets, evaluated by a **weighted correlation** (not MSE/MAE) — models should focus on getting the *direction and relative ranking* right on large-magnitude moves, since small/zero targets carry ~zero weight in scoring.
- A **recurrent architecture (GRU/LSTM) or comparable stateful/causal design** (causal attention window, state-space model, etc.) fits the "must update state row-by-row, causal-only" contract naturally — the baseline's GRU is a reasonable structural starting point, not necessarily the ceiling.
- Model size/speed must respect the **60-minute / 1-vCPU / no-GPU** inference budget — favor compact architectures and cheap per-step inference; **ONNX export** is the proven path to a fast, portable, dependency-light submission.
- Validate locally with `utils.ScorerStepByStep` against `datasets/valid.parquet` to get a **platform-accurate WP score** before submitting, remembering the block-eligibility rule can silently drop sequences from your local score if your validation subset is small or biased.
