# Feature & Target Analysis

Exploratory analysis of all 112 input features and both targets (`t0`, `t1`)
from `train.parquet` — computed over the **complete dataset**: all 10,607
sequences, 212,140,000 rows, no sampling. Read this after
`PROBLEM_STATEMENT.md` — that document describes the *schema*; this one
describes what the data actually *looks like*. See also
**[MARKET_GLOSSARY.md](MARKET_GLOSSARY.md)** for what the feature *types*
(bid/ask, price/volume, trade data) mean in general market terms, and
**[results/feature_statistics.md](results/feature_statistics.md)** for the
full per-column statistics table with a plain-language glossary of every
statistic.

## How this was computed

An earlier version of this analysis used a random sample of sequences to
avoid the cost of reading the full ~95GB file. That turned out to matter:
re-running on the complete dataset revealed the sample had significantly
**understated the extremity of the target distributions** (see Finding 1
below) — bulk statistics (means, typical scale, correlation patterns) were
fine from a sample, but rare extreme-tail behavior was not, precisely
because it's rare. So this version reads every row.

To do that without loading 95GB into memory or reading the file more than
once, `scripts/compute_statistics.py` is a **single-pass streaming
aggregator**:

1. **Exact scalar statistics via raw-moment accumulation.** For each of the
   114 columns, running sums of `x`, `x²`, `x³`, `x⁴` (float64) are
   accumulated one sequence at a time, along with exact min/max and exact
   zero/negative/positive counts — O(1) memory per column regardless of
   dataset size. Mean, variance, skewness, and excess kurtosis are derived
   from these sums afterward using standard raw-moment identities
   (e.g. `Var = E[x²] − E[x]²`) — exact, not approximated, and verified
   against direct NumPy computation on a held-out check before trusting the
   full run.
2. **Exact Pearson correlations**, feature-vs-target and t0-vs-t1, via the
   same trick extended to `sum(x·y)` — also verified against direct
   computation.
3. **Histograms** (for percentiles and the plots) use pre-set bins, since a
   true single-pass histogram needs bin edges decided before seeing all the
   data. Bins are seeded from *this script's own previous output* if one
   exists (self-correcting on rerun — see `seed_histogram_ranges()`), with
   a generous hardcoded fallback for a completely fresh run. Any value
   landing outside the assumed range is counted exactly
   (`hist_underflow`/`hist_overflow` in `feature_statistics.csv`) rather
   than silently dropped or clipped.

One read of the full file took **57.9 minutes**. On the run that produced
these results, the very first one, bin ranges were seeded from an earlier
sample and turned out too narrow for the targets specifically: **665/212.14M
(0.0003%) of t0 values and 1,288/212.14M (0.0006%) of t1 values** fell
outside the assumed histogram range. This does **not** affect any scalar
statistic reported (those come from the exact raw-moment sums, not the
histogram) and is far too small a fraction to shift any reported percentile
— it only means the very rarest extreme events don't get their own visible
bar in the target plots. A rerun now would self-correct (verified: rerunning
`seed_histogram_ranges()` after this run picks up the true measured extremes
automatically).

### Reproducing / extending this

```bash
cd analysis/scripts
python compute_statistics.py         # ~58 min: full pass -> ../results/*.csv, ../data/*.npy
python generate_plots.py             # seconds: -> ../figures/*.png
python generate_stats_markdown.py    # seconds: ../results/feature_statistics.csv -> feature_statistics.md
```

## Folder structure

```
analysis/
├── README.md                          - this file
├── MARKET_GLOSSARY.md                 - what bid/ask, price/volume, trade data mean (general domain knowledge)
├── scripts/
│   ├── compute_statistics.py          - single-pass full-dataset stats + correlations -> CSV
│   ├── generate_plots.py              - all PNGs below, from the saved histograms
│   └── generate_stats_markdown.py     - readable, glossary-annotated version of the stats CSV
├── data/
│   ├── histogram.npy, bin_edges.npy   - full-dataset histograms (regenerable via compute_statistics.py)
│   └── histogram_columns.txt
├── results/
│   ├── feature_statistics.csv         - one row per column (114 total): exact count, mean,
│   │                                     std, min, max, percentiles, skew, kurtosis,
│   │                                     frac_zero/negative/positive, hist_underflow/overflow
│   │                                     (kept for programmatic use - compute_statistics.py
│   │                                     reads it back on a rerun to seed histogram ranges)
│   ├── feature_statistics.md          - the same data, for reading: grouped tables + a
│   │                                     plain-language glossary of what each statistic means
│   ├── feature_target_correlations.csv - each feature's exact Pearson corr with t0 and t1
│   └── target_correlation.txt          - exact corr(t0, t1)
└── figures/
    ├── group_i0_bid_price.png / group_i0_ask_price.png       (i0_p0-p10 / p11-p21)
    ├── group_i0_bid_volume.png / group_i0_ask_volume.png     (i0_v0-v10 / v11-v21)
    ├── group_i0_trade_price.png / group_i0_trade_volume.png  (i0_dp0-3 / dv0-3)
    ├── group_i1_* (same six groups, mirrored for instrument i1)
    ├── group_additional.png                                   (a0-a7)
    ├── targets.png                                             (t0, t1, log scale)
    ├── targets_raw.png                                         (t0, t1, raw/linear scale)
    └── feature_target_correlations.png                        (all 112, bar chart)
```

## Key findings

### 1. The targets are far more extreme-tailed than a 1.4% sample suggested

Exact, full-dataset numbers:

| | t0 | t1 |
|---|---:|---:|
| mean | -0.272 | -0.267 |
| std | 1.006 | 1.011 |
| min / max | **-125.57 / 74.50** | **-179.92 / 77.31** |
| excess kurtosis | **22.36** | **90.10** |
| % exactly 0 | 21.9% | 21.8% |
| corr(t0, t1) | colspan | **-0.7223** |

For comparison, a 150-sequence (1.4%) sample had estimated kurtosis at 12.04
(t0) and 5.76 (t1), and min/max around ±10-16 — the *direction* of every
finding held up, but the *magnitude* of how extreme the tails are was
substantially understated, especially for t1 (sample kurtosis 5.76 vs. the
true 90.10 — off by more than 15x). This makes sense: kurtosis is driven by
the rarest, largest events, which by definition a small sample is unlikely
to capture. **This is the clearest justification in this whole analysis for
why the complete-dataset pass mattered, not just a sample.**

The ~22% exact-zero rate and the strong t0/t1 anti-correlation (-0.72) are
confirmed essentially unchanged from the sample — those are bulk-frequency
effects, not rare-tail effects, so a sample estimated them fine.

Two versions of the target distribution plot are kept side by side:
`figures/targets.png` (log-scale y-axis) and `figures/targets_raw.png`
(raw/linear y-axis, no transformation). The raw version makes the point
above vivid on its own — the exact-zero spike is so tall it visually
swallows everything else, including the entire fat tail out to ±125/180.
That's not a plotting bug, it's the actual shape of the data; the log-scale
version exists specifically to make the tail visible, which the raw
version cannot do at this scale.

**Why this matters for modeling**: the WP metric weights each row by
`|target|`, automatically down-weighting the zero-inflated rows in scoring.
But the *very* fat tails (kurtosis in the tens to ~90) mean a handful of
rows carry enormous weight and enormous target magnitude — a training loss
that isn't robust to this (e.g. plain unweighted MSE, which is dominated by
squared error and therefore by the largest outliers) will be disproportionately
shaped by a tiny number of extreme rows. The magnitude-weighted losses
already used in `shared/losses.py` are a reasonable response to this, but
the degree of fat-tailedness found here (kurtosis ~90 for t1) suggests it's
worth explicitly checking training stability/gradient magnitude on the most
extreme rows, not just trusting the loss function handles it gracefully.

### 2. t0 and t1 are strongly, and even more strongly than estimated, negatively correlated

`corr(t0, t1) = -0.7223` (exact, all 212.14M rows) vs. the sample's -0.7051
estimate — same conclusion as before, now on firmer footing. See finding 1's
"why this matters" in the earlier draft: worth trying to model this
correlation explicitly (e.g. predicting `t0` and `t0 − t1`) rather than two
fully independent output heads.

### 3. Price-like features are heavily quantized/multi-modal, not smoothly continuous

Confirmed unchanged on the full dataset (see `group_i0_bid_price.png` etc.)
— every `i0_p0`-`i0_p10` shows a sharp multi-spike histogram, each index at
a different characteristic value/range, consistent with the docs' note that
these are level *identifiers*, not depth-sorted continuous quantities.
Price-like features also have visibly lower variance (mean std ≈ 0.144-0.154)
than volume/trade features (≈ 0.65-0.81) — see the group-scale table below.

### 4. A symmetric clip at exactly ±5.1993... is far more widespread than a small sample shows — but wildly uneven in frequency

On the full dataset, **51 of 114 columns hit the exact floor `-5.199337482452393`,
and 54 hit the exact ceiling `+5.199337482452393`** — compare the earlier
sample's count of just 12. This is the second-clearest example (after
finding 1) of why completeness mattered: most of these are extremely rare
events for most columns, invisible in a 1.4% sample.

But frequency varies enormously by column. Checking `i0`'s price group
specifically: only **`i0_p0`** among all 22 `i0_p*` columns ever hits the
floor, and even then it's rare (a handful of times in 212 million rows) —
essentially every other `i0` price feature never approaches this bound at
all. Contrast with `a5`, `a6`, `a7` (finding 5) and most trade-related
(`dp`/`dv`) and `i1` features, which hit these bounds regularly enough to
show up clearly even in a small sample. **The practical read: almost every
feature has some theoretical exposure to this clip, but for the great
majority of `i0`'s core price/volume signal it's a non-event; for a specific
subset (trade features, most of `i1`, and especially `a5`/`a6`/`a7`) it's a
real, common, structural feature of the data.**

### 5. `a5`, `a6`, `a7` are bimodal — a likely sentinel-value encoding, confirmed on the full dataset

`a7`'s exact percentiles: p1 = -5.214, p5 = -5.211, **p25 = -5.193** — still
essentially at the clip floor, confirming **at least 25% of all `a7` values
across the entire training set** are the sentinel value, not a rare
artifact. `a5`/`a6` show the same pattern at the p1/p5 level, with real
(non-sentinel) values appearing from around p25 onward. See
`group_additional.png`: a large isolated spike at the floor, a gap, then a
separate bell-shaped bump — visually distinct populations, not a smoothly
clipped continuous distribution.

**Why this matters**: as before — an explicit `is_a5_sentinel` /
`is_a6_sentinel` / `is_a7_sentinel` indicator feature (derived from
`value == floor`) is a cheap, concrete experiment, now with more confidence
this affects a genuinely large fraction of rows (≥25% for `a7`), not a rare
edge case.

### 6. Trade-volume features are dominated by a "no trade happened" zero-spike

Unchanged from the sample: `i0_dv0`-`i0_dv3` and `i1`'s equivalents show one
dominant spike at/near zero (most rows have no trade in that slot) with a
sparse, roughly symmetric spread of nonzero values around it.

### 7. No single feature is a strong linear predictor — `i0`'s own features carry most of the (modest) linear signal

Best exact full-dataset correlations: `i0_p3` (+0.182 with t0), `i0_p20`
(-0.148 with t1) — both very close to the sample's estimates (+0.174,
-0.146), confirming correlation *rankings* were reliable from the sample
even though the *targets'* own tail statistics were not. Every other
feature correlates more weakly still. `i1`'s features and the `a0`-`a7`
features cluster near zero correlation with both targets (see
`feature_target_correlations.png`) — consistent with `i1` and the
additional features contributing mostly nonlinear/contextual signal, if
they contribute meaningfully at all through a purely linear lens.

## Summary table of per-group scale (exact, full dataset)

| Group | mean(std) | mean(mean) |
|---|---:|---:|
| i0 price | 0.144 | -0.002 |
| i0 volume | 0.689 | -0.026 |
| i0 trade price | 0.627 | -0.031 |
| i0 trade volume | 0.653 | +0.017 |
| i1 price | 0.154 | -0.004 |
| i1 volume | 0.666 | +0.045 |
| i1 trade price | 0.701 | +0.012 |
| i1 trade volume | 0.810 | -0.020 |
| additional (a0-a7) | **1.454** | **-0.346** |

Essentially unchanged from the sample's estimates (e.g. i0 price 0.144 vs.
0.156, additional 1.454 vs. 1.469) — this table is a case where the sample
was already trustworthy, since it's a bulk/typical-scale statistic, not a
rare-tail one.

## What the sample got right vs. wrong (a note for future analyses on this data)

Worth internalizing for next time: the earlier 1.4%-sample analysis was
**accurate for bulk/typical statistics** — group-level scale, correlation
rankings, the existence of quantization/zero-inflation/sentinel patterns —
but **understated rare-tail extremity** by a wide margin (kurtosis off by up
to ~15x, the ±5.1993 clip's column count off by ~4x). If a future analysis
on this dataset only cares about typical behavior, a sample is probably
fine and much cheaper. If it touches anything about outliers, tail risk, or
"how often does X happen" for a genuinely rare X, use the complete dataset —
this is exactly the class of question a small sample structurally cannot
answer reliably, no matter how "representative" the sample seems otherwise.

## Ideas this suggests for future experiments (not yet tried)

- Add `is_sentinel` indicator features for `a5`/`a6`/`a7` (finding 5) — now
  confirmed to affect ≥25% of rows for `a7`, not a rare edge case.
- More broadly, consider a generic `is_clipped` indicator (`value == ±5.1993`)
  for any of the 51-54 affected columns where it's not already captured by
  an `a5`/`a6`/`a7`-style dedicated indicator — finding 4 shows this is a
  real, if unevenly distributed, structural feature of the preprocessing.
- Try predicting `(t0, t0 − t1)` or otherwise encoding the strong `t0`/`t1`
  anti-correlation explicitly (findings 1-2), rather than two fully
  independent output heads.
- Given how fat-tailed the targets truly are (kurtosis up to ~90), explicitly
  check training stability/gradient behavior on the most extreme rows rather
  than assuming a magnitude-weighted loss alone handles it gracefully.
- Compare per-feature-group normalization (price vs. volume vs. trade vs.
  additional normalized separately) against uniform treatment, given how
  different their scales/shapes are (findings 3, 4, 6, and the group-scale
  table above).
- Investigate whether the exact-zero target rows (finding 1) correlate with
  anything observable in the features (e.g. specific `step_in_seq` values,
  or periods of zero trade volume) — not checked here.
