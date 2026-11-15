# 018: Nonlinear-expanded + rolling-normalized grouped-encoder LSTM

Combines the two feature-engineering levers that have actually moved WP in
this project, at full capacity for both, rather than testing either alone:

1. **Experiment 012's 5x nonlinear expansion** — every raw group
   (price/volume/trade/additional) expanded into
   `[x, asinh(x), sin(x), cos(x), tanh(x)]` before its own MLP encoder.
   Unchanged from 012.
2. **Experiment 013's rolling EWMA-normalization, price-only, 2 windows
   (50/200 steps), vectorized across windows.** Unchanged from 013's
   latency-fixed version — fixes #1 (vectorization) and #3 (price-only
   scoping) are baked in from the start here, not retrofitted after a
   timeout like 013's own history.

## Why this experiment

See `EXPERIMENTS.md`'s "Open directions for future experiments" section —
this is the highest-priority not-yet-tried idea it identified. Neither 012
nor 013 alone answers whether *temporal* normalization (EWMA) carries
signal beyond what *static* nonlinear transforms (012) already capture:
013-alone (price-only EWMA, no nonlinear expansion) scored **0.678768**
local WP, below both plain 004 (0.682425) and 012 (0.682697) — so this
combination is a genuine open question, not a pre-decided win.

## How the two are combined for price

Only price gets both treatments; volume/trade/additional get exactly 012's
nonlinear expansion only (unchanged — 013's own excess-kurtosis argument
for why rolling normalization is price-only still applies).

Per-step price encoder input, additive, no duplication:

```
expand(price_raw)                    44*5 = 220   (012's nonlinear 5x)
+ ewma_normalized_blocks_only         44*2 =  88   (013's 2 EWMA windows;
                                                     the raw block from
                                                     013's own output is
                                                     dropped here since
                                                     expand()'s identity
                                                     channel already
                                                     carries raw price —
                                                     no need to duplicate)
= price_encoder input width                  308
```

Volume/trade/additional: byte-for-byte identical to 012.

## Pre-training checkpoint (2026-10-06, random-init weights)

- **Export correctness**: verified against batched `model.forward()`,
  **2.980e-08 max diff** (well within the established atol=1e-2, same
  bound 013 uses for the same reason — the EWMA half of this model has the
  same training/export discrepancy 013 validated; the nonlinear half is
  identical on both paths and contributes no discrepancy of its own).
- **ONNX node count: 76** (`Slice` 13, `Concat` 9, `Add` 7, `Mul` 6,
  `Reshape` 5, `MatMul` 5, `Asinh`/`Sin`/`Cos`/`Tanh`/`Relu`/`Transpose` 4
  each, `Sub` 2, `LSTM` 2, `Clip` 1). For context: 012 alone is 62 nodes,
  013 (price-only, 2-window, vectorized) alone is 58 nodes — the
  combination's 76 is close to simple addition minus the shared base
  (62+58-43≈77), confirming the two feature paths compose close to
  independently rather than interacting in some node-count-multiplying way.
- **Latency: 146.1% of budget uncorrected** (random-init weights,
  single-threaded synthetic benchmark, same methodology as every other
  experiment). **Over budget.** This matches the pre-implementation
  prediction in `model.py`'s module docstring (140-170% expected) almost
  exactly, and lands worse than this project's empirically-observed safe
  threshold (≤125% has always scored on the platform across 21
  submissions; ≥180% has always timed out; 146.1% is solidly in between,
  closer to the risky end — see `analysis/results/timing_correlations.md`).

**Not trainable as-is without a further latency cut, per the measurement
above** — but per explicit user instruction, training was started anyway
on 2026-10-06 (same 30-epoch warmup+decay+early-stop recipe, GPU), the same
"flag the risk, train anyway for the local-comparison result" pattern this
project used for 012/013-original/014-016. If this submission times out
later, that would be the expected outcome given 146.1% uncorrected, not a
surprise — consistent with how every other over-budget submission in this
project's history has been read.

## Next step (not yet decided / not yet implemented)

The most direct lever, mirroring 013's own history: **drop from 2 EWMA
windows to 1** (keep span=200 only, drop span=50 — or vice versa; which one
carries more signal is itself an open question, unlike 013's "keep both
extremes" reasoning when going from 4→2). This would roughly halve the
EWMA-specific node and state-I/O cost. Not implemented yet — pending a
decision on which single window to keep, and whether that's preferable to
other levers (e.g., dropping one of the 4 nonlinear transforms, since not
all of `[asinh, sin, cos, tanh]` have been individually ablated for
usefulness here or in 012).

## Files

- `model.py` — `NonlinearRollingNormGroupedEncoderLSTMModel`, the combined
  architecture. Full design rationale and the node-count/latency
  expectation are in its module docstring.
- `dataset.py`, `train.py` — generic, same recipe (30 epochs, warmup +
  `ReduceLROnPlateau` + early stopping) as every 30-epoch experiment in
  this project. **Do not run training until the latency question above is
  resolved.**
- `export_onnx.py` — exports `checkpoints/model.onnx`, verified against
  batched `forward()`.
- `solution.py` — competition entrypoint; threads (h, c) and the EWMA
  (mean, meansq) state between steps, identical contract to 013's.

## Results (2026-10-08)

Trained on the server A backup server (torch 2.2.0, shared L40S) with the
standard 30-epoch recipe; weights copied back here and checksum-verified,
then deleted from server A. An earlier local run was interrupted; its
partial files are in `checkpoints_interrupted_local_backup/` and
`logs/train_interrupted_local.log`.

- **Training:** early-stopped at epoch 13, best valid_loss **0.3439** at
  epoch 8. Effectively tied with 012 (0.3438) and 013 (0.3440).
- **Export on trained weights:** max diff 5.662e-06 vs. batched forward.
- **Latency on trained weights:** **135.6% of budget uncorrected**
  (146.1% on random-init). 271,934 params, 76 nodes.
- **Local WP:** **0.681888** (t0=0.676098, t1=0.687677) — below 012
  (0.682697), 017 (0.682929) and 004's 30-epoch run (0.682425).
- **Platform:** **0.6539** (ID 03PQ153A), runtime **54m7s** (90.2% of
  budget, implied speedup 1.50x). Did not time out despite the latency
  flag — the highest-latency submission to clear so far — but with only
  ~6 minutes of margin, so this is not a safe operating point to design
  toward.

**Answer to the question this experiment was built for:** no. Temporal
EWMA normalization (price-only, 2 windows) adds nothing measurable on top
of 012's static nonlinear expansion: local WP −0.0008 and platform
−0.0017 versus plain 012, both inside the project's ~0.002 noise but
pointing the wrong way, at +14 ONNX nodes and +23 points of uncorrected
latency. 012/017 remain the better submissions.

Submission zip: `submissions/018_nonlinear_rolling_norm_encoder_2026-10-08.zip`
(already submitted).
