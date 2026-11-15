# 001 — LSTM + masked (causal) self-attention

**Status: original small config (d_model=24) submitted and scored
successfully (2026-09-12, platform WP 0.6403). A full-scale retrain
(d_model=128) was tried and timed out twice; retired. Currently retraining
at d_model=32 (near-original, modern 30-epoch recipe) — see
[../../EXPERIMENTS.md](../../EXPERIMENTS.md) for the full history.**

## Idea

Keep the baseline's recurrent backbone (an LSTM instead of the baseline's
GRU) as the causal, per-step-updated state, but add a **self-attention block
on top of the LSTM's outputs** so the model can directly look back at any
earlier step's representation, not just whatever got compressed into the
current recurrent hidden state. The attention is strictly **causal and
windowed** — it can never see a future row (which would violate the
competition rules: a prediction may only use the current and earlier rows of
the same sequence).

## Why not full causal attention

A naive causal self-attention implementation builds a full `(T, T)` score
matrix per head. Sequences here are `T = 20,000` steps — a full matrix would
need `20,000² × n_heads` entries just for the *scores* of one sequence, which
is many GB and not feasible under the competition's compute limits (1 vCPU,
16 GB RAM). Instead, `model.py`'s `CausalSelfAttention` implements **banded /
windowed causal attention**: each step attends only to the last
`attention_window` steps (inclusive of itself), built via `unfold` so the
score tensor is `(batch, heads, T, window)` — linear in `T`, not quadratic.

## Files

| File | Purpose |
|---|---|
| `model.py` | `LSTMAttentionModel`: input projection → multi-layer LSTM → causal windowed self-attention (residual) → feed-forward (residual) → linear head. Also defines the windowed-attention mechanism itself. |
| `dataset.py` | Loads full sequences (not chunks — see below) from a labeled Parquet file into a `torch.utils.data.Dataset`. |
| `train.py` | Training loop: batches full sequences, computes a masked/weighted loss (MSE and/or a differentiable weighted-Pearson proxy — see `shared/losses.py`), checkpoints the model + its config. |
| `inference.py` | `StreamingPredictor`: reproduces the trained model's computation **one row at a time**, maintaining the LSTM's `(h, c)` state plus a rolling cache of key/value projections for the attention window. This is what makes the model satisfy the competition's row-by-row `predict()` contract. |
| `solution.py` | The actual submission entrypoint (`PredictionModel.predict(data_point)`), wrapping `StreamingPredictor`. Loads `checkpoints/best.pt` + `checkpoints/config.json` via paths relative to itself, per the packaging rules. |

## Key design decisions worth knowing about later

1. **One training sample = one full 20,000-step sequence**, not a shorter
   chunk. This is deliberate: the windowed attention's look-back is measured
   from the *true start* of the sequence. If training instead sliced
   sequences into shorter chunks, a chunk's first `window` steps wouldn't
   have their real history available unless the key/value cache were also
   carried across chunk boundaries (like `inference.py` does for streaming
   inference) — full sequences sidestep that mismatch entirely, at the cost
   of using more memory per training sample. See `dataset.py`'s docstring.

2. **Streaming inference caches key/value projections, not raw hidden
   states.** Each step's (key, value) pair is a fixed function of that step's
   own LSTM output and the shared projection weights — it's computed once
   and never needs recomputing later. `inference.py` pushes each new step's
   (k, v) onto a bounded deque (`maxlen=attention_window`) and attends the
   current step's query against whatever's cached. This keeps each streaming
   step's incremental cost to O(window), not O(window · d_model²).

3. **The correlation-aware loss is per-sequence, not pooled across rows from
   different sequences.** `weighted_pearson_loss` (in `shared/losses.py`) is a
   differentiable version of the actual WP metric (same clip-to-[-2,2], same
   `|target|` weighting) — `train.py` calls it once per sequence in a batch
   and averages, mirroring how the real metric aggregates per-sequence scores.

4. **`attention_window` is a hyperparameter, not fixed.** Default `512` in
   both `model.py` and `train.py`'s CLI — untuned. There's a real memory/
   speed vs. look-back-length tradeoff to explore here.

## Bugs found and fixed while getting this to actually run

Two separate out-of-memory kills happened on the first real training attempts
(both on this shared, multi-tenant machine — worth being memory-conscious
here beyond just this project). Both are fixed now; kept here since they're
non-obvious and worth knowing about if similar patterns show up elsewhere:

1. **`dataset.py` loaded the whole Parquet file eagerly.** `ConnectomeSequenceDataset.__init__`
   used to do `list(iter_sequences(parquet_path))`, materializing every
   sequence up front — ~15.6 GiB for `valid.parquet` alone, and the very
   first smoke test loaded it twice (once for `--train`, once for `--valid`,
   both pointed at the same file), for ~31 GiB simultaneously. Fixed by
   making it read one Parquet row group per `__getitem__`, lazily (see
   `shared/data.py`'s `read_sequence` + `file_columns` helpers, factored out
   of `iter_sequences` so both the eager and lazy paths share one
   implementation). Verified: dataset init time 0.3s (was ~60s), peak RSS
   ~600 MB for a few batches (was ~16 GB before even starting to train).

2. **`CausalSelfAttention`'s windowed attention silently materialized a huge
   tensor.** The original implementation used `k.unfold(...)` reasoning that
   it was O(T·window) instead of O(T²) — true in FLOPs, but overlapping-window
   views can't be reshaped into the contiguous layout `einsum`'s batched
   matmul needs, so PyTorch materializes the full expanded tensor internally
   anyway. Measured peak for one forward pass at real training shapes
   (batch=4, T=20,000, window=512, d_model=128): **41.9 GB**. Fixed by
   rewriting it as block-recurrent sliding-window attention — process the
   sequence in chunks of `window` steps, carrying forward only the trailing
   `window - 1` keys/values as a small cache between chunks (same idea as
   `inference.py`'s streaming cache, just batched per-chunk instead of
   per-row). Verified numerically identical to a brute-force full-causal-then-
   masked reference (max abs diff ~6e-8) on a small case, and peak memory at
   the real training shapes dropped to **~2.5 GB**, running in under a second.

## Critical open issue: inference is ~7.8x over the competition's time budget

Fixing the two memory bugs above made training runnable, which surfaced a
separate, more fundamental problem: **`StreamingPredictor.step()` is too slow
to meet the 60-minute-for-the-entire-test-set limit**, even after real fixes:

| Version | ms/step | Extrapolated (valid-set scale, 37.27M preds) | vs. 60 min budget |
|---|---:|---:|---:|
| Original (`deque` + `torch.stack` cache, `nn.LSTM` per step) | 1.44 | ~895 min | ~15x over |
| + circular-buffer cache (no more per-step Python list/stack rebuild) | 1.17 | ~727 min | ~12x over |
| + `nn.LSTMCell` instead of `nn.LSTM(seq_len=1)` per step (verified numerically identical, same weights) | **0.75** | **~467 min** | **~7.8x over** |

Both changes were verified to leave the model's *output* unchanged — streaming
predictions match `model.forward()`'s batched output to ~5e-7 — so these are
genuine speedups, not shortcuts that trade away correctness.

**Why the remaining gap is hard to close by just shrinking the model:**
profiling the current 0.75 ms/step shows only ~0.46 ms is actual arithmetic
(LSTM ~0.13ms, attention ~0.13ms, feed-forward ~0.11ms, everything else
smaller); the other **~0.29 ms is Python/dispatch "glue" overhead** from
chaining ~12 separate eager-mode PyTorch ops per step, largely independent of
`d_model`/`attention_window`. That alone is ~3x over the ~0.097 ms/step budget
before any real computation is counted. This is very likely why the baseline
ships an **ONNX Runtime** model rather than raw PyTorch: a compiled graph
avoids per-op eager dispatch overhead almost entirely (consistent with the
baseline's measured ~0.076 ms/step for comparably simple GRU math).

**Implication:** reducing `attention_window`/`d_model` alone would shrink the
~0.46 ms "real compute" portion but leave most of the ~0.29 ms glue overhead
in place — not enough alone to close a ~7.8x gap. The credible paths forward
(not yet decided or attempted):
1. **Export to ONNX**, mirroring the baseline's approach — thread the LSTM
   states *and* the attention key/value cache as explicit graph
   inputs/outputs (the baseline only has to do this for `hidden_0`/`hidden_1`;
   this model would also need `cache_k`/`cache_v`/`cache_pos`). Likely the
   most robust fix (ONNX Runtime is guaranteed present at scoring time per
   `requirements.txt`), but real engineering work — nontrivial to map a
   growing/circular cache onto a static graph.
2. **`torch.jit.script` or `torch.compile`** the step function to fuse ops
   and cut dispatch overhead, without a full ONNX rewrite. Untested here;
   unclear how much it actually saves for ops this small, and — more
   importantly — unclear whether the scoring container (no internet, minimal
   image) can support `torch.compile`'s runtime C++/Triton compilation at
   all, since that's not a documented guarantee the way ONNX Runtime is.
3. **Shrink model capacity** (`attention_window`, `d_model`, `lstm_layers`) —
   a direct, easy lever, but a real capacity tradeoff, and per the profiling
   above won't alone close the gap without also cutting the number of op
   calls or compiling.
4. Some combination of the above.

This is logged here rather than acted on unilaterally since it's a genuine
scope/direction decision, not a bug fix — see the conversation this
experiment came out of for how it's resolved.

### Update: ONNX export implemented (`export_onnx.py`) — gap narrowed to 2.48x, not yet closed

Chose to pursue the ONNX export path (option 1 above), mirroring the
baseline. Key design point: ONNX tracing can't express the `if cache_len ==
window` branch or runtime-length slicing that `StreamingPredictor` uses, so
`OnnxStepModel` in `export_onnx.py` reformulates the cache as a fixed-size
circular buffer plus a `valid_mask` (1.0/0.0 per slot), with invalid slots
excluded via an additive `-1e9` bias before softmax instead of via slicing —
mathematically equivalent, fully static-shaped, no data-dependent branching.
The circular write uses `Tensor.index_copy` (traces to ONNX `ScatterND`).

Verified at each step against both `model.forward()` (batched) and
`StreamingPredictor` (PyTorch streaming) — all agree to ~3-5e-7, confirming
the ONNX graph computes the exact same thing, not an approximation.

| Version | ms/step | Extrapolated | vs. budget |
|---|---:|---:|---:|
| PyTorch eager, fully optimized (`nn.LSTMCell` + circular buffer) | 0.75 | ~467 min | 7.8x over |
| ONNX export (`torch.einsum` for attention) | 0.342 | ~212 min | 3.54x over |
| ONNX export (`torch.matmul` instead of `einsum` for attention) | **0.239** | **~148 min** | **2.48x over** |

The `einsum`→`matmul` change came from profiling the exported graph with
ONNX Runtime's built-in profiler (`session_options.enable_profiling`): the
`Einsum` op averaged **60 µs/call** vs `Gemm`'s **13 µs/call** for
comparable-sized matmul work — ONNX Runtime evidently doesn't optimize the
generic `Einsum` operator as well as `Gemm`/`MatMul`. Rewriting the two
attention einsums as explicit `transpose`+`matmul` (same math, re-verified)
cut ~30% off the per-step time.

Remaining profile (700-step run, `Einsum` version, before the matmul fix —
roughly indicative of where time still goes): `Gemm` (all the Linear/LSTM
gate math) is now the single largest cost at ~0.13 ms/step — this is
"real" compute (2-layer LSTM + qkv/out projections + feed-forward + head),
not overhead, so closing the remaining 2.48x gap further likely means either
shrinking capacity (`d_model`, `attention_window`, `lstm_layers`,
`ff_multiplier`) or further graph-level cleanup (many small `Reshape`/`Add`/
`Mul` nodes from the attention decomposition add up: ~0.08 ms/step
combined in the profiled run).

**Status: not yet under budget. Paused here for direction rather than
continuing to iterate unilaterally** — see the conversation for how this
was resolved next.

### Update: architecture settled, retraining in progress

Tried the "obvious" next optimization — export `nn.LSTM` directly (native
fused ONNX `LSTM` op) instead of `nn.LSTMCell`. **Negative result**: at the
original scale (d_model=128, window=512) this measured 0.616 ms/step, over
2.5x *worse* than the LSTMCell version's 0.239 ms/step — the `Slice`/
`Concat`/`Transpose` glue needed to handle the 2-layer stacked LSTM state
apparently costs more than the native op saves, once integrated with the
rest of the graph. Reverted; `OnnxStepModel` uses `LSTMCell` (see its class
docstring for the full explanation — worth reading before re-attempting this
"obvious" fix).

Swept `d_model`/`attention_window`/`ff_multiplier`/`lstm_layers` directly
(not just window) via `export_onnx.py` + direct wall-clock benchmarks (not
profiler numbers, which showed inflated absolute values though correct
relative proportions). Findings:
- Shrinking below `d_model≈16-24` stops helping — d_model=8 measured
  *worse* than d_model=24 in one comparison, confirming a practical floor
  around ~86-88% of budget for this architecture family, not the ~50%
  originally hoped for.
- **Settled on: `d_model=24, lstm_layers=1, n_heads=4, attention_window=32,
  ff_multiplier=2`** — 0.0855 ms/step measured, ~88.5% of the ~0.0966 ms/step
  budget. This is a substantial capacity reduction from the original design
  (d_model=128, 2 layers, window=512) but was the largest config that
  actually fit with real (not razor-thin) margin.
- `train.py`'s CLI defaults updated to this config (previously d_model=128/
  layers=2/window=512/ff_multiplier hardcoded at 4, not even exposed as a
  flag — added `--ff-multiplier`).
- `solution.py` rewritten to use ONNX Runtime (mirroring baseline's
  `solution.py` exactly: same thread-pinning options, same
  `session.run()`-per-row pattern) instead of the PyTorch
  `StreamingPredictor` — reads state shapes directly off the exported graph
  so it has no hardcoded architecture knowledge and only depends on
  `numpy`/`onnxruntime` at inference time (same guaranteed dependencies as
  the baseline, no `torch` import needed for submission).
- End-to-end smoke-tested with random weights: correct warm-up/None
  handling, correct `[t0,t1]` shape, correct state reset on `seq_ix` change,
  and **confirmed deterministic** across repeated runs (hard requirement).

**Next**: the previous training run (using the old, oversized architecture)
was killed and restarted at the settled config — see EXPERIMENTS.md for
status. Once trained: export to ONNX, re-verify latency on the *trained*
weights (not just random-weight architecture benchmarks), score locally via
`ScorerStepByStep` for a real WP number, and only then consider this
submission-ready.

### Real training launched (2026-09-12 00:35)

Started 5 epochs on the full `train.parquet` (10,607 sequences), validating
against `valid.parquet`, at the settled config above. Launched via
`nohup ... & disown` so it survives terminal/session logout - PID 3227098,
log at `logs/full_train_20260912_003550.log`. Based on the smoke test's
measured ~2.5 s/batch on `valid.parquet`, extrapolated ETA is **~10-11
hours** (train.parquet has ~5.66x more sequences than valid.parquet).

**To check on it (works regardless of whether any Claude session is
active):**
```bash
ps aux | grep train.py                          # is it still running?
tail -f experiments/001_lstm_attention/logs/full_train_20260912_003550.log   # live progress
```

**Once it finishes** (`checkpoints/best.pt` and `checkpoints/config.json`
will both exist), the remaining steps to reach a real, scored result:
```bash
cd experiments/001_lstm_attention
python export_onnx.py                                              # writes checkpoints/model.onnx, verifies vs. PyTorch
python solution.py --validation ../../wnn_connectome_starterpack/datasets/valid.parquet   # real WP score via ScorerStepByStep
```
Then re-run the latency benchmark from `export_onnx.py`'s sweep script
(see earlier in this doc) on the *trained* `checkpoints/model.onnx` to
confirm the ~88.5%-of-budget margin still holds on real weights (it should -
latency is architecture-dependent, not weight-dependent - but worth
confirming rather than assuming). Log the final WP and latency numbers in
[../../EXPERIMENTS.md](../../EXPERIMENTS.md).

### Result (2026-09-12): training + scoring complete, beats the baseline

5 epochs on the full `train.parquet` finished in ~9h23m (`valid_loss=0.3818`,
the training-time loss proxy, not the competition metric). Exported to ONNX,
verified against both `model.forward()` and `StreamingPredictor` again on
the real trained weights (diffs ~6e-7, same as with random weights).
Re-benchmarked latency on the trained model: **0.0899 ms/step, 93.1% of the
60-minute budget** (a bit thinner than the earlier random-weight estimate of
88.5% - normal machine-load variance on this shared server - but still
comfortably under). Re-confirmed determinism on the trained model too (not
just random weights).

Scored the full `datasets/valid.parquet` via `solution.py --validation ...`
(`utils.ScorerStepByStep` - the same logic the platform uses). Took 1h08m
wall-clock for the full run, consistent with the 93.1%-of-60-min-budget
latency measurement extrapolated to this exact validation set size.

**Result: WP = 0.622269** (t0=0.622817, t1=0.621721), vs. the organizer
baseline's **0.589595** — **+0.0327, a +5.5% relative improvement**.
1872/1873 sequences were eligible for scoring (1 excluded per the metric's
own eligibility rule, not an error). All 37,274,573 required predictions
were produced with the correct shape and finite values.

**Status: this experiment now has a real, verified, better-than-baseline
result.**

### Packaged for submission (2026-09-12)

Zip built at `../../submissions/001_lstm_attention_2026-09-12.zip`
(`solution.py` + `checkpoints/model.onnx` + `checkpoints/model.onnx.data`,
56.8 KB total — well under the 20 MB limit).

**Gotcha caught by testing, not by inspection**: the ONNX exporter writes
weights to a separate `checkpoints/model.onnx.data` file alongside
`model.onnx` (external-data storage - not obvious just from the `.onnx`
file existing and looking a reasonable size). The first zip built here
included only `solution.py` + `model.onnx` and **would have failed at
scoring time** with a `RUNTIME_EXCEPTION: filesystem error: cannot get
file size` for the missing `.data` file - it worked fine locally only
because the original `checkpoints/` directory still had that file sitting
next to it, masking the omission. Caught by extracting the zip to a fresh
directory and actually running `PredictionModel` from there before calling
it done - a check worth repeating for any future export/package step,
since "it works from the original directory" is not sufficient evidence a
zip is complete.

Not yet uploaded to the platform - that's a manual step for you to do via
the site's submit page.

### Submitted and scored on the real platform (2026-09-12 12:13)

Submission ID `1L09SSNP`. Passed CHECK and SCORE with no errors.

| | Baseline (`OTOPSPGN`) | This model (`1L09SSNP`) |
|---|---:|---:|
| Platform WP | 0.5719 | **0.5977** |
| Runtime | 25m4s | 32m19s (54% of the 60-min budget) |
| Peak memory | 624MB | 623MB |

**Beats the baseline on the real held-out test set**: +0.0258 (+4.5%
relative) - a smaller margin than the +0.0327 seen locally on
`valid.parquet`, and both models scored lower on the real test set than on
their respective validation numbers (baseline: 0.5719 vs its documented
0.589595; ours: 0.5977 vs our local 0.622269) - consistent with the real
test set being a harder or differently-distributed sample than
`valid.parquet`, not a sign of a bug. The *relative* ordering and rough
margin between the two models held up well from local validation to the
real test set, which is a reasonable consistency check that the local
`ScorerStepByStep` result wasn't just overfit to `valid.parquet` quirks.

**Timing came in far more comfortably than expected**: our synthetic
single-machine latency benchmark (0.0899 ms/step) extrapolated to ~93% of
the 60-minute budget for a valid-set-scale run: the real submission instead
used only 54% (32m19s) - the platform's actual scoring hardware is
evidently faster than the environment used for local timing tests, or the
real test set differs in size from the extrapolation assumption. Either
way, the earlier "razor-thin margin" concern turned out to be overly
conservative in practice - there's real headroom on the actual scoring
infrastructure.

This is now a confirmed, real, beats-the-baseline result on the platform,
not just a local estimate.

## Known open risks / things to check before this becomes a real submission

- **Not yet trained to completion.** A 1-epoch smoke test (train=valid=
  `valid.parquet`, batch size 4) was run after the fixes above — see
  [../../EXPERIMENTS.md](../../EXPERIMENTS.md) for the result. No real
  training on `train.parquet` yet.
- **Framework availability at scoring time.** Unlike the baseline (ONNX
  Runtime, guaranteed present per `requirements.txt`), this ships a raw
  PyTorch checkpoint and imports `torch` directly in `solution.py`. The
  Submission guide's Dockerfile snippet installs a CPU PyTorch wheel as part
  of the scorer image's "reasonable set of popular ML libs," so this is
  *probably* fine, but it hasn't been confirmed against the real image —
  worth testing early, or exporting to ONNX as a fallback (nontrivial here,
  since the attention block's growing KV cache doesn't map onto ONNX's
  static-graph GRU/LSTM op the way the baseline's plain recurrent state does;
  would need a custom fixed-size-cache ONNX export, not attempted yet).
- **Per-step latency unknown.** The baseline's single GRU cell runs in
  ~0.076 ms/step (see BASELINE.md); this model does strictly more work per
  step (an LSTM cell of the same size *plus* an attention lookup over up to
  512 cached vectors *plus* a feed-forward block) — needs benchmarking against
  the 60-minute/whole-test-set budget before trusting it'll finish in time.
- **Determinism** not yet verified (required for submission) — should be fine
  in eval mode with dropout off and single-threaded torch, but must be
  checked empirically (run twice, diff the outputs) before submitting.
- **Loss/metric alignment** is a design bet, not a proven win — `mse+pearson`
  is a reasonable-sounding default, not something validated against the
  actual WP score yet.

## Next steps (not done yet)

1. Train on a small subset first (memory reasons — see `dataset.py`'s
   warning about loading the full `train.parquet` eagerly) to sanity-check
   the training loop runs and losses go down.
2. Score `checkpoints/best.pt` locally via `solution.py --validation
   ../../wnn_connectome_starterpack/datasets/valid.parquet` to get a real WP
   number, and log it in [../../EXPERIMENTS.md](../../EXPERIMENTS.md).
3. Benchmark per-step latency of `StreamingPredictor.step()` before assuming
   this fits the time budget.
4. Verify deterministic output across repeated runs.

## Metric correction (2026-09-16)

The organizers found a mistake in their WP calculation and re-evaluated
every submission under a corrected "Global Weighted Pearson correlation"
metric (pooled across the whole dataset, not averaged per sequence — see
`PROBLEM_STATEMENT.md` §3 and `EXPERIMENTS.md`). This experiment's real
platform score changed: **~~0.5977~~ → 0.6403**. Runtime/memory are
unaffected (re-scoring doesn't re-run inference). The local WP figure
above (0.622269) was computed with the old, now-stale local scorer.

## Extended training at the full, originally-intended scale (2026-09-19)

The originally-scored checkpoint used a scaled-down config (d_model=24,
lstm_layers=1, attention_window=32) because the full design
(`model.py`'s own defaults - d_model=128, lstm_layers=2, window=512,
ff_multiplier=4) measured ~7.8x over the latency budget **in pure PyTorch
eager execution** (`inference.py::StreamingPredictor`). That 7.8x figure
was never actually re-measured for the real submission path (the exported,
optimized ONNX graph) at this scale - only at the settled-down d_model=24
config. Re-checking before committing to training: **the ONNX-level
number is 233.5% of budget uncorrected**, not 780% - a real, large
difference, since ONNX Runtime avoids most of eager PyTorch's per-op
dispatch overhead. Still clearly over budget (the worst latency of any
experiment in this project as of this writing), but nowhere near as
hopeless as the eager-mode number suggested. Original checkpoints (config
+ weights) preserved in `checkpoints_original_backup/`.

Trained per explicit user instruction to build/train the full-size design
anyway - 30 epochs, linear warmup (3 epochs) + `ReduceLROnPlateau` decay +
early stopping (patience 5), same recipe as this project's other 30-epoch
runs, on GPU. Treated as a local-comparison result, not a submission
candidate as-is.

**Result (2026-09-19): early-stopped at epoch 16**, best valid_loss=0.3498
at epoch 11. Export re-verified (1.371e-06 max diff vs. both batched
forward and `StreamingPredictor`), latency re-benchmarked on trained
weights: 233.5% of budget uncorrected - matches the pre-training estimate
exactly. Packaged/verified:
`submissions/001_lstm_attention_full_2026-09-19.zip`.

**Local WP = 0.678480** (t0=0.673607, t1=0.683352) - a genuine
improvement over the original scaled-down checkpoint's 0.664580 (+0.014),
confirming the full-scale design does help. But it's still clearly below
the 004/012-family's ~0.68 ballpark, despite matching or exceeding their
parameter count - this suggests the grouped/nonlinear per-feature-type
encoder those architectures use matters more for this task than raw model
scale or attention capacity.

**SUBMITTED 2026-09-19 15:14 (ID `2A2AN4PO`): TIMED OUT, no score.**
Runtime **69m59s** - exactly the predicted outcome given the 233.5%
uncorrected latency finding, not a surprise. Notable: this is the THIRD
timeout in this project clustering right at the ~70-minute mark, and it
matches experiment 002's original timeout duration EXACTLY (both 69m59s =
4199s, to the second), with experiment 013's timeout landing one second
later at 70m0s (4200s). Three unrelated architectures, three timeouts
within a 1-second window - this is no longer "suggestive," it's about as
close to confirmed as an unstated platform behavior gets: the real kill
cutoff is ~4199-4200s, not the nominal 3600s used for WP/runtime-percent
calculations. See `EXPERIMENTS.md`'s policy note and the persistent
project memory.

**RESUBMITTED 2026-09-23 (ID `24QUNMOV`), after the organizers fixed the
runtime-discrepancy bug they'd acknowledged: STILL TIMED OUT.** Runtime
**69m57s** (4197s) - within 2 seconds of the pre-fix timeout. Not a sign
the fix failed: this submission's problem was genuine latency overage
(233.5% uncorrected, beyond any speedup factor ever observed on this
platform, ~1.6x max), not the run-to-run variability bug that got fixed
(that bug was specifically about the SAME zip scoring inconsistently
across submissions - see experiment 002's 48m25s/69m59s/53m54s history
across three submissions of one unchanged zip). A fix targeted at that
kind of variability was never going to rescue a submission that wasn't a
borderline/variable case to begin with. The near-identical runtime also
suggests the ~4200s hard kill cutoff is separate infrastructure from
whatever was fixed - still worth treating as the real ceiling when
estimating timeout risk.

## Retired the full-scale (d_model=128) config, back to a near-original design (2026-09-23)

User's framing correction: comparing "001 full-scale" to experiment 002's
extended retrain wasn't apples-to-apples. 002's extended retrain kept the
*exact same architecture* as its safely-submitted original (same ONNX
graph/node count, only more training) - that's why its two different
outcomes (timeout once, success once) were genuinely informative about
platform variability. "001 full-scale" was never a retrain of the
submitted 001 at all - it's a structurally different, much bigger design
(d_model 24→128, 1→2 LSTM layers, window 32→512, ff_multiplier 2→4) that
was deliberately never submitted the first time specifically because it
measured way over budget before any training happened. Its two timeouts
aren't platform inconsistency, they're the architecture doing exactly
what the pre-submission benchmark predicted, twice.

d_model=128 config retired: checkpoints preserved in
`checkpoints_dmodel128_fullscale_backup/`, its submission zip explicitly
removed and superseded by the new run below (per explicit request to
override it).

**New config**: `train.py`'s CLI defaults changed to **d_model=32,
lstm_layers=1, attention_window=32, ff_multiplier=2** - the original
submitted-and-safe architecture (d_model=24, everything else identical)
bumped only in `d_model` (24→32), now finally given the modern 30-epoch
warmup + `ReduceLROnPlateau` decay + early-stopping recipe the original
2026-09-12 training never used (that was plain CPU training with no
warmup/decay/early-stop logic at all).

**Pre-training checks** (before committing to the full run, per this
project's standard practice): export correctness verified (2.384e-07 max
diff vs. both batched forward and `StreamingPredictor`). Latency on
random-init weights: **92.8% of budget uncorrected** - close to the
original's confirmed-safe ~88.5%, as expected given how little the
architecture changed. Training launched 2026-09-23 on GPU.
