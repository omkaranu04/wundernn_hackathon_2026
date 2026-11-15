# 016 — Nonlinear grouped-encoder + 2-layer LSTM + dilated sparse attention

**Status: complete and submitted. Local WP = 0.682750 — 2nd-best of the
entire fleet. Platform: TIMED OUT (ID 2000TFPF, runtime 69m59s) — exactly
as predicted given 387.3% uncorrected (worst latency in the project). The
best-performing model of this batch is also the one guaranteed not to fit
the budget.**

## Idea

Same as experiment 015 (012's input encoding and output structure
unchanged), but with **2 LSTM layers** (not 1) and **sparse** attention
instead of a fixed window: each query attends to a sparse set of past
positions at geometric strides `(1, 2, 4, 8, 16, ..., 2048)` plus itself
(13 fixed positions total), instead of every position within a contiguous
512-wide window. This is the "LogSparse"/dilated attention pattern from
long-sequence time-series transformer work - O(log T) attended positions
per query instead of O(window), giving access to history up to 2048 steps
back for a fraction of the attended-position count.

## Implementation and validation

Validated in isolation before building the full model (see
`EXPERIMENTS.md`): a batched (shifted-K/V-gather) formulation for training
against an exact streaming circular-buffer recursion for export. Unlike
experiment 013's EWMA, this needs **no train/export approximation gap** -
gathering exact values at fixed offsets is trivially exact in both forms.
One real bug caught during validation: a circular buffer sized exactly
`max(offsets)=2048` has the current step's write clobber the exact slot
the max-offset lookup needs at that same step (both map to `t % 2048`).
Buffer size must be `max(offsets)+1 = 2049` - fixed and re-validated
(4.77e-07 max diff on a 3000-step synthetic test after the fix).

Export follows the same proven techniques as experiments 001/015: LSTMCell
instead of native LSTM, `index_copy` for the circular buffer write. The
sparse gather itself uses 13 `index_select` calls (one per offset, each
computing its target slot as `(write_pos - offset) % 2049`) instead of a
single contiguous slice.

## Pre-training checks

- **Correctness**: verified over 3000 streaming steps, max abs diff
  1.527e-07 vs. batched forward - confirms both the sparse-gather logic
  and the buffer-size fix work correctly end-to-end.
- **Latency**: **396.5% of budget uncorrected** on randomly-initialized
  weights - worse than 015's windowed attention (236.3%), despite
  attending to far fewer positions (13 vs up to 512). This is a genuinely
  counterintuitive, worth-keeping finding: it directly confirms this
  project's repeated lesson (004/012/013's revision histories) that ONNX
  Runtime latency is driven by **node/op count**, not FLOPs. Windowed
  attention does one bulk contiguous slice + one big matmul; sparse
  attention needs 13 separate `index_select` gathers (×2, for K and V)
  plus the glue to stack/concat them before a much smaller matmul - more
  nodes, even though it touches far less data. "Sparse" is not a free
  latency win on this runtime. Not re-litigated with the user given the
  already-established risk profile across this batch of experiments.

## GPU scheduling note

GPU was fully saturated by 017 and 001's full-scale run at build time.
Queued behind `../queue_scheduler.sh` - a strict two-slot scheduler gated
purely on **process completion**, not GPU memory polling (which was tried
first and explicitly corrected by the user). It tracks the two originally-
running jobs; whenever either exits, it launches the next queued
experiment into that freed slot and writes the new PID to `train.pid`,
which `auto_pipeline.sh` waits for before polling it - the export/zip/
scoring steps are CPU-only and run alongside other GPU training
regardless.

## Training

30 epochs (linear warmup for 3, then `ReduceLROnPlateau` decay + early
stopping patience 5) - identical recipe to every 30-epoch experiment here.

## Status / next steps

- [x] Sparse attention validated numerically (batched vs. exact streaming,
      including catching and fixing the buffer-size off-by-one)
- [x] Architecture: 012's encoder + 2-layer LSTM + dilated sparse attention
- [x] Export correctness verified (1.527e-07 max diff, 3000 steps)
- [x] Latency benchmarked (396.5% uncorrected - worse than windowed
      attention despite fewer attended positions; see note above)
- [x] Trained on full `train.parquet` — early-stopped epoch 20, best
      valid_loss=**0.3431**, a new low across this project's entire
      30-epoch-recipe fleet
- [x] Re-exported/re-verified/re-benchmarked on trained weights: 387.3%
      of budget uncorrected — matches the pre-training estimate (396.5%)
- [x] Loss curve plotted (`loss_curve.png`)
- [x] Packaged and verified: `submissions/016_nonlinear_sparse_attention_2026-09-19.zip`
- [x] Scored via `solution.py --validation ...`: **local WP = 0.682750**
      (t0=0.677861, t1=0.687638) — took ~4h15m given the heavier per-step
      graph (~8s/seq vs. the fleet's usual 2-4s)
- [x] Compared against experiments 012, 015, and the rest of the fleet:
      **2nd-best local WP of the entire fleet** (just behind 017's
      0.682929), matching plain 012 (0.682697) almost exactly. Cross-
      experiment comparison (see EXPERIMENTS.md) shows this is because
      012's grouped encoder + 2 LSTM layers together drive the result -
      015 (same encoder, only 1 layer) drops ~0.004 to the raw-feature
      experiments' level; attention type (sparse here) barely matters
- [ ] Submitted to the real platform — not yet (387.3% uncorrected, worst
      latency in the project, makes a timeout near-certain)
