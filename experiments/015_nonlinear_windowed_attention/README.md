# 015 — Nonlinear grouped-encoder + 1-layer LSTM + windowed attention (512)

**Status: complete and submitted. Local WP = 0.678887 — below the
012/017 ballpark. Platform: TIMED OUT (ID 0JIX02Z9, runtime 69m59s) —
exactly as predicted given 235.5% uncorrected.**

## Idea

User's request: take experiment 012's exact input encoding (all 4 feature
groups expanded 5x via `[x, asinh, sin, cos, tanh]`) and output structure
(2-layer MLP head predicting `(t0, t1)` directly), but replace 012's plain
2-layer LSTM backbone with a **1-layer** LSTM followed by experiment 001's
attention block (LayerNorm → causal windowed self-attention, window=512 →
residual → LayerNorm → feed-forward → residual) - the same "LSTM +
attention on top" pattern 001 introduced, transplanted onto 012's richer
input encoding.

## Implementation

`CausalSelfAttention` (windowed/banded, chunked implementation) is copied
verbatim from experiment 001's `model.py` - see that file for the full
story on why it's windowed rather than O(T²), and the OOM history behind
the chunked (not `unfold()`-based) implementation. The streaming ONNX
export reuses 001's proven pattern too: `nn.LSTMCell` instead of native
`nn.LSTM`, `index_copy` for the circular key/value buffer write, explicit
`matmul` instead of `einsum`, and a `valid_mask` + large-negative-bias
trick in place of runtime-length slicing.

## Pre-training checks

- **Correctness**: verified, max abs diff 1.192e-07 vs. batched forward.
- **Latency**: **236.3% of budget uncorrected** on randomly-initialized
  weights - consistent with experiment 001's full-scale finding (233.5%
  at d_model=128 with the same window=512 attention buffer); attention at
  this hidden size is inherently costly regardless of which encoder feeds
  it. Not re-litigated with the user given the consistent, already-accepted
  risk profile across this batch of experiments (d_model=128 scale chosen
  explicitly, latency accepted).

## GPU scheduling note

Attempted to launch alongside experiments 017 and 001's full-scale
extended run; hit a real CUDA OOM (GPU fully saturated: 017 using 2.72GB,
001-full using 5.28GB, ~8MiB free). Initially queued behind a GPU-memory
poll, per explicit user correction this was replaced with
`../queue_scheduler.sh`: a strict two-slot scheduler gated purely on
**process completion**, not memory. It tracks the two originally-running
jobs (017, 001-full); whenever either exits, it launches the next queued
experiment into that freed slot and writes the new training PID to
`train.pid`, which this experiment's `auto_pipeline.sh` waits for and then
polls (the export/zip/scoring steps are CPU-only and can run alongside
other GPU training, so only the *training* PID gates this pipeline).

## Training

30 epochs (linear warmup for 3, then `ReduceLROnPlateau` decay + early
stopping patience 5) - identical recipe to every 30-epoch experiment here.

## Status / next steps

- [x] Architecture: 012's encoder + 1-layer LSTM + 001's windowed attention
- [x] Export correctness verified
- [x] Latency benchmarked (236.3% uncorrected)
- [x] Trained on full `train.parquet` — early-stopped epoch 13, best
      valid_loss=0.3494
- [x] Re-exported/re-verified/re-benchmarked on trained weights: 235.5%
      of budget uncorrected — matches the pre-training estimate
- [x] Loss curve plotted (`loss_curve.png`)
- [x] Packaged and verified: `submissions/015_nonlinear_windowed_attention_2026-09-19.zip`
- [x] Scored via `solution.py --validation ...`: **local WP = 0.678887**
      (t0=0.673726, t1=0.684048)
- [x] Compared against experiments 001, 012, and the rest of the fleet:
      below the 012/017 ballpark (~0.682-0.683), in the same
      underperforming range as 001-full (0.678480) — windowed attention
      on top of 012's encoder doesn't beat the plain head locally
- [ ] Submitted to the real platform — not yet (235.5% uncorrected is a
      real timeout risk)
