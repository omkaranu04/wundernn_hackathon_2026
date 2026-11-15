# 014 — LSTM + dilated sparse attention (001's full-scale design)

**Status: complete and submitted. Local WP = 0.678463 — clusters with
001-full and 015 in the lower ~0.678-0.679 band. Platform: TIMED OUT (ID
04UAPZDI, runtime 70m0s) — exactly as predicted given 356.4% uncorrected.**

## Idea

User's request (item 1 of a 5-experiment batch): "like experiment 1
[001], apply attention, but instead of the full attention window of 512,
try sparse attention for (1,2,4,8,...,2048), and increase the LSTM layers
to 2 (as against experiment 1). Rest all stays the same in the
architecture."

Two clarifications resolved before implementation (see EXPERIMENTS.md):
1. "Experiment 2" in the original phrasing meant experiment **001** - the
   "window of 512" description only matches 001's mechanism.
2. "Rest stays the same" - and "increase lstm_layers to 2" - only makes
   sense against `model.py`'s own class defaults (d_model=128,
   lstm_layers=2 already, n_heads=4, ff_multiplier=4), NOT the scaled-down
   config (d_model=24, lstm_layers=1) 001 was actually trained/scored
   with. That scaled-down config existed only because the full design
   measured ~7.8x over the latency budget in PyTorch eager mode - never
   actually trained. Per explicit user instruction (after being shown this
   finding), this experiment builds/trains the full design anyway.

## Architecture

Same overall block shape as 001's `LSTMAttentionModel` (Linear input
projection → LSTM → LayerNorm → attention → residual → LayerNorm →
feed-forward → residual → head), at d_model=128, lstm_layers=2,
ff_multiplier=4 (001's own class defaults), with the attention mechanism
swapped from a fixed window=512 to `DilatedSparseSelfAttention` - see
experiment 016's model.py/README for the full design and numerical
validation of that mechanism (offsets `1,2,4,...,2048`, buffer size
`max(offsets)+1=2049`, the off-by-one that validation caught and fixed).
No grouped/nonlinear input encoder here - raw 112 features go straight
into the input projection, matching 001's original design exactly.

## Pre-training checks

- **Correctness**: verified over 3000 streaming steps, max abs diff
  3.576e-07 vs. batched forward.
- **Latency**: **359.6% of budget uncorrected** on randomly-initialized
  weights. Two things worth noting:
  - Much better than the "~7.8x" (780%) figure historically cited for
    this scale - that number was PyTorch-eager-only; ONNX Runtime cuts
    it roughly in half again from what 001's own full-scale windowed
    variant measured (233.5%, see 001's README) once attention is swapped
    to sparse.
  - Still worse than budget by a wide margin, and - consistent with
    experiment 016's finding - sparse attention (13 positions) is not
    cheaper here than a dense window despite touching far fewer values;
    node/op-count overhead (13 separate gathers vs. one bulk slice)
    dominates over FLOP savings on this runtime.
  Not re-litigated with the user given the already-established risk
  profile across this batch of experiments (d_model=128 scale explicitly
  chosen, latency explicitly accepted).

## GPU scheduling note

GPU was fully saturated by experiments 017 and 001's full-scale run at
build time. Queued behind `../queue_scheduler.sh` - a strict two-slot
scheduler gated purely on **process completion**, not GPU memory polling
(which was tried first and explicitly corrected by the user: schedule
strictly on whether a prior job's training has finished, never on
measured free memory). It tracks the two originally-running jobs; whenever
either exits, it launches the next queued experiment into that freed slot
and writes the new PID to `train.pid`, which `auto_pipeline.sh` waits for
before polling it - the export/zip/scoring steps are CPU-only and run
alongside other GPU training regardless.

## Training

30 epochs (linear warmup for 3, then `ReduceLROnPlateau` decay + early
stopping patience 5) - identical recipe to every 30-epoch experiment here.

## Status / next steps

- [x] Clarified which experiment "experiment 2" referred to, and which
      scale ("rest stays the same") to build from
- [x] Architecture: 001's full-scale design + dilated sparse attention +
      2 LSTM layers
- [x] Export correctness verified (3.576e-07 max diff, 3000 steps)
- [x] Latency benchmarked (359.6% uncorrected)
- [x] Trained on full `train.parquet` — ran solo on GPU once the queue
      emptied (~2.2 batch/s), early-stopped epoch 18, best valid_loss=0.3489
- [x] Re-exported/re-verified/re-benchmarked on trained weights: 356.4%
      of budget uncorrected — matches the pre-training estimate (359.6%)
- [x] Loss curve plotted (`loss_curve.png`)
- [x] Packaged and verified: `submissions/014_sparse_attention_lstm_2026-09-19.zip`
- [x] Scored via `solution.py --validation ...`: **local WP = 0.678463**
      (t0=0.673638, t1=0.683289)
- [x] Compared against experiments 001 (both scales), 016, and the rest of
      the fleet: clusters with 001-full (0.678480) and 015 (0.678887) in a
      lower ~0.678-0.679 band, well below 012/016/017's ~0.6827-0.6829.
      Cross-experiment comparison shows attention (windowed vs. sparse vs.
      none) barely matters - what separates the bands is 012's grouped
      encoder combined with 2 LSTM layers together (016 matches 012/017);
      the same encoder with only 1 layer (015) drops to this lower band.
- [ ] Submitted to the real platform — not yet (356.4% uncorrected makes
      a timeout likely)
