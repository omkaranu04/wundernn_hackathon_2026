# 017 — Nonlinear-expanded grouped-encoder LSTM + sum-reparam head

**Status: complete and submitted. Local WP = 0.682929 (best of the entire
fleet). Platform WP = 0.6553 (ID 1CYUNOV2, runtime 47m42s/79.5% of
budget) — 2nd-best platform score overall (just behind 012's 0.6556),
despite having the best local WP — a local/platform ranking mismatch.**

## Idea

User's request: take experiment 012's exact architecture (all 4 feature
groups expanded 5x via `[x, asinh, sin, cos, tanh]`, 2-layer/128-hidden
LSTM, MLP head) and swap in experiment 010's sum-reparameterization -
predict `(t0, t0+t1)` instead of `(t0, t1)` directly, exploiting the
targets' negative correlation. Everything else stays exactly as 012 built
it.

## Implementation

Byte-for-byte identical to 012 except the last step of `forward()`: the
head's raw 2-dim output `(z0, s)` is reinterpreted as `(t0_hat, s_hat)`
where `s_hat` targets `t0+t1`, then `t1_hat = s_hat - t0_hat` is
reconstructed before returning - same pattern as experiment 010's
`model.py`, just applied on top of 012's richer input encoding instead of
004's plain one. `shared/losses.py` needs no changes since the
reconstruction happens inside the model, not the loss.

## Pre-training checks

- **Correctness**: verified, max abs diff 2.98e-08 vs. batched forward.
- **Latency**: 121.9% of budget uncorrected on randomly-initialized
  weights - matches 012's own 122.7% almost exactly (one extra `Sub` op
  for the reconstruction, same risk profile). Not re-litigated with the
  user since 012's identical latency profile was already explicitly
  accepted ("ignore benchmarking").

## Training

30 epochs (linear warmup for 3, then `ReduceLROnPlateau` decay + early
stopping patience 5) - identical recipe to every 30-epoch experiment here.

## Status / next steps

- [x] Architecture: 012's encoder + 010's sum-reparam head
- [x] Export correctness verified
- [x] Latency benchmarked (121.9% uncorrected, matches 012)
- [x] Trained on full `train.parquet` — early-stopped epoch 16, best
      valid_loss=0.3420 (vs. 012's own 0.3438)
- [x] Re-exported/re-verified (5.960e-07 max diff)/re-benchmarked on
      trained weights — 123.6% of budget uncorrected, matches 012
- [x] Loss curve plotted (`loss_curve.png`)
- [x] Packaged and verified: `submissions/017_nonlinear_sum_reparam_2026-09-19.zip`
- [x] Scored via `solution.py --validation ...`: **local WP = 0.682929**
      (t0=0.678749, t1=0.687108)
- [x] Compared against experiments 010, 012, and the rest of the fleet:
      **new best local WP of the entire fleet**, +0.000232 over 012.
      Notably the sum-reparam trick helped here but *hurt* on 004's
      architecture (010 vs 004: −0.0013) — effect isn't consistent across
      backbones, though both deltas are small enough to be within noise.
- [x] Submitted to the real platform: **0.6553** (ID 1CYUNOV2), runtime
      47m42s (79.5% of budget) — 2nd-best platform score overall (just
      behind 012's 0.6556), despite having the best local WP of the
      fleet — another local/platform ranking mismatch, like 010's earlier
      one
