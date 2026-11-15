# 010 — Sum-reparameterized head: predict (t0, t0+t1)

**Status: complete and submitted. Local WP = 0.681145. Platform WP =
0.6546 (ID 1IC0616L, runtime 40m41s/67.8% of budget) — 2nd-best platform
score overall, beating both 004 variants despite a lower local WP.**

## Idea

Experiment 004's exact architecture (grouped feature-type encoders,
2-layer/128-hidden LSTM, 2-layer MLP head — our best-performing model),
except the head predicts `(t0, t0+t1)` instead of `(t0, t1)` directly,
exploiting the targets' strong negative correlation (-0.7223).

**Why the sum, not the difference** (the more common intuition): for two
variables with correlation `rho`,

```
Var(X - Y) = Var(X) + Var(Y) - 2*Cov(X,Y)
Var(X + Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
```

Since `corr(t0,t1)` is *negative*, `Cov(t0,t1)` is negative — so the
**sum** has *lower* variance (the anti-correlated wiggles partially
cancel) and the **difference** has *higher* variance (they reinforce).
Using the real numbers (`std(t0)=1.006`, `std(t1)=1.011`,
`corr=-0.7223`): `std(t0+t1) ≈ 0.75` vs. `std(t0-t1) ≈ 1.87` vs. the
individual targets' own `std ≈ 1.01`. The sum is the genuinely
lower-variance, "easier" second target — the difference would have made
things worse, not better.

## Implementation

The reconstruction (`t1_hat = s_hat - t0_hat`) happens inside
`model.forward()` and the exported ONNX graph — `shared/losses.py`'s loss
functions compute loss directly in raw `[t0, t1]` units, completely
unchanged, same pattern used for the `sinh` output activation in
experiments 006/007/009.

## Pre-training checks

- **Correctness**: verified, max abs diff 2.98e-08 vs. batched forward.
- **Latency**: **101.4% of budget uncorrected** (52 ONNX nodes — 1 more
  than 004's own graph, for the added `Sub` reconstruction op). A real
  but modest overage — not a "safe regardless of factor" config, but far
  milder than experiment 007's 137% or 005's 120%. Flagged explicitly
  given the recently-hardened finding that even a previously-safe
  architecture (002's extended retrain) can fail on resubmission due to
  platform-side variability — this is not assumed safe just because the
  overage is small.

## Training

30 epochs (linear warmup for 3, then `ReduceLROnPlateau` decay + early
stopping patience 5) — identical recipe to experiment 004's own current
30-epoch run, on GPU, running concurrently with it and with experiment
011.

## Status / next steps

- [x] Architecture designed: 004's backbone + sum-reparam head
- [x] Export correctness verified
- [x] Latency benchmarked and flagged (101.4% uncorrected — real but
      modest risk)
- [x] Trained on full `train.parquet` — early-stopped epoch 16, best
      valid_loss=0.3461 at epoch 11
- [x] Re-exported/re-verified (7.153e-07 max diff)/re-benchmarked on
      trained weights — 100.5% of budget uncorrected, still barely over
- [x] Loss curve plotted (`loss_curve.png`)
- [x] Packaged and verified: `submissions/010_sum_reparam_head_2026-09-17.zip`
- [x] Scored via `solution.py --validation ...`: **local WP = 0.681145**
      (t0=0.676348, t1=0.685942)
- [x] Compared against experiment 004 and the rest of the fleet: 3rd-best
      local WP of the fleet (behind 012 and 004's 30-epoch run), −0.0013
      vs. 004's plain head — the sum-reparameterization did not
      measurably help or hurt locally
- [x] Submitted to the real platform: **0.6546** (ID 1IC0616L), runtime
      40m41s (67.8% of budget) — no timeout despite the 100.5% uncorrected
      flag; 2nd-best platform score overall, beating both 004 variants
      even though 004 had the higher local WP
