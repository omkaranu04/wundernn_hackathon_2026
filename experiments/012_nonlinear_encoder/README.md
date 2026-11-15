# 012 — Nonlinear-expanded grouped-encoder LSTM (all groups, 4 nonlinearities)

**Status: complete and submitted. Local WP = 0.682697 (best of the
fleet). Platform WP = 0.6556 (ID 1UIY3LQX, runtime 45m36s/76.0% of
budget) — new best platform score overall. Did not time out despite the
112.2% uncorrected latency flag.**

## Idea

User's proposal: for each of the 112 raw features, add at least 4
nonlinear transforms (`x`, `log(x)`, `sin(x)`, `e^x`, `cos(x)` or similar)
before feeding into experiment 004's grouped per-feature-type encoder
structure (2-layer LSTM, hidden=128, MLP head, 30 epochs) - explicitly
overriding any latency-budget objection ("only 1 more would do nothing,
just ignore your benchmarking").

## Design investigation (kept for the record, not gating the final choice)

An intermediate version of this experiment scoped the expansion down to
price-only with a single extra transform, after two rounds of grounding
the literal proposal in real data - both checked before committing to a
multi-hour GPU run.

**1. Feature statistics ruled out the literal transform list.**
`analysis/results/feature_statistics.md` shows features are already
normalized to roughly `[-5.5, 5.5]` and are heavily MIXED-SIGN
(`frac_negative`/`frac_positive` both nontrivial for nearly every column) -
not the non-negative price/volume magnitudes a literal `log(x)`/`e^x` pair
would assume. `log(x)` is undefined for roughly half of most columns' rows;
`e^x` is finite at this range but badly scale-mismatched against the other
O(1) channels (`e^5.5 = 245`). Re-checking `excess_kurtosis` **per
feature-type section** (not just globally) showed the fat-tail problem a
log-like transform actually helps with is almost entirely a PRICE
phenomenon: avg excess_kurtosis ~109-225 (max 1064) for price columns vs.
~1-5 for volume/trade and ~0.1 for additional. Volume/trade/additional are
already close to normal-tailed - expanding them was never well-justified in
the first place, independent of any latency concern.

**2. Latency then ruled out expanding even price+volume together.**
Benchmarked every candidate on randomly-initialized weights before
training (same practice as experiments 010/011):

| design | channels | % of budget (uncorrected) |
|---|---|---:|
| all 4 groups x [x, signed-log(sign*log1p), sin, cos, tanh] | 5 | 122.7% |
| all 4 groups x [x, asinh, sin, cos, tanh] (asinh swap) | 5 | 113.0% |
| all 4 groups x [x, asinh, sin, tanh] | 4 | 111.1% |
| all 4 groups x [x, asinh, tanh] | 3 | 106.1% |
| all 4 groups x [x, asinh] | 2 | 101.0% |
| **price only x [x, asinh]** (final) | 2 (price), 1 (rest) | **97.5%** |

Two findings here, both worth keeping for future architecture decisions:
- Swapping `sign(x)*log1p(|x|)` for the single-op `torch.asinh(x)` (same
  shape: odd, monotonic, log-like tail compression, safe for all reals)
  saved ~10 points of budget for free - a decomposed multi-op "cheap"
  function can cost more than a native single-op equivalent, same lesson
  004's own GELU->ReLU swap taught (see 004's model.py revision history).
- Even the single narrowest multi-group version (`[x, asinh]` on all 4
  groups, i.e. just DOUBLING every group's width) still measured 101.0% -
  confirming this architecture family's latency is driven by the grouped
  encoder's Slice/Concat/MatMul node structure itself, not by which or how
  many transform functions are chosen. Scoping to price only (rather than
  further shrinking the transform count) was the only remaining lever that
  both fit the budget and kept a real statistical justification.

## Final architecture (per explicit user override)

The price-only compromise above was superseded: **every group** (price,
volume, trade, additional) gets the full `[x, asinh(x), sin(x), cos(x),
tanh(x)]` expansion (5x its raw width), per explicit instruction to include
at least 4 nonlinearities and disregard the latency finding.

    i0+i1 price   (44*5=220) -> Linear(220,24) -> ReLU               = 24
    i0+i1 volume  (44*5=220) -> Linear(220,24) -> ReLU               = 24
    i0+i1 dp+dv   (16*5=80)  -> Linear(80,12)  -> ReLU               = 12
    additional a0-a7 (8*5=40) -> untouched, passed through as-is     = 40
                                                               total = 100

100-dim LSTM input (vs. 004's 48-dim), same 2-layer/128-hidden LSTM and MLP
head otherwise.

**This measured 122.7% of budget uncorrected on randomly-initialized
weights** (see the table above) - the worst latency margin of any
experiment in this project, clearly over the nominal 60-minute limit even
before accounting for any platform variance. Training and the full
pipeline are proceeding anyway per explicit user instruction. If this
submission times out on the real platform, that would be consistent with
(not contradicting) the local synthetic measurement - unlike 002's
extended-retrain timeout, which was a genuine surprise given a *safe*
synthetic estimate, a timeout here would be the expected outcome, not a
new finding.

## Pre-training checks

- **Correctness**: verified at every design iteration, max abs diff always
  ~1.5-2.2e-08 vs. batched forward (unaffected by the transform-set changes).
- **Latency**: 122.7% of budget uncorrected on randomly-initialized weights
  - over budget, proceeding per explicit user instruction (see above).

## Training

30 epochs (linear warmup for 3, then `ReduceLROnPlateau` decay + early
stopping patience 5) - identical recipe to experiments 004/010/011, on
GPU, running concurrently with experiments 010 and 011 (3-way GPU
contention, same pattern as the earlier 002/004/006 concurrent run).

## Status / next steps

- [x] Nonlinearity set investigated down to price-only `[x, asinh(x)]`
      (97.5% uncorrected), then overridden back to the full
      `[x, asinh, sin, cos, tanh]` x all 4 groups per explicit user
      instruction (122.7% uncorrected)
- [x] Latency benchmarked across 6 candidate designs on random-init
      weights; final (as-trained) design measured 122.7% uncorrected -
      over budget, training proceeding anyway per explicit instruction
- [x] Export correctness verified at every design iteration
- [x] Trained on full `train.parquet` — early-stopped epoch 17, best
      valid_loss=0.3438 at epoch 12
- [x] Re-exported/re-verified/re-benchmarked on trained weights:
      **112.2% of budget uncorrected** — down from 122.7% on random-init
      weights, but still clearly over budget
- [x] Loss curve plotted (`loss_curve.png`)
- [x] Packaged and verified: `submissions/012_nonlinear_encoder_2026-09-18.zip`
- [x] Scored via `solution.py --validation ...`: **local WP = 0.682697**
      (t0=0.678701, t1=0.686692)
- [x] Compared against experiment 004 and the rest of the fleet: **new
      best local WP of the entire fleet**, beating 004's 30-epoch result
      by +0.000272 — a real but very small gain, bought with a
      substantially heavier (100-dim vs. 48-dim LSTM input) and slower
      (112.2% vs. 95.6% uncorrected) architecture
- [x] Submitted to the real platform: **0.6556** (ID 1UIY3LQX), runtime
      45m36s (76.0% of budget) — new best platform score overall. The
      112.2% uncorrected latency flag did not materialize into a timeout;
      real platform speedup was 1.48x vs. the local estimate, in line
      with most of this project's submissions (excluding 002's anomaly)
