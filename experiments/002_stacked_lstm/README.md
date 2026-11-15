# 002 — Stacked LSTM (deeper/wider than baseline)

**Status: training launched, not yet scored.**

## Idea

The baseline is a 2-layer, 128-dim stacked GRU. This experiment asks: does
more depth/width, and LSTM cells (4 gates) instead of GRU cells (3 gates),
help? No attention this time — deliberately simpler than experiment 001,
to isolate the effect of "more recurrent capacity" on its own.

## Architecture

```
features (T, 112) -> multi-layer nn.LSTM(hidden_size, num_layers) -> Linear(hidden_size, 2)
```

No input projection, no attention, no windowed cache — mirrors the
baseline's own minimal structure. See `model.py`.

Unlike experiment 001, a plain recurrent stack's hidden state **is** the
complete, fixed-size summary of everything seen so far, so training
(whole-sequence `nn.LSTM` call) and streaming inference (the same module
called one step at a time) are naturally numerically identical — no
reformulation needed the way attention's growing cache needed one in 001.
`dataset.py` still uses one full sequence per training sample for
simplicity/consistency with 001, though truncated-BPTT on shorter chunks
would be numerically valid here (a possible future optimization, not
implemented).

## Two findings from the latency exploration (before committing to training)

### 1. Forcing the legacy ONNX exporter is a correctness trap here, not just verbose

The baseline's `baseline.onnx` was built with **torch 2.5.1** (checked via
`onnx.load(...).producer_version`), before PyTorch's dynamo-based ONNX
exporter became default (torch 2.9+) — that's why its graph is so lean
(10 nodes: `GRU`x2, `Squeeze`x2, `Transpose`x2, `MatMul`, `Add`, 2
`Constant`s). Our environment has torch 2.14.0, where `torch.onnx.export`
now defaults to the newer dynamo/`torch.export`-based path, which introduces
much more glue (`Reshape`/`Slice`/`Concat`/`Transpose`) around `nn.LSTM`'s
multi-layer state handling — 42 nodes for an equivalent 2-layer/128-hidden
model.

The tempting fix — pass `dynamo=False` to force the old TorchScript-based
exporter — **silently produces wrong output**: max abs diff of **0.254**
against the correct PyTorch reference, confirmed present even before any
further processing. This is not a verbosity difference, it's a broken
export for this torch version's legacy path on a stateful multi-layer LSTM
with explicit initial states. **Do not use `dynamo=False` here.**

The safe fix: export normally (dynamo, correct), then run it through
**`onnx-simplifier`** (`onnxsim`) as a post-processing pass — this cuts
42 → 18 nodes and was re-verified numerically identical (~4.5e-8 diff) to
the unsimplified version. `export_onnx.py`'s `export()` does this
automatically (`simplify_graph=True` by default).

Note: the node-count reduction from `onnxsim` did **not** meaningfully
change measured latency (0.0745ms vs 0.0753ms/step at 128×2, essentially
noise) — the removed nodes were apparently near-zero-cost (constant-folding
candidates), not the actual bottleneck. Still worth doing (smaller model
graph, no correctness cost), just don't expect it to be a speed fix on its
own.

### 2. Our synthetic latency benchmarks run pessimistic vs. the real platform

Cross-checking against actual submitted results (see experiment 001's
README and `EXPERIMENTS.md`):

| | Synthetic benchmark (this dev server) | Real platform submission |
|---|---:|---:|
| Baseline | 0.076 ms/step (~79% of budget) | 25m4s (~42% of budget) |
| Experiment 001 | 0.0899 ms/step (~93% of budget) | 32m19s (~54% of budget) |

Both show the real platform running **~1.7-1.9x faster** than our synthetic
single-machine benchmark predicts. This is only two data points, not a
proven constant, but it's consistent enough to inform sizing decisions - see
the config sweep below, presented both raw and with this ~1.8x correction
applied.

## Config sweep (random weights - latency only, not accuracy)

All exported via the dynamo+onnxsim pipeline, benchmarked single-threaded
(matching the competition's 1-vCPU constraint):

| hidden_size | layers | ms/step (synthetic) | % of budget (synthetic) | % of budget (÷1.8, corrected estimate) |
|---:|---:|---:|---:|---:|
| 128 | 2 | 0.0753 | 78.0% | ~43% |
| 128 | 3 | 0.1053 | 109.0% | ~61% |
| **128** | **4** | **0.1321** | **136.8%** | **~76%** |
| 192 | 2 | 0.1077 | 111.5% | ~62% |
| 192 | 3 | 0.1510 | 156.3% | ~87% (thin) |
| 256 | 2 | 0.1460 | 151.2% | ~84% (thin) |
| 256 | 3 | 0.2143 | 221.9% | ~123% (still over) |
| 384 | 2 | 0.2538 | 262.8% | ~146% (over) |
| 384 | 3 | 0.4016 | 415.8% | over |
| 512 | 2/3 | 0.42-1.10 | 440-1144% | over |

**Settled on hidden_size=128, num_layers=4** — twice the baseline's depth
at the same width, with a corrected estimate of ~76% of budget. This is a
genuine bet on the ~1.8x correction factor holding; if it doesn't hold at
this scale, the real submission could come in slower than hoped (though
still likely to finish, just with less margin than estimated). Given only
5 submissions/day are allowed, this was a deliberate choice to test — see
the conversation this experiment came from for the explicit tradeoff
discussion before committing.

## Files

Same structure as experiment 001, minus the attention-specific pieces:
`model.py`, `dataset.py`, `train.py`, `export_onnx.py`, `solution.py`. No
`inference.py` needed — there's no growing cache requiring a separate
streaming reformulation the way attention needed one.

## Status / next steps

- [x] Architecture designed, latency-explored, config settled
- [x] Export pipeline (dynamo + onnxsim) verified correct
- [x] `solution.py` smoke-tested: correct shapes, deterministic
- [x] Trained on full `train.parquet` — 5 epochs, ~13h24m, best
      `valid_loss=0.3574`
- [x] Exported to ONNX, re-verified correct on trained weights (diff ~1e-6),
      latency re-confirmed (~0.124ms/step, ~71-76% of budget corrected)
      and determinism re-confirmed
- [x] Packaged: `../../submissions/002_stacked_lstm_2026-09-13.zip`
      (`solution.py` + `checkpoints/model.onnx` + `checkpoints/model.onnx.data`,
      ~3.9MB) — tested from a fresh extraction directory before trusting it,
      same lesson as experiment 001 (the `.data` file is easy to miss)
- [x] Scored via `solution.py --validation ...` for a real WP number:
      **0.634285** (t0=0.633174, t1=0.635397, 1872/1873 eligible blocks,
      all 37,274,573 required predictions produced correctly)
- [x] Compared: beats experiment 001 (0.622269 local, +0.0120/+1.9%) and the
      baseline (0.589595 local, +0.0447/+7.6%) — **new best local result**
- [ ] Not yet uploaded to the platform

## Result (2026-09-13): new best local WP score

| | Baseline | Experiment 001 | **Experiment 002** |
|---|---:|---:|---:|
| Local WP (valid.parquet) | 0.589595 | 0.622269 | **0.634285** |

The plain, deeper stacked LSTM (no attention, hidden_size=128, num_layers=4)
edges out experiment 001's attention-based approach on local validation,
despite being architecturally much simpler (no windowed cache, no
reformulation needed for streaming inference — see `model.py`). Whether
this ordering holds on the real platform test set (as it did for experiment
001 vs. baseline) isn't confirmed yet - that requires an actual submission.

Full scoring run took 1h31m locally (this dev machine, not the platform) -
notably longer than the ~77min extrapolated from the per-step latency
benchmark, consistent with `ScorerStepByStep`'s validation/bookkeeping
overhead adding real cost beyond raw ONNX `session.run()` time, plus normal
shared-machine load variance.

### Submitted and scored on the real platform (2026-09-13 11:23)

Submission ID `1C8UCKOX`. Passed CHECK and SCORE with no errors.

| | Baseline (`OTOPSPGN`) | Exp 001 (`1L09SSNP`) | This model (`1C8UCKOX`) |
|---|---:|---:|---:|
| Platform WP | 0.5719 | 0.5977 | **0.6106** |
| Runtime | 25m4s (42% of budget) | 32m19s (54% of budget) | **48m25s (80.7% of budget)** |
| Peak memory | 624MB | 623MB | 632MB |

**New best result on both local validation and the real platform** — beats
baseline by +0.0387 (+6.8% relative) and experiment 001 by +0.0129 (+2.2%
relative).

**Timing came in much tighter than the other two submissions, and tighter
than hoped.** Working backward from runtime: local synthetic benchmark was
~77 min (0.124 ms/step raw), real platform took 48m25s - a ~1.59x speedup,
notably smaller than the ~1.8x factor seen for the baseline and experiment
001. Likely explanation: those were much smaller/cheaper-per-step models,
so proportionally more of their local-benchmark time was fixed per-call
overhead that the platform's environment happens to shave off efficiently;
this model does more genuine FLOPs per step (4 LSTM layers vs. 1-2), so
less of its time is that same shaveable overhead. **Practical implication
for any future scale-up of this architecture**: don't assume the ~1.8x
correction factor holds - it appears to shrink as the model gets heavier,
so margin could disappear faster than a naive linear extrapolation would
suggest. This run had real but thin margin (~11.5 minutes of slack); a
similarly-sized-but-heavier next step could plausibly time out.

## Metric correction (2026-09-16)

The organizers found a mistake in their WP calculation and re-evaluated
every submission under a corrected "Global Weighted Pearson correlation"
metric (pooled across the whole dataset, not averaged per sequence — see
`PROBLEM_STATEMENT.md` §3 and `EXPERIMENTS.md`). This experiment's real
platform score changed: **~~0.6106~~ → 0.6507** — and under the corrected
metric, this is now **essentially tied for the best score of any
submission** (with experiment 004 at 0.6508), unlike under the old metric
where it was edged out by experiment 005. Runtime/memory are unaffected.
The local WP figure above (0.634285) was computed with the old, now-stale
local scorer.

## Extended re-training (2026-09-16/17) — improved, but same latency risk as before

`train.py` in this directory was updated in place (epochs 5 -> 15,
`ReduceLROnPlateau` LR decay + early stopping, GPU), run concurrently
with experiments 004 and 006 on the one available GPU. Original 5-epoch
checkpoints preserved in `checkpoints_5epoch_backup/`. See
`EXPERIMENTS.md`'s "extended re-training" section for the full rationale,
setup, and how this compares to 004/006's results.

Ran the full 15 epochs — early stopping (patience 5) never formally
triggered, but came close: best epoch was 10 (valid_loss=0.3444), and
valid_loss climbed for the remaining 4 epochs despite two LR decays
(1e-3→5e-4 at epoch 9, →2.5e-4 at epoch 13), ending at 0.3566 by epoch
14 — a mild overfitting pattern. The saved `best.pt` correctly holds
epoch 10's weights, not the final epoch's. Total training time 13h35m20s
(slower than a solo GPU run due to 3-way GPU contention).

| | Original (5 epochs) | Extended (15 epochs, LR decay) |
|---|---:|---:|
| Local WP | 0.677570 | **0.680390** (+0.0028) |
| best valid_loss | 0.3574 | 0.3444 |

Re-exported and re-verified correct (max abs diff 1.192e-06),
re-benchmarked latency: **127.7% of budget uncorrected** — essentially
unchanged from this architecture's known risk profile (the original
5-epoch checkpoint measured similarly high and only worked because of the
platform's observed speedup factor, landing at 48m25s/80.7% actual on
submission). This is expected: retraining changes the weights, not the
architecture, so the ONNX graph's node count and structure — what
actually drives this number — is identical to before. Packaged and
verified: `../../submissions/002_stacked_lstm_extended_2026-09-16.zip`.

**Submitted 2026-09-17 12:58 (ID `0BGLDK10`): TIMED OUT, no score.**
Runtime **69m59s** — about 10 minutes over the 60-minute limit, well
past the ~1.5-2min grace period observed elsewhere. This is a genuinely
important finding, not just bad luck: **the ONNX graph submitted here is
byte-identical in structure to the original 002 submission** (only the
trained weights differ; same architecture, same node count) — and that
original submission ran safely at 48m25s with a 1.59x platform speedup
factor. This resubmission only got a ~1.095x factor and timed out. Since
the model didn't change, the speedup difference has to be platform-side
(load, scoring-infra conditions, or similar), not anything about this
architecture specifically. **Conclusion: a platform speedup factor is not
a fixed, reusable property of a model — it can change between submissions
of the exact same graph.** See `EXPERIMENTS.md`'s hardened policy note.
**Do not resubmit this exact zip** — if these weights are worth having on
the platform, they'd need a genuinely <100% (ideally <90-95%) uncorrected
architecture first, not a retry of the same risky config.

**UPDATE 2026-09-18 — resubmitted anyway (same zip), and this time it
worked.** Submission ID `0C0GWEUO`: runtime **53m54s** (89.8% of budget),
632MB, **scored 0.6522**. Implied platform speedup factor this time:
~1.42x (127.7% uncorrected → predicted 76.6min at 1.0x → actual 53.9min →
factor ≈ 1.42x) — between the original submission's 1.59x and the
timed-out resubmission's ~1.095x. Three submissions of architecturally
identical (or near-identical, for the 5ep→15ep weight change) graphs now
give three different speedup factors: 1.59x (safe), 1.095x (timeout),
1.42x (safe). This is a direct, concrete confirmation of the
already-hardened finding — the same ONNX graph got a real 45%
runtime difference (69m59s vs. 53m54s) across two submissions with
nothing but time between them. Consistent with (though not proof of) the
user's hypothesis that platform runtime depends on scoring-infrastructure
load at submission time, not solely on the architecture. **This does not
retroactively make 127.7% uncorrected a safe design target** — it means
this specific submission got lucky twice out of... well, once out of two
tries, a 50% real-world failure rate on a coin flip we have no way to call
in advance. The policy stands: don't design toward this margin on
purpose.
