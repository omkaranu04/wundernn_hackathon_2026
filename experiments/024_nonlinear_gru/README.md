# 024: Experiment 012 with a 2-layer GRU instead of the 2-layer LSTM

Single change from 012 (platform 0.6556, the best so far): the recurrent
core is `nn.GRU(100 -> 128, 2 layers)` instead of `nn.LSTM`. Everything
else is identical: grouped per-feature-type encoders (price 220->24,
volume 220->24, trade 80->12, additional 40 raw), the 5x
`[x, asinh, sin, cos, tanh]` expansion, the 100-dim per-step vector, the
Linear(128,64)-ReLU-Dropout-Linear(64,2) head, and the training recipe
(Adam 1e-3, batch 4, 0.5 weighted MSE + 0.5 per-sequence weighted Pearson,
3-epoch warmup, ReduceLROnPlateau, early stopping on valid_loss, 30
epochs max, seed 0). So the comparison to 012 is apples-to-apples; this
uses 019's `train.py` (also logs corrected global WP each epoch) with
`--select-metric loss` to match 012's selection rule.

## Why

- A GRU has 3 gate blocks instead of 4: **207,358 params vs. 012's
  269,822** (-62,464), and carries a single state tensor `h` instead of
  `(h, c)`.
- The organizer baseline (experiment 000) is itself a 2-layer/128 GRU, so
  ONNX Runtime's `GRU` op is a known-good path.
- 012's weak point is latency (112.2% of budget uncorrected, trained).

## Pre-training checks (2026-10-09, random-init weights)

- Export correctness vs. batched forward: 3.7e-08 max diff.
- ONNX graph: **59 nodes** (012: 62), inputs `[features, h]`, outputs
  `[prediction, next_h]` (`export_onnx.py` and `solution.py` adapted for
  the single-tensor state).
- Latency benchmarked alternately against 012's trained model in the same
  run: **109.7% vs. 122.3% uncorrected, ~10% faster**. Random-init
  figure; to be re-checked on trained weights.

## Status

Trained 2026-10-09 on server A; weights/logs copied back to `results/` and
`logs/` (checksum-verified, deleted there). Target was 012's valid_loss
0.3438 / local WP 0.682697 (platform 0.6556).

## Results

**Worse than 012.** Early-stopped at epoch 12 (best valid_loss **0.3505** at
epoch 7; LR halved at epoch 6 and again at epoch 10). Corrected global WP by
epoch: 0.6669, 0.6733, 0.6759, **0.6780** (3), **0.6780** (4, best WP),
0.6749, 0.6759, 0.6745 (7, the best-loss checkpoint that would ship),
0.6774 (8), 0.6727, 0.6763, 0.6747, 0.6700. Versus 012 (valid_loss 0.3438, WP 0.682697):
**-0.0067 loss and about -0.005 to -0.008 WP** (best-WP epoch 0.6780 vs 0.6827; shipped checkpoint 0.6745), well outside the ~0.002
noise. The lower latency (random-init ~10% below 012) does not buy anything
since the model is clearly less accurate. Verdict: the LSTM's extra gate
(and 62k extra params) matter for this task; a 2-layer GRU of the same width
is a worse backbone, consistent with the fleet-wide finding that backbone
tweaks do not help. Not exported/scored with the real scorer; no zip.

## Submission zip (2026-10-09)

`submissions/024_nonlinear_gru_2026-10-09.zip` (1.5 MB) built from the
shipped best-loss checkpoint (epoch 7) at the user's request. Export vs.
batched forward: 2.0e-06 max diff; 59 nodes; inputs `[features, h]`,
outputs `[prediction, next_h]`; fresh-extraction check passed (shapes,
`None` handling, deterministic). **Latency on trained weights: 103.9% vs.
117.6% for 012 in the same alternating run (~12% faster)**, i.e. a much
safer margin than 012 (absolute numbers are inflated by machine load; 012's
quiet-machine figure is 112.2%). Local WP of this checkpoint is 0.6745
(batched) - 0.008 below 012 - and it has NOT been run through the real
scorer or submitted. Expectation was a platform score a few thousandths below
012's 0.6556, in exchange for the lower latency.

## Platform result (2026-10-09)

**0.6524** (ID 0LFJY04U), runtime **40m17s** (67.1% of budget, 626 MB).
- vs. 012 (0.6556, 45m36s): **-0.0032 score, -12% runtime** - the GRU buys
  ~5 minutes of headroom for ~0.003 of score.
- vs. the 004-architecture band (0.6530-0.6534): -0.0006 to -0.0010, i.e.
  slightly below the plain grouped-encoder models.
- The local gap (-0.008 WP, batched, shipped checkpoint 0.6745 vs 0.6827)
  shrank to -0.0032 on the platform - another case of local differences not
  carrying over one-for-one (local-vs-platform noise ~0.002).
- Implied platform speedup ~1.47x on the estimated ~99% uncorrected latency,
  in the usual 1.45-1.6x range.
Verdict: the LSTM is worth keeping for accuracy; the GRU is a viable
latency-saving fallback, not an improvement.
