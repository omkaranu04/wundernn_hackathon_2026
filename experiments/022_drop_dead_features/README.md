# 022: Retrain 004's architecture without the features experiment 021 found unused

Motivation: permutation importance (experiment 021) showed instrument 1's
price and volume (44 of 112 features) are essentially unused by 004/012
(group drops 0.001-0.003) and `a1`, `a3` are slightly harmful. This
retrains 004's grouped-encoder LSTM without them to confirm they are
dead. Same recipe as the 019 control (Adam 1e-3, per-sequence Pearson
0.5/0.5, WP-based selection, 30 epochs, seed 0), run 2026-10-08/09 on
server A. `model.py` is 004's with the dropped inputs removed (class name
kept); `keep_additional` selects the additional columns.

## Results (best batched valid_wp, corrected global WP)

| Run | dropped | best valid_wp | early-stopped |
|---|---|---|---|
| 019 control (all 112 features) | - | **0.681582** | epoch 17 |
| arm 1 | i1 price + i1 volume (44) | 0.679939 | epoch 17 |
| arm 2 | i1 price + i1 volume + a1 + a3 (46) | 0.680496 | epoch 13 |

Removing 44-46 features cost 0.0011-0.0016 WP versus the control (and
0.0019-0.0025 versus 004's published 0.682425). Control-vs-004 noise is
~0.0008, so this is roughly noise-sized, with a slight lean toward lower.
Dropping a1/a3 on top of instrument 1 did not hurt (arm 2 >= arm 1).
**Verdict: the features are close to dead weight but not entirely - dropping
them is harmless-ish, not a gain.**

## Latency (a surprise)

Dropping features does NOT make the graph faster: the 39-node graph
(vs. 004's 43) benchmarks ~2-3% SLOWER (97.9% vs 95.3% uncorrected,
medians of 5 alternating runs; arm 2's Gather-based a1/a3 drop is a bit
worse). Latency here is not a function of input width or node count in this
range. Corrected in experiment 021's README.

## Files

`results/checkpoints_*` (weights/config/ONNX for the finished arms),
`logs/train_*.log`, `submissions/022_drop_dead_features_a1a3_2026-10-08.zip`
(arm 2, packaged, NOT submitted) and `submissions/022_drop_dead_features_i1pv_2026-10-09.zip`
(arm 1).

## Platform result (arm 1, 2026-10-09)

**0.6533** (ID 1X63GTOU), runtime **37m57s** (63.2% of budget, 627 MB). Another
tie with the 004-architecture band (004 0.6530/0.6531, 019 pooled60 0.6534)
despite the lowest local WP of the family (0.679939 batched vs. 004's
0.682425). Confirms 021's finding on the platform too: instrument 1's price
and volume are dead weight - removing 44 inputs changed the platform score
by ~0 - and that local WP differences of ~0.002 do not carry over.
