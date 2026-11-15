# 036: orthogonal (non-redundant) group representations (user proposal, 2026-11-06)

**Proposal (user):** price, volume and trade features are collinear, so the group encoders of the all-feature LSTM
(i0 price 66->24, i0 volume 66->24, i1 price 66->24, i1 volume 66->24, trades 48->12) may learn overlapping, redundant
codes and waste the LSTM's 128-dim input. Penalise the overlap between groups in the loss (training only: zero nodes
added to the ONNX graph), and pair it with `CosineAnnealingWarmRestarts`.

## Implementation (code in `experiments/031_seed_ensemble_distillation/train.py`)

- Forward hooks on every `GroupEncoder`'s ReLU capture the group embeddings during training; `export_slim.py` is
  untouched, so the shipped graph and runtime are exactly the 0.66658 pair's.
- `--ortho-weight W --ortho-mode decor|cosine`:
  - **`decor` (used):** on (up to 8,192 sampled) prediction rows of the batch, standardise every unit of every group,
    C_ab = Z_a^T Z_b / N, penalty = mean over the 10 group pairs of mean(C_ab^2), the mean squared cross-correlation.
  - `cosine`: the proposal's literal version, |cos| of the sequence-mean-pooled embeddings (equal-dim pairs only).
- `--cosine-restarts T0`: after the 3 warmup epochs, per-batch cosine annealing to `--min-lr` with restarts every T0
  epochs (T0 = 7: three cycles over epochs 3-24, ending at a minimum); the fixed LR drops are then skipped.

**Why `decor` and not the literal cosine:** coordinate k of the price encoder has nothing to do with coordinate k of
the volume encoder (each feeds its own LSTM input columns), so a raw cosine between them measures no redundancy: the
network can satisfy it by permuting units with the same information. The outputs are post-ReLU (non-negative), so cosine
-> 0 forces the groups to switch off DIFFERENT units, an arbitrary loss of capacity. And the trade group is 12-dim vs
24, so the cosine is undefined for 4 of the 10 pairs. Cross-correlation over rows is zero exactly when no linear
combination of one group's units correlates with any of another's: the "orthogonal subspaces" the proposal wants,
invariant to unit order and defined for any dimensions. Weight 0.5 (penalty ~0.006-0.012 at initialisation).

## Runs (local 1080 Ti, both at once, ~46 min/epoch -> ~06:00 on 11-07)

| run | change vs `R23_FULL_LSTM_LA12` (0.66597 single, 0.66658 in the mixpair) |
|---|---|
| `R27_FULL_LSTM_ORTHO` | + decor penalty 0.5 (step schedule unchanged): the clean test |
| `R27_FULL_LSTM_ORTHO_CWR` | + decor penalty 0.5 + cosine warm restarts (T0 7): the proposal's full package |

Same targets (0.5 TV12 + 0.5 LA12), recipe A, seed 0, train + valid. `build_036.sh` packages each as a single and as a
mixpair with the existing lean GRU LA50 (R16): `036_full_LSTM_LA12_{ORTHO,ORTHO_CWR}_single`,
`036_mixpair_fullLSTM_LA12_{ORTHO,ORTHO_CWR}_leanGRU_LA50` (~50-51 min).

Caveat from experiment 031: a 28-epoch schedule was worse than 24 (-0.0008), and the step schedule has been tuned for a
month, so the warm-restart arm changes two things; it is there because the proposal pairs them.

## Platform results (2026-11-07)

`036_mixpair_..._ORTHO_leanGRU_LA50` **0.66634** (50m42s); `..._ORTHO_CWR_...` **0.66634** (53m16s). Both -0.00024 vs
the 0.66658 pair (one change each): no gain from decorrelating the group encoders; warm restarts change nothing.
