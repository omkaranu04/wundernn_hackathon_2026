# 021: Model-based permutation feature importance (no training)

Analysis only. On trained 004 (30ep) and 012, each of the 112 input features
(and groups of them) had its 20,000-step trace in every sequence replaced by
another sequence's trace (batch roll): same marginal distribution and
temporal structure, broken alignment with the targets. Drop in corrected
global WP (019's validated accumulator) = reliance. 192 evenly spaced valid
sequences, two different roll offsets per condition (their spread is the
noise estimate; baseline WP on the subset 0.6848 / 0.6837). Run 2026-10-08
on server A. `importance.py`, `results_{004,012}.json`, `imp_*.log`.

## Result: the two models agree closely

Group importance (drop in WP when permuted; 004 / 012):

| group | n | 004 | 012 |
|---|---|---|---|
| **ALL instrument 0** | 52 | **0.327** | **0.335** |
| ALL volume (i0+i1) | 44 | 0.104 | 0.112 |
| i0_volume | 22 | 0.090 | 0.100 |
| additional | 8 | 0.098 | 0.090 |
| ALL trade dp+dv | 16 | 0.096 | 0.093 |
| i0_price | 22 | 0.047 | 0.045 |
| **ALL instrument 1** | 52 | **0.036** | **0.030** |
| i1_dv / i0_dv / i1_dp / i0_dp (4 each) | | 0.014 / 0.009 / 0.007 / 0.005 | 0.016 / 0.010 / 0.006 / 0.009 |
| **i1_price** | 22 | **0.0026** | **0.0011** |
| **i1_volume** | 22 | **0.0017** | **0.0018** |

Findings:
1. **Instrument 0 carries almost everything**: removing its information
   costs ~0.33 WP versus ~0.03 for all of instrument 1.
2. **Instrument 1's price and volume (44 features) are essentially unused**
   by both models (group drops 0.001-0.003, within noise). They are NOT
   redundant with instrument 0 (mean max |corr| with any i0 feature of the
   same type is only ~0.18-0.22), and they barely correlate with the targets
   directly (mean |corr| 0.015-0.017 vs 0.074 for i0 price). So they carry
   different information that the models do not use - either it is not
   predictive, or it is hard to extract with this architecture. Instrument
   1's trade features (dp/dv) do matter a little (0.005-0.016).
3. **Within instrument 0, volume and trade matter more than price** (volume
   0.09-0.10, price 0.045). Volume-heavy, price-light.
4. **The `additional` features are important and uneven**: a6, a7, a5, a0
   top the single-feature list (a6 alone: 0.023/0.015); a1 and a3 have
   slightly NEGATIVE drops (-0.0003 to -0.0013), i.e. permuting them
   helps marginally - candidates for removal.
5. **Most single features are individually dispensable**: 69/112 (004) and
   71/112 (012) have a drop under 0.0005. Per-feature drops sum to far less
   than group drops (top-10 single drops sum to 0.08-0.11 vs 0.33 for i0),
   which is the redundancy signature: features within a group substitute for
   each other.
6. **Top single features** (both models): a6, i0_v15, a7, i0_v21, i0_v0,
   i1_dp2, a5, a0, i0_v4, i0_v19.

## Caveats

Permutation importance measures reliance by THIS trained model, not the
value of a feature to a model retrained without it, and it under-credits
redundant features (the model re-routes to a correlated sibling). It ranks;
it does not prove dispensability. Group drops are more trustworthy than
single-feature ones (noise per single feature ~0.0003-0.001).

## What this suggests (not yet tried)

- **Drop i1_price and i1_volume (44 features) and retrain.** Cheap, clean
  test of finding 2: if WP holds, the features are confirmed dead.
  CORRECTION (2026-10-08): I originally wrote that this would also lower
  latency by removing ~3-5 ONNX nodes. Measured, it does not: the 39-node
  graph without those features benchmarks SLOWER than 004's 43-node graph
  (97.9% vs 95.3% uncorrected, medians of 5 alternating runs), so dropping
  features is not a latency lever here. See experiment 022.
- Also drop `a1`, `a3`.
- Conversely, since i1 is not redundant but unused, a model that
  *combines* instruments explicitly (e.g. i0-i1 differences/spreads) is an
  untested way to try to extract what the current architectures miss. Speculative.

## 012-specific: which expanded channel does the model use? (2026-10-08)

012 expands every group into `[x, asinh(x), sin(x), cos(x), tanh(x)]`.
`channel_importance_012.py` permutes one channel type (batch roll) AFTER the
expansion, leaving the others intact. Drop in WP (192 seqs, 2 rolls;
`channels_012.json`):

| channel permuted (all 4 groups) | dWP |
|---|---|
| x (raw) | +0.0972 |
| **cos(x)** | **+0.0389** |
| asinh(x) | +0.0077 |
| sin(x) | +0.0017 |
| tanh(x) | -0.0003 |

By group (price / volume / trade / additional): cos(x) = 0.0096 / 0.0099 /
0.0081 / 0.0010; x = 0.0037 / 0.0197 / 0.0159 / 0.0108.

**Reading, with a caveat.** asinh, sin and tanh are all (near-)monotone
copies of x over this input range, so once `x` is present they are largely
redundant (small drops). cos(x) is the only EVEN function in the set - it
carries magnitude/|x| information (~1 - x^2/2) that no other channel
encodes - and the model relies on it much more than on any other added
channel. Caveat: permuting a single channel produces inputs that are
internally inconsistent (x no longer matches its own asinh/cos), so the
large drop for `x` in particular overstates how much the model "needs" it;
this measures damage from broken consistency as well as reliance. It is
suggestive of which channels are redundant, not proof.

**Hypothesis worth testing by retraining (not yet done):** most of 012's
gain over 004 comes from an even/magnitude channel. A leaner 012 using only
`[x, cos(x)]` (or `[x, |x|]` / `[x, x^2]`) per group would drop the Asinh,
Sin and Tanh nodes (~12 fewer ops across 4 groups) and might match 012's
WP at lower latency (012 is 112.2% uncorrected).
