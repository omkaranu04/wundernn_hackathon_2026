# 040: numba engine v3 and four-member ensembles (2026-11-07)

**Trigger:** `039_NUMBA_triple_fullLSTM_LA12_leanGRU_LA50_leanLSTM_LA25` scored **0.66676 (new best) in 41m21s**:
the numba engine runs on the platform at x0.59 of local time (107.5 us/row -> 63 us/row), the same ratio as ONNX.
A third member came 9.5 min CHEAPER than the ONNX pair, so the question became: how many members fit?

## 1. Engine v3 (`031.../numba_engine3.py`, `solution_numba3.py`, `make_zip_numba.py --engine 3`)

Profile of v2 (pinned core, 4,000 real rows): kernel alone 35.7 / 24.4 / 28.4 us (full LSTM / lean GRU / lean LSTM)
vs 41.8 / 29.7 / 33.3 us through numba's dispatcher: **~5.5 us per member per row was argument unboxing** (25 array
arguments per call). v3 reads the same npz, flattens all members into fp32 / int16 / int32 arrays + an int64 layout
table, and runs the whole ensemble in ONE compiled call (8 arguments); cos/sin of the 112 inputs computed once per
row and shared. Same arithmetic: v3 vs v2 max diff 8e-7. Kernel variants tried and rejected: 8 inputs per pass
(slower), 32-wide output blocks (5x slower).

## 2. Zips (all verified vs torch <= 6e-5; ONNX fallback = the 0.66658 pair graph, plain session.run)

Local full predict loop (zip_time.py, 4 x 3,000 real rows, pinned core, interleaved, 3 reps); platform projection
= 41m21s x local / 107.5:

| zip (`040_NUMBA3_...`) | members | local us/row | projected |
|---|---|---|---|
| `triple_fullLSTM_LA12_leanGRU_LA50_leanLSTM_LA25` | the 0.66676 triple | 95.5 | ~36.7 min |
| `quad_triple_plus_leanGRU_LA25` | + R12_LA25_GRU | 121 | ~46.5 min |
| `quad_triple_plus_leanLSTM_LA50` | + R6_LA50_A (single 0.66503) | 125 | ~48.2 min |
| `quad_triple_plus_fullGRU_LA50` | + R23_FULL_GRU_LA50 (single 0.66485) | 126.5 | ~48.7 min |
| `quad_triple_plus_fullLSTM_LA12s1` | + R25_FULL_LSTM_LA12_s1 (seed replicate of member A) | 132 | ~50.8 min |

Equal weights (0.25). A five-member ensemble projects to ~56-58 min: NOT shipped (no grace-period targeting).

## 3. Member choice: in-sample scans do not work

Predictions of 16 members on 858k scored valid rows: every member was trained on valid, so in-sample WP rewards
memorisation (fine-tuned FT_GRU 0.742, GRUs > LSTMs) and the triple scores BELOW the pair in-sample (0.7181 vs
0.7199) while it gained +0.00018 on the platform. Correlations with the triple: 0.9965 (full GRU LA50) to 0.9989.
Members are chosen by priors (different architecture / look-ahead window / seed) and judged on the platform.

## 4. New full-feature members (training)

- `R29_FULL_LSTM_LA25` (local 1080 Ti, ~24 min/epoch, ETA ~23:00 11-07): all-feature LSTM on LA25 labels, to
  replace the lean LSTM LA25 (the full architecture was +0.0004 as a single).
- `R29_FULL_LSTM_LA50` (server B GPU 1, shared with another user, ~30 min/epoch, ETA ~02:30 11-08).

## 5. Member weights: train-only ("honest") twins (queued 2026-11-08 00:05)

Every shipped member trained on valid, so weights cannot be fitted there. Each member gets a twin trained on train
only (`run_student_honest.sh`: recipe A minus `--extra-train`), weights fitted on valid (non-negative, sum 1,
sequence bootstrap), shrunk halfway to equal, then one platform slot vs the equal-weight score. Note sunArise's
"0.5/0.25/0.25" is 0.5 per STRUCTURE (the two 0.25s are two seeds of one fusion member).
Shared queue `queue_honest.txt` (server A/server B share the home; `queue_honest_worker.sh`, mkdir lock):
H2_FULL_LSTM_LA12, H2_LEAN_GRU_LA50, H2_LEAN_LSTM_LA25, H2_FULL_GRU_LA50, H2_LEAN_GRU_LA25. Workers: server B GPU0 after
the ELA6 pipeline, server B GPU1 after R29_FULL_LSTM_LA50, server A GPU0/1 only when idle 15 min. Local: H2_LEAN_LSTM_LA50
after R29_FULL_LSTM_LA25 (`queue_honest_local.sh`). Soft labels on train come from teachers that saw valid (TV12):
a mild leak shared by all twins, acceptable for RELATIVE weights.

## Platform results (2026-11-08)

| zip | platform | runtime | projected | local us |
|---|---|---|---|---|
| `040_NUMBA3_quad_triple_plus_fullGRU_LA50` | **0.66690 (NEW BEST, +0.00014 vs triple)** | 50m46s | 48.7 | 126.5 |
| `040_NUMBA3_quad_triple_plus_leanLSTM_LA50` | 0.66683 | 50m37s | 48.2 | 125 |
| `040_NUMBA3_quad_triple_plus_leanGRU_LA25` | 0.66682 | 49m02s | 46.5 | 121 |
| `040_NUMBA3_quad_triple_plus_fullLSTM_LA12s1` | 0.66681 | 53m06s | 50.8 | 132 |

- Every 4th member adds +0.00005..+0.00014: the ensemble is saturating (members share ~99% of their errors). The
  most different member (all-feature GRU) helps most, the seed replicate least.
- **Runtime calibration for v3 zips: platform ~0.40 min per local us** (v2's 0.385 under-projected v3 by ~4-5%,
  ~2 min). A five-member zip (~150 us) would be ~60 min: impossible without a faster engine.

## 6. Swap quads (built 2026-11-08 12:15): best quad with its weakest member (lean LSTM LA25) replaced

Base = all-feature LSTM LA12 + lean GRU LA50 + all-feature GRU LA50 (the 0.66690 quad minus R6_LA25_A). New members
`R29_FULL_LSTM_LA25` (local) and `R29_FULL_LSTM_LA50` (server B), both recipe A on all features, finished 11-08 morning.
Projection = 50m46s x local / 127.3 (the 0.66690 quad timed in the same run).

| zip (`041_NUMBA3_quad_fullLSTM_LA12_leanGRU_LA50_fullGRU_LA50_plus_...`) | local us | projected |
|---|---|---|
| `fullLSTM_LA25` | ~134 | ~53.4 min |
| `fullLSTM_LA50` | ~131 | ~52.2 min |
| `leanLSTM_LA50` | ~124 | ~49.5 min |
| `leanGRU_LA25` | ~121 | ~48.3 min |

Two all-feature LSTMs cost the same as the seed-replicate quad, which ran 53m06s.

## 7. Honest twins scored on valid; weight fit (2026-11-09)

Six train-only twins (recipe A without `--extra-train`, valid unseen), all 1873 valid sequences, 3,528,273 scored rows.
Absolute values sit above the platform (valid regime); only differences matter.

| twin | single WP |
|---|---|
| FULL_LSTM_LA12 (A) | 0.69851 |
| LEAN_LSTM_LA25 (C) | 0.69771 |
| LEAN_GRU_LA25 | 0.69708 |
| LEAN_LSTM_LA50 | 0.69694 |
| FULL_GRU_LA50 (D) | 0.69674 |
| LEAN_GRU_LA50 (B) | 0.69607 |

Ensembles (equal weights): pair A+B 0.69831, triple ABC 0.69857, best quad ABCD 0.69871 (platform: triple +0.00018
over the pair, quad +0.00014 over the triple; valid: +0.00026, +0.00014 - same direction). ABD + X: lean LSTM LA25
0.69871 (current), lean LSTM LA50 0.69858, lean GRU LA25 0.69855 - the 041 lean swaps are predicted slightly worse.
Best 4-of-6: A + C + D + lean LSTM LA50 = 0.69878 (+0.00007, drops the lean GRU LA50); all six 0.69866.

Fitted weights for ABCD: [0.54, 0.05, 0.20, 0.21] in-sample 0.69895, but over 30 sequence bootstraps the weights
are unstable (sd 0.10-0.19) and the out-of-bag gain over equal weights is -0.00013 +- 0.00028 (positive in 47%).
**Equal weights stay; no fitted-weight submission.**

Built `041_NUMBA3_quad_fullLSTM_LA12_leanLSTM_LA25_fullGRU_LA50_leanLSTM_LA50_2026-11-09.zip` (A+C+D+R6_LA50_A,
verified 4.4e-5): local 127.2 us vs 124.0 us for the 0.66690 quad in the same run -> ~52.1 min projected.

## 8. Platform 2026-11-09: swap quads (A + B + D + X)

| X (replaces lean LSTM LA25) | platform WP | runtime | projected |
|---|---|---|---|
| all-feature LSTM LA50 (R29_FULL_LSTM_LA50) | **0.66709 (NEW BEST)** | 52m07s | 52.2 |
| lean GRU LA25 (R12_LA25_GRU) | 0.66687 | 47m07s | 48.3 |
| lean LSTM LA50 (R6_LA50_A) | 0.66686 | 51m47s | 49.5 |
| all-feature LSTM LA25 (R29_FULL_LSTM_LA25) | 0.66675 | 53m26s | 53.4 |

- New best: all-feature LSTM LA12 + lean GRU LA50 + all-feature GRU LA50 + all-feature LSTM LA50, +0.00019 over
  0.66690, 7m53s margin.
- The two all-feature LSTMs (LA25 vs LA50, same recipe) differ by 0.00034: member-level luck is about that size, so
  the new best is partly noise; every quad sits in 0.66675-0.66709.
- The honest-valid prediction for the two lean swaps (slightly below the 0.66690 quad) held: -0.00003 / -0.00004.
- Runtime projections were within 0.1 min for the two full-LSTM quads but off by +2.3 / -1.2 min for the lean ones:
  treat projections as +-2 min and keep them at or below ~52 min.

## 9. Fine-tune swap (built 2026-11-09)

`042_NUMBA3_quad_newbest_with_fullLSTM_LA12ft_2026-11-09.zip`: the 0.66709 quad with the all-feature LSTM LA12
replaced by its scored-row fine-tune `FT_FULL_LSTM_LA12` (039 section 3; never scored on the platform). Same
shapes as the 0.66709 quad, so the same runtime (~52 min). Verified 3.5e-5. md5 e6f20bec16aba97b0dc2e5cf7fa21773.

Platform 2026-11-09: **0.66717 (NEW BEST, +0.00008 over 0.66709), 53m21s**. The same shapes as the 0.66709 quad
ran 74 s longer (52m07s): platform runtime noise for identical arithmetic is about +-1.2 min.
Follow-up: fine-tune the other two all-feature members (`FT_FULL_GRU_LA50`, `FT_FULL_LSTM_LA50`, local) and score
quads with 2-4 fine-tuned members.

## 10. Offline fine-tune recipe search (043, 2026-11-09)

CORRECTED: the EMA in `train.py` warms its decay up as min(D, (1+n)/(10+n)), so over a ~700-step fine-tune it never
gets near 0.9998 (0.987 at the end) and `ema.pt` is a trajectory average close to the fine-tuned weights (it is NOT
a 13% move, as first written here; the --ema-decay value hardly matters for fine-tunes). `run_ft_search.sh`: the HONEST twin H2_FULL_LSTM_LA12 is fine-tuned on half of valid
(`--exclude-fold F --n-folds 2`), scored on the other half by `ft_eval.py`, which also scores
WiSE-FT weight interpolations base + a (last - base). Gain over the base, fold 0 / fold 1 / mean:

| recipe | ema (trajectory avg) | a=0.25 | a=0.5 | a=0.75 | a=1.0 |
|---|---|---|---|---|---|
| r0 (lr 1e-4, 6 ep, distill 0.8) | -4 / +62 / +29 | +14 / +30 / +22 | +21 / +52 / +37 | +18 / +67 / +43 | +5 / +73 / +39 |
| lr 3e-4 | -3 / +81 / +39 | +26 / +42 / +34 | **+36 / +68 / +52** | +27 / +76 / +52 | -1 / +63 / +31 |
| 12 epochs | -1 / +84 / +42 | +19 / +38 / +29 | +23 / +63 / +43 | +12 / +73 / +43 | -20 / +67 / +24 |
| distill 0.5 | -97 / +65 / -16 | +7 / +41 / +24 | -8 / +69 / +31 | -52 / +83 / +16 | -128 / +81 / -24 |

(units 1e-5 WP.) The folds disagree in size (fold 1 gains 2-3x more) but agree on the shape: a higher LR followed by
moving only halfway from the base beats both the EMA and the full fine-tune (a=1 is past the optimum on fold 0). **Chosen: lr 3e-4, a = 0.5** (best mean,
best worst-fold). Caveat: the twin never saw valid, so its fine-tune gain overstates the production members' (which
trained on valid); platform evidence so far: +0.00008 for one member at the old strength.
`run_ft3_local.sh` applies it to the four members of the 0.66717 quad -> `FT3_*_W50` (`interp_ckpt.py`).

Built 2026-11-09 14:47: `043_NUMBA3_quad_FT3W50_ABDE` (all four members FT3 a=0.5) and `043_NUMBA3_quad_FT3W50_A`
(only the all-feature LSTM LA12), verified 4.1e-5 / 4.0e-5. Prediction corr vs base on valid (in-sample): FT3 W50
0.9986-0.9995 (scale x1.011-1.019); the old FT (0.66717) 0.9988 (x1.024) - the W50 members move LESS than the old FT.

## 11. Model soups of the fine-tunes (offline, 2026-11-10)

`soup_eval.py H2_FULL_LSTM_LA12`: uniform weight averages of the search's fine-tunes (last.pt, same base, same half
of valid), with WiSE interpolation a; gain over the base on the held-out half (fold 0 / fold 1 / mean, 1e-5):
lr3 alone a=0.5 +36/+68/**+52** (the chosen FT3W50); r0+lr3 a=0.75 +27/+76/+52; r0+lr3+ep12 a=0.75 +23/+77/+50;
lr3+ep12 a=0.75 +22/+78/+50; all four a=0.75 +9/+81/+45. **Soups add nothing over lr 3e-4 + a=0.5** - the
interpolation already captures the averaging gain. Closed.

## 12. Platform 2026-11-10: fine-tuning more members HURTS; runtime +4 min on identical arithmetic

| zip | WP | runtime |
|---|---|---|
| 043 FT3W50_A (only A, lr3e-4 a=0.5) | 0.66711 | 57m11s |
| 042 FT_A_E (old FT on A and E) | 0.66709 | 54m07s |
| 043 best with fullLSTM_LA50_V3 (valid x3) | 0.66694 | 57m12s |
| 043 FT3W50_ABDE (all four, lr3e-4 a=0.5) | 0.66672 | 57m27s |
| 042 FT_ABDE (old FT on all four) | **0.66589** | 54m37s |

Best stays 0.66717 (042 newbest, old FT on A only). Every A-only / A+E variant lands 0.66709-0.66717 = noise around
the un-fine-tuned 0.66709; fine-tuning all four costs -0.0005 (W50) to -0.0013 (full). The offline half-valid test
measured ONE model and missed the ensemble effect: fine-tuning every member on the same data pulls them together and
the diversity loss outweighs the per-member gain. V3 (valid oversampled x3) -0.00015 vs its sibling. **Fine-tuning and
valid oversampling are closed**; head-only FT not pursued (same mechanism).

RUNTIME: the five zips are the same arithmetic as the 0.66717 quad (52m07s / 53m21s on 11-09) but ran 54m07s-57m27s.
Observed range for this quad is now 52m07s-57m27s (5.3 min spread), not the +-1.2 min assumed before; the margin to
60 min was 2m33s at worst.

## 13. Engines v4 / v5 (2026-11-10): 0.66x of v3's per-row time, same score

Profile of v3 on the 0.66717 quad (local E5-1660 v4, pinned core): kernel 128 us/row = gemvs ~83 + activations ~21 +
encoders ~16 + cos/sin ~2 + allocations ~6. Findings:
* numba's default `error_model="python"` puts a ZeroDivisionError branch before every float division, which blocks
  vectorization of the rational tanh/sigmoid (asm: scalar `vdivss`). `error_model="numpy"` -> activations 21 -> 6 us.
* the encoder (24-output dot products of length <= 66) as axpy over outputs with transposed weights, NOT inlined
  into the big kernel (inlined it ran ~7x slower): 16 -> ~7 us. One preallocated scratch instead of ~45 np.empty.
* float gemv (int16 -> fp32 convert + FMA) runs 11.5-15 GMAC/s whatever the unroll (4 inputs/pass is best).
  An integer pair multiply-add (int16 weights x int16 activations -> int32) written as generic LLVM vector IR
  through a numba `@intrinsic` (x86 lowers it to vpmaddwd; SSE2 elsewhere) runs 21-28 GMAC/s. Activations are
  quantized per row and per input segment with qmax = floor((2^31-1)/max_o sum_j|w_jo|) (~1050-1300 for our
  members), so int32 cannot overflow for ANY input. Weights are the same int16 values (re-laid out in input pairs).

| engine | kernel us/row | end-to-end predict() us/row | vs v3 | diff vs v3/v4 | valid WP (64 seqs, 127k scored rows) |
|---|---|---|---|---|---|
| v3 (`numba_engine3.py`) | 128 | 127.3 | 1.00 | - | - |
| v4 (`numba_engine4.py`) | 98 | 103.4 | 0.81 | max 1.0e-5 vs v3 over 160k rows | 0.719489 |
| v5 (`numba_engine5.py`) | 77 | 84.0 | 0.66 | max 2.1e-3 vs v4, corr 0.99999998 | 0.719491 |

`solution_numba5.py` chain: v5 if it compiles AND agrees with v4 within 1e-2 on 300 synthetic rows (local diff
~1e-3) -> v4 -> ONNX. Tested: deleting numba_engine5.py from the zip falls back to v4. Zips (2026-11-10):
`044_NUMBA4_quad_newbest` (v4, 0.66717 models), `045_NUMBA5_quad_newbest` (v5, same models), and five-member
equal-weight zips (0.66717 quad + one lean member; ruled out before ONLY by runtime): `045_NUMBA5_quint_plus_R6_LA25_A`
(lean LSTM LA25, 101 us), `..._plus_R12_LA25_GRU` (lean GRU LA25, 97.5 us), `..._plus_R6_LA50_A` (lean LSTM LA50,
93 us) - all below the v3 quad's 127 us, i.e. ~42-45 min projected (platform/local 0.42 min per us; +8% slow day).

## 14. New directions started 2026-11-10

* **Post-processing (closed, offline):** all-row predictions of the 4 honest twins (H2 full LSTM LA12, lean GRU LA50,
  full GRU LA50, lean LSTM LA50) on every 3rd valid sequence (625), equal-weight mean, two halves of sequences.
  Gain vs base (1e-5, half 0 / half 1): causal EMA a=0.95 -6.5/-6.8, a=0.9 -17/-18, a=0.8 -52/-53; extrapolation
  p+0.1(p-p_prev) +1.7/+2.1 (noise), b=0.2 -11; cross-mixing the two targets +-0.05 -38..-50; scale x1.3 -27,
  x0.6-0.9 -1 (only 0.017% of predictions exceed the clip); shrink by member disagreement -15..-107. Nothing.
* **New members (never trained): full GRU on LA25 and LA12 labels** (only full GRU LA50 exists; full LSTM has
  LA12/25/50). R23 recipe (train+valid, 24 epochs, floor 0.1, seed 0): `R31_FULL_GRU_LA25` server A GPU 0,
  `R31_FULL_GRU_LA12` server B GPU 0, launched ~11:00, ~25 min/epoch -> ~21:00-23:00. `logs/watch_r31.sh` (local,
  background) fetches each when it finishes and builds `046_NUMBA5_quint_plus_<N>` (0.66717 quad + it) and
  `046_NUMBA5_quad_swapD_<N>` (it in place of the LA50 full GRU).
* 2026-11-10 evening: `R31_FULL_GRU_LA25` (exit 15:50) and `R31_FULL_GRU_LA12` (17:48) finished; the watcher built
  `046_NUMBA5_quint_plus_R31_FULL_GRU_LA{25,12}` (92.5 / 98 us local) and `046_NUMBA5_quad_swapD_R31_FULL_GRU_LA{25,12}`
  (84 us). Diversity on valid (in-sample, every 3rd sequence, scored rows), corr with the 0.66717 quad's mean:
  current members 0.9969-0.9975; new full GRU LA25 0.9966, LA12 0.9967; lean LSTM LA25 (R6_LA25_A) 0.9962 -> the new
  members are slightly LESS correlated with the ensemble than any current member (i.e. they add diversity).

## 15. Target structure probe (2026-11-10, offline, nothing shipped)

60 train sequences (170k rows): t0 and t1 have identical quantiles and corr -0.74. t0-t1 is symmetric (skew -0.03),
t0+t1 is strongly right-skewed (skew 3.6) and nearly uncorrelated with t0-t1 (-0.03). Single features correlate with
t0+t1 up to 0.31 but with t0-t1 at most 0.07. A market mirror would SWAP t0/t1 (not negate them), but the feature
side is hidden: bid column k vs ask column k+11 do not pair up under the anonymization (means/stds/correlations
differ), so a direct flip augmentation is impossible. Open (speculative, not started): learn the feature mirror M
such that a trained model's outputs swap under M, then use M as free training augmentation.

## 16. Platform 2026-11-11: engine v5 confirmed; NEW BEST 0.66735 (new full GRU LA25 replaces the LA50 full GRU)

| zip | WP | runtime |
|---|---|---|
| 046 quad swapD: A + B + **R31_FULL_GRU_LA25** + E | **0.66735 NEW BEST** | 36m06s |
| 046 quint: 0.66717 quad + R31_FULL_GRU_LA25 | 0.66730 | 38m01s |
| 045 quint: 0.66717 quad + lean LSTM LA25 (R6_LA25_A) | 0.66723 | 44m57s |
| 045 quad: the 0.66717 members on v5 | 0.66717 (identical) | **33m02s** (v3: 52m07s-57m27s) |
| 046 quint: 0.66717 quad + R31_FULL_GRU_LA12 | 0.66716 | 44m26s |

- v5 on the platform: same score, 33m02s vs 52-57 min for v3 -> 0.58-0.63x (local estimate 0.66x). Runtime noise
  persists (quints with near-identical arithmetic: 38m01s vs 44m26s/44m57s).
- The never-trained full GRU on LA25 labels is the best single addition: +0.00018 replacing the LA50 full GRU,
  +0.00013 as a fifth member; the LA12 version adds nothing (-0.00001). A fifth lean LSTM LA25 adds +0.00006.
- 11-11: built `047_NUMBA5_swapD25_plus_R6_LA25_A` and `..._plus_R12_LA25_GRU` (quints on top of 0.66735);
  launched `R32_FULL_GRU_LA25_s1` (seed 1 of the new best member) on server B GPU 0 (server A unreachable).

## 17. Test-time adaptation idea (2026-11-11, offline oracle, closed)

Inspired by online learning in Jane Street Kaggle solutions: per-sequence causal recalibration y ~ a p + b fitted on
the TRUE targets of rows <= t-LAG (an upper bound: true targets are not available at test time), honest 4-twin mean,
625 valid sequences, halves. LAG 100: -0.004..+0.011 depending on shrinkage and wildly different between halves
(overlap leakage: targets 100 rows apart share their future window); LAG 500 (targets surely realized): -0.006 to
-0.062. Even with oracle labels, per-sequence recalibration hurts -> closed.

## 18. Platform 2026-11-13 (11-12 slots lost to a resubmission of the 11-10 zips)

| zip | WP | runtime |
|---|---|---|
| 048 quint: 0.66735 quad + R32_FULL_GRU_LA25_s1 | **0.66737 NEW BEST** | 42m42s |
| 047 quint: 0.66735 quad + lean GRU LA25 (R12) | 0.66733 | 41m51s |
| 047 quint: 0.66735 quad + lean LSTM LA25 (R6) | 0.66732 | 44m57s |
| 048 quad: 0.66735 quad with seed 1 instead of seed 0 | 0.66727 | 34m01s |
| 047 quad swapE: A + B + D(LA50 GRU) + R31 GRU LA25 | 0.66708 | 34m02s |

Seed spread of the new member: 0.66735 / 0.66727 (0.0001) - the top five candidates (0.66727-0.66737) are all
inside that noise. The full LSTM LA50 (E) matters (-0.00027 when dropped). Fifth members add +0.00002 at most on top
of 0.66735. Built `050_NUMBA5_bestquint_unFT_A` (= 0.66737 quint with the un-fine-tuned R23_FULL_LSTM_LA12, a
robustness variant: the fine-tune was fitted on valid).
* 2026-11-13 ~11:00: launched `R34_FULL_LSTM_LA50_s1` (seed 1 of E, the member whose removal costs most) on server B
  GPU 0, ~12.5 h. `logs/watch_r34.sh` builds `051_NUMBA5_swapD25_plus_R34_Es1` (0.66735 quad + E seed 1) and
  `051_NUMBA5_bestquint_Es1_for_E` (0.66737 quint with E seed 1 instead of E) for 11-14. Internet search 11-13:
  no new technique; diversity of targets/architectures > seeds (consistent with our results); LAmix labels lost before.

## 19. Platform 2026-11-14 (last full submission day): NEW BEST 0.66754

| zip | WP | runtime |
|---|---|---|
| 049 quint: A + B + G25 s0 + E + **G25 s2** | **0.66754 NEW BEST** | 47m01s |
| 049 quad: A + B + **G25 s2** + E | 0.66750 | 37m16s |
| 051 quint: 0.66735 quad + full LSTM LA50 seed 1 | 0.66734 | 49m22s |
| 050 quint: the 0.66737 quint with un-fine-tuned A | 0.66730 (FT: 0.66737) | 44m02s |
| 051 quint: the 0.66737 quint with E seed 1 instead of E | 0.66718 | 43m16s |

Full GRU LA25 seeds in the quad: s0 0.66735 / s1 0.66727 / s2 0.66750 -> seed spread 0.00023, as large as most
recipe effects of the last week. The 0.66754 quint averages two GRU seeds (s0 + s2), so its gain over the s2 quad
is the more robust part. E's seed 1 is worse than seed 0 in both uses; fine-tuned A +0.00007 over un-fine-tuned.
Runtime: the 0.66754 quint's arithmetic equals the 0.66737 quint's (42m42s on 11-13) but ran 47m01s.

## 20. 2026-11-14 afternoon: engines v6/v7, seed-averaged sextets for 11-15

* Profile of v5 on the 0.66754 quint: integer gemvs ~75 us, encoders ~20 us in isolation; but **v6** (encoders through
  the integer path; also tried 24 register accumulators - slower) gained only 2.3% end-to-end: the gemvs are
  memory-bandwidth bound and hide the encoder cost. Not shipped.
* **v7 (`numba_engine7.py`)**: int8 weights (per-output scale), same pair multiply-add with int8 loads; small weights
  allow finer activations (qmax up to 32767). Interleaved local timing 0.81x of v5 (0.91x in a run perturbed by the
  local training job). Valid WP on 64 sequences (140k scored rows), 0.66754 quint: v5 0.681071, v7 0.681121
  (+0.00005; per-chunk -27..+19e-5, i.e. a noise-level perturbation, corr 0.999993).
* `solution_numba7.py` chain: v7 (|diff vs v4| < 0.1 on 300 synthetic rows) -> v5 (< 1e-2) -> v4 -> ONNX; tested by
  deleting engine files from an extracted zip (numba7 / numba5 / numba4 selected as expected).
* Seed-averaged GRU slot: A, B, E keep 0.2 each; three full GRU LA25 seeds share 0.4 (0.1333 each) - the 0.66754
  balance with less seed luck (seed spread 0.00023). v7 sextet = 1.05x the v5 quint locally (~45 min, ~49.5 slow day).
  Built: `053_NUMBA7_bestquint_int8` (0.66754 members on v7, engine calibration), `053_NUMBA7_sext_G25seeds012`.
  `R35_FULL_GRU_LA25_s3` training on the local 1080 Ti (~19:45); `logs/watch_r35.sh` builds
  `054_NUMBA7_sext_G25seeds023` and `054_NUMBA7_sext_G25seeds123`.

## 21. Internet round 2026-11-14 (harder) + two offline tests (closed)

Sources: MarkBrezina research log (rival, best local 0.679 vs a 0.672 LB benchmark; measured win only scored-row head
FT +0.0008 - we have the FT), krish-rRay23 wunderTrader (0.6517; global WP between-sequence component r_b 0.77;
EMA smoothing cost up to 22.9 bps), KlishevDA (11th, earlier Wunder RNN challenge: 4 averaged heads +0.003 R2,
3 models on different data splits +0.003 R2). Tests on the honest 4-twin mean (625 valid sequences, halves):
* re-weighting a causal running sequence mean of the predictions (p' = a m + (p - m), windows all/2000/500,
  a 0.5-1.5): -0.001 to -0.017 everywhere;
* step-dependent affine calibration (20 step bins, cross-fitted between halves): -0.0014 / -0.0036.
Both closed: the ensemble is already calibrated across sequences and steps.

## 22. Multi-head members (2026-11-14 evening; idea: KlishevDA's 4 averaged heads)

`mh.py` FullFeatureMultiHead: K heads 128->64->2 on one trunk, each trained on its own loss (`train_mh.py` =
train.py + `--n-heads`, per-head losses averaged via `head_loss`); predictions = mean of heads. The mean of K
two-layer heads is exactly one head with K*64 hidden units (`mh.py to_single`: stacked first layers, concatenated
second layers / K) -> standard checkpoint, no packing/engine change. Smoke test (CPU, 3 batches, K=4): converted vs
multi-head max diff 1.5e-8; packed Z=256; v5 vs torch 6e-5, v7 3e-4.
Queued: `logs/chain_mh.sh` (local 1080 Ti, after R35 ~19:45): `R36_FULL_GRU_LA25_MH4` (seed 4, ~5 h), convert,
in-sample valid WP vs the four single-head GRU LA25 seeds (same recipe/data -> comparable), zips
`055_NUMBA5_quad_swapD_MH4`, `055_NUMBA5_quint_s2_MH4`, `055_NUMBA7_sext_s02_MH4`.
`logs/poll_server B_mh.sh`: starts `R37_FULL_GRU_LA25_MH4_s5` on server B if a GPU frees >= 20 GB before 23:30.
* 2026-11-15 morning: `R35_FULL_GRU_LA25_s3` finished 19:32; `054_NUMBA7_sext_G25seeds023` / `..._123` built.
  `R36_FULL_GRU_LA25_MH4` FAILED at start (21:44, CUDA OOM: another process held 10.15 GB of the local 1080 Ti);
  server B never freed (poller gave up 23:30) -> multi-head untested. Submitted 11-15: the three v7 sextets,
  `053_NUMBA7_bestquint_int8`, `052_NUMBA5_quint_G25_s1s2`.
* 2026-11-15 platform scores (last submission day):
  `053_NUMBA7_bestquint_int8` 0.66753 / 38m36s (same members as the 0.66754 v5 quint at 47m01s: int8 costs
  0.00001, saves 8.5 min); `054_NUMBA7_sext_G25seeds023` 0.66752 / 44m06s; `053_NUMBA7_sext_G25seeds012`
  0.66749 / 42m37s; `052_NUMBA5_quint_G25_s1s2` 0.66748 / 43m46s; `054_NUMBA7_sext_G25seeds123` 0.66747 / 43m01s.
  Seed-averaged sextets do not beat the best quint on public (all within the 0.00023 seed spread), but run well
  inside the limit on v7. Public top is flat at 0.66747-0.66754 across 7 compositions.
