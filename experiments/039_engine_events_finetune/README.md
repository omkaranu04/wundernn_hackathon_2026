# 039: numba inference engine, event-time teacher, scored-row fine-tune (2026-11-07)

From the user's request after the web search (sunArise log, 10-24): (1) test a faster inference engine so the
ensembles can grow; (2) act on "t0 = F - ask, t1 = bid - F, t0 + t1 = -spread, F settles within ~8-16 quote
changes"; (3) try the log's scored-row fine-tune stage. Code in `experiments/031_seed_ensemble_distillation/`.

## 1. Numba engine (`numba_engine2.py`, `solution_numba.py`, `make_zip_numba.py`)

- One compiled call per member per row; members of any mix (full_lstm / full_gru / lean / lean_gru), weighted.
- Recurrent + head matrices stored transposed as **int16 with a per-output fp32 scale** (fp32 products are
  weight-bandwidth bound: 512x260 gemv 14.4 us fp32 vs 9.4 us int16, rel. error 3e-5; int8 8.6 us but error 1e-2).
- tanh / sigmoid via Eigen's rational approximation (numba has no SVML here); 4 inputs per pass over the outputs.
- Numba cannot type float16 (`NotImplementedError: float16`), hence int16 rather than the log's IEEE half.
- `solution_numba.py` compiles in `__init__` and falls back to an IOBinding ONNX graph if numba cannot be imported.
- Exactness: numba vs torch max diff 8e-5 (pair) / 4e-5 (triple) on 2 real valid sequences; the shipped int8 ONNX
  pair itself differs from torch by up to 7e-2.
- Local full predict loop (12k real rows, 4 sequences, pinned core, alternating):
  IOBinding ONNX pair ~102 us/row; **numba pair ~74 us/row (-27%)**; **numba triple ~107 us/row**.
  Scaling from the 0.66658 zip (130 us local -> 50m47s): numba pair ~29 min, triple ~42-46 min.
- Zips: `039_NUMBA_pair_fullLSTM_LA12_leanGRU_LA50` (same members as 0.66658: platform check of numba + runtime),
  `039_NUMBA_triple_fullLSTM_LA12_leanGRU_LA50_leanLSTM_LA25` (+ R6_LA25_A, equal weights; fallback = the ONNX pair).

## 2. Target structure (sunArise findings)

- Checked on valid: in the 0.6569 unit t0 + t1 is -1 tick 71-76% / -2 ticks 19-26% of rows (= -spread); a linear
  fit from the 22 i0 price columns explains it on UNSCORED rows (R^2 0.319) but not on SCORED rows (0.038): the
  spread is hidden exactly where the metric looks. Post-hoc output mixing was already ~+0.0001 (debate F5): no lever.
- Lever taken: **event-time look-ahead teacher** `EventLookaheadModel` (`--arch ela6`): sees the encoded book at
  each of the next 6 i0-price-change rows (change vs row t + row gap), events > 50 rows ahead ignored -> ~12 rows on
  fine units, up to 50 on the coarse unit; stays below the 8-16 changes F needs to settle. Window test: changing row
  130 affects rows 80+ only. `run_ela.sh` on server B GPU 0 (teacher ~15:00, targets ~16:00, student
  `R25_FULL_LSTM_ELA6` ~23:00); `build_039.sh` packages single + mixpair with the lean GRU LA50.

## 3. Scored-row fine-tune (`run_ft_local.sh`, local 1080 Ti)

From the best members' EMA weights: valid only, rows weighted by the TRUE mask (scored 1, unscored 0.1), labels
0.2 y + 0.8 soft (`--blend-targets --distill-weight 0.8`), AdamW lr 1e-4 wd 1e-4, clip 1.0, 6 epochs, no warmup,
batch 16 (32 does not fit the 1080 Ti). `FT_FULL_LSTM_LA12`, `FT_GRU_LA50` (~7 min each). Predictions vs the base:
corr 0.995 (LSTM) / 0.992 (GRU), scale x1.025 / x1.057. Zips: `039_FT_mixpair_fullLSTM_LA12ft_leanGRU_LA50`
(LSTM fine-tuned only) and `039_FT_pair_fullLSTM_LA12ft_GRU_LA50ft` (both).

## Platform result (2026-11-07)

| zip | platform | runtime |
|---|---|---|
| `039_NUMBA_triple_fullLSTM_LA12_leanGRU_LA50_leanLSTM_LA25` | **0.66676 (NEW BEST, +0.00018)** | **41m21s** (705 MB) |

The numba engine runs on the platform, and FASTER than projected: 107 us/row locally -> 63 us/row there (x0.59, the
same local->platform ratio as the ONNX zips: 130 us -> 77 us). A third member at 9.5 minutes LESS than the ONNX pair.

`039_mixpair_fullLSTM_ELA6_leanGRU_LA50` (event-time teacher student + lean GRU LA50): **0.66639** (50m02s), -0.00019 vs
the same pair with LA12 labels (0.66658). The far stronger teacher (held-out 0.793 vs 0.741) did NOT give a better
student - the third time a stronger look-ahead teacher failed to transfer (LA50, FLA12, ELA6). Closed.
