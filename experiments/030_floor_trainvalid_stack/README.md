# 030: stack the two 029 wins (softer weights x train+valid) and sweep the floor

Follow-up to 029's platform results (2026-10-12): D (028-B's recipe on
train+valid) **0.6596** and floor10 (row weights 0.1+0.9p) **0.6588**, both
above every 028 run (0.6574-0.6583). Each of the two changes was worth about
+0.0013 on the platform on its own:

| change | pair | platform gain | local gain |
|---|---|---|---|
| floor 0.1 on the scored-probability weights | floor10 vs 028-A | +0.0014 | +0.0011 (EMA) |
| train on train+valid | D vs 028-B, C vs A | +0.0013, +0.0006 | not measurable |

## Code

No code changes. `train.py`, `model.py`, `dataset.py`, `eval_wp.py`,
`export_onnx.py`, `solution.py` are 029's (only train.py's docstring differs).
Lean model, 248,510 params, 46 ONNX nodes, ~41 min on the platform.
`queue_job.sh` / `launch_all.sh` queue the arms on server A and server B (shared
home; an atomic lock makes each arm start once).

Common to all arms: v1 mask labels (`025.../mask_out/mask_probs_train.npy`;
029's `A_v2labels` showed sharper labels do not help), EMA decay 0.9998,
standard loss 0.5 MSE + 0.5 per-sequence Pearson, Adam 1e-3, batch 4.
Row weight = floor + (1 - floor) * p, applied to train's predicted
probabilities and, in the train+valid arms, to valid's TRUE 0/1 mask.

## Arms

| arm | flags on top of the common ones | question | local score? |
|---|---|---|---|
| `E_floor10_aux_tv` | `--weight-floor 0.1 --aux-weight 0.2 --extra-train valid --no-validation --epochs 14 --lr-milestones 10` | Do the two wins add up? (D with floor 0.1) | no - platform only |
| `E_floor10_aux_tv_seed1` | same, `--seed 1` | How large is seed noise on the platform for this recipe? Also a second candidate. | no - platform only |
| `floor20` | `--weight-floor 0.2` | Where does the floor stop helping? | yes |
| `floor30` | `--weight-floor 0.3` | | yes |
| `floor50` | `--weight-floor 0.5` | | yes |

Local reference points for the sweep (best EMA WP, batched): floor 0 =
028-A 0.686746; floor 0.1 = 029 floor10 0.687892; floor 1.0 would be the
unweighted lean model (027 control 0.683450 raw, no EMA run exists);
p^2 weights (sharper than floor 0) = 0.684285.

## Status

All five arms started 2026-10-12 02:07-02:20 on server A (server B had no free GPU
memory). `E_floor10_aux_tv`, `E_floor10_aux_tv_seed1`, `floor30` run on GPU 1
(~1.7 batch/s each); `floor20` and `floor50` were restarted by hand on GPU 0
(~1.4 batch/s each) because all five on one GPU ran at 1.05 batch/s.
Expected: the two E runs ~7 h (around 09:30), the sweep ~10-11 h with early
stopping. Logs on the shared server home: `logs/train_<arm>.log`. Results: none yet.

## Results so far (2026-10-12 10:15)

**Both E runs finished** (fixed 14 epochs, ~7 h each). No local score is
possible. Sanity check only (batched WP on 61 valid sequences, every 31st;
INFLATED for train+valid models because valid is in their training data):
floor10 (never saw valid) 0.7229; D 0.7406; E seed 0 0.7408; E seed 1 0.7378.
So E is healthy and behaves like D.

| zip (`submissions/`) | weights | export vs batched forward |
|---|---|---|
| `030_E_floor10_aux_trainvalid_ema_2026-10-12.zip` | seed 0 `ema.pt` (final, epoch 13) | 9.24e-07 |
| `030_E_floor10_aux_trainvalid_seed1_ema_2026-10-12.zip` | seed 1 `ema.pt` (final, epoch 13) | 5.36e-07 |

Both 1.85 MB, 46 nodes, outputs `[prediction, next_h, next_c]`, same graph
and `solution.py` as the 028/029 zips (~41 min on the platform),
fresh-extraction checks passed. Weights and logs in `results/` and `logs/`
(copied from the server, 8 checksums verified, deleted there).

**Floor sweep, still training** (best EMA WP, batched; reference floor 0 =
0.686746, floor 0.1 = 0.687892):

| arm | epoch reached | best EMA so far | same-epoch comparison |
|---|---|---|---|
| `floor30` | 17 (LR already halved, EMA falling) | 0.687487 (epoch 13) | floor10 at epoch 13: 0.687892 |
| `floor20` | 11 | 0.685984 (epoch 10) | floor10 at epoch 10: 0.686511; 028-A: 0.686541 |
| `floor50` | 11 | 0.685729 (epoch 10) | |

Reading so far: no floor above 0.1 is ahead of 0.1; 0.3 will end ~0.0004
below it and ~0.0007 above floor 0. All within single-seed noise - the floor
is a shallow optimum somewhere in 0.1-0.3, not something to tune further.

## Platform results (submitted 2026-10-12) - new best

| zip | platform | runtime |
|---|---|---|
| E seed 0 | **0.6605** (ID 15407ND0) - best so far | 41m2s, 627 MB |
| E seed 1 | 0.6592 (ID 151LH3KD) | 41m27s, 627 MB |

- **First direct measurement of seed noise on the platform: 0.0013** between
  two runs that differ only in the seed. That is the same size as every
  single-change gain since 028 (+0.0006 to +0.0014), so no single pair of
  submissions resolves a change; only repeated agreement does.
- Mean of the two E runs is 0.65985, vs D (same recipe without the floor, one
  seed) 0.6596: the floor's effect on top of train+valid is +0.0003 +- seed
  noise, i.e. UNRESOLVED. (Without train+valid it was +0.0014 platform /
  +0.0011 local, also one seed each.)
- What IS resolved, by groups rather than pairs: the four train+valid runs
  (C 0.6580, D 0.6596, E 0.6605, 0.6592; mean 0.6593) vs the three
  train-only runs (A 0.6574, B 0.6583, floor10 0.6588; mean 0.6582), and all
  seven (0.6574-0.6605) vs the same lean model without weights/EMA (0.6555).

## Floor sweep (finished / running)

`floor30` finished 2026-10-12 (early stop, epoch 19): raw 0.685748 / EMA
0.687487. `floor20`, `floor50` at epoch 12: EMA 0.686345 / 0.686324.
Floor 0 -> 0.1 -> 0.3 gives 0.686746 -> 0.687892 -> 0.687487: flat within noise.

## Second batch (started 2026-10-12 11:03 on server A GPU 1), all with validation

Every run so far peaks around epoch 11-13 and then DECLINES once the LR is
halved (valid WP falls while train loss keeps falling), and the wider h160
model did not help - both point at overfitting, not capacity. Dropout has never
been varied (0.1 everywhere; it sits after each group encoder and inside the
head, not between LSTM layers).

| arm | flags (on floor 0.1 + EMA) | question |
|---|---|---|
| `drop20` | `--dropout 0.2` | Does more regularization raise the peak / delay the decline? |
| `drop30` | `--dropout 0.3` | |
| `floor10_seed1` | `--seed 1` | LOCAL seed noise for the reference run (seed 0: raw 0.686718 / EMA 0.687892) |

`drop30` ran out of GPU memory at start-up (another user took the memory) and was requeued
through `queue_job.sh` on server A and server B; `drop20` and `floor10_seed1` run on server A GPU 1.

## Floor sweep: final (2026-10-12 16:20)

| floor | run | best raw WP | best EMA WP |
|---|---|---|---|
| 0 | 028-A | 0.685089 | 0.686746 |
| 0.1 | 029 floor10 | 0.686718 | **0.687892** |
| 0.2 | floor20 | 0.686326 | 0.686662 |
| 0.3 | floor30 | 0.685748 | 0.687487 |
| 0.5 | floor50 | 0.684899 | 0.686405 |

No trend: the five values span 0.0015, the size of seed noise, and are not
monotone (0.2 is below both 0.1 and 0.3). The floor is not worth tuning;
0.1 stays the default. Weights and logs in `results/` and `logs/` (18
checksums verified, deleted from the server).

Second batch at 16:20: `drop30` at epoch 13, best EMA 0.686263 (floor10 seed 0
had 0.687669 at epoch 12) - behind so far; `drop20` and `floor10_seed1` only at
epoch 3 because server A is carrying the 031 teachers as well.

`drop30` finished 2026-10-12 18:40 (early stop, epoch 18): raw 0.685390 / EMA 0.686418, i.e. -0.0015 vs
floor10 seed 0 (0.687892). More dropout does not help at 0.3; `drop20` still running (epoch 6).

## Second batch: final (2026-10-13)

| arm | best raw WP | best EMA WP | stopped |
|---|---|---|---|
| `floor10_seed1` | 0.684685 | 0.686549 | epoch 14 |
| `drop20` (dropout 0.2) | 0.684964 | 0.686429 | epoch 18 |
| `drop30` (dropout 0.3) | 0.685390 | 0.686418 | epoch 18 |
| reference: 029 floor10 seed 0 | 0.686718 | 0.687892 | epoch 18 |

The second seed of the reference itself lands at 0.686549, 0.0013 below seed
0 - so seed 0 was a lucky draw and both dropout arms are indistinguishable
from the reference recipe. (031's ten clean seeds agree: 0.6855-0.6873.)
Dropout 0.2 / 0.3: no effect. Closed. All weights and logs are local
(12 checksums verified, deleted from the server); nothing of 030 is left there.
