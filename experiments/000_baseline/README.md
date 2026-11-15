# 000 — Organizer baseline

Reference point, not our work. Kept here unmodified for comparison so later experiments have a fixed target to beat.

- **Approach:** 2-layer stacked vanilla GRU (hidden size 128 each), linear 128→2 regression head, exported to ONNX, run one timestep at a time with hidden state threaded manually between calls.
- **Local WP:** 0.589595 (full `valid.parquet`, via `utils.ScorerStepByStep`).
- **Full writeup:** [../../BASELINE.md](../../BASELINE.md)
- **Source:** `wnn_connectome_starterpack/baseline/` (`solution.py`, `baseline.onnx`)

No changes made here — this folder just documents the starting point for the log in [../../EXPERIMENTS.md](../../EXPERIMENTS.md).
