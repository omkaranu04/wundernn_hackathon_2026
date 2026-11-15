"""Extract per-epoch train/valid loss from each experiment's training log
and plot a loss curve, saved into that experiment's own directory as
`loss_curve.png`. Reads log text directly (tqdm's carriage-return output
means the "epoch N: train_loss=... valid_loss=..." lines aren't on their
own clean lines - extracted via regex over raw file bytes, not line-by-line
parsing).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"

# (experiment dir name, log file relative to that dir, display title)
EXPERIMENTS = [
    ("001_lstm_attention", "logs/full_train_20260912_003550.log", "001 - LSTM + causal windowed attention"),
    ("002_stacked_lstm", "logs/full_train_20260912_144030.log", "002 - Stacked LSTM (4 layers)"),
    ("003_lstm_mlp_head", "logs/full_train_20260913_132711.log", "003 - Stacked LSTM (2 layers) + MLP head"),
    ("004_grouped_encoder_lstm", "logs/full_train_20260913_141326.log", "004 - Grouped-encoder LSTM (i0+i1 merged)"),
    ("005_lstm3_mlp_head", "logs/full_train_20260914_005058.log", "005 - Stacked LSTM (3 layers) + MLP head"),
    ("006_wide_lstm_sinh", "logs/train.log", "006 - Wide single LSTM + sinh head"),
    ("007_stacked_lstm4_sinh", "logs/train.log", "007 - Stacked LSTM (4 layers) + sinh head"),
    ("008_narrow_deep_lstm", "logs/train.log", "008 - Narrow-and-deep LSTM (h=32, 6 layers)"),
    ("009_split_grouped_encoder_lstm", "logs/train.log", "009 - Split grouped-encoder LSTM (i0/i1 separate)"),
]

# Extended (15-epoch, LR-decay + early-stopping, GPU) re-training runs -
# same experiment directories, different log file and output filename so
# the original 5-epoch loss_curve.png is never overwritten.
EXTENDED_RUNS = [
    ("002_stacked_lstm", "logs/train_extended.log", "002 (extended, 15ep) - Stacked LSTM (4 layers)"),
    ("004_grouped_encoder_lstm", "logs/train_extended.log", "004 (extended, 15ep) - Grouped-encoder LSTM"),
    ("006_wide_lstm_sinh", "logs/train_extended.log", "006 (extended, 15ep) - Wide single LSTM + sinh head"),
]

# Further-extended (30-epoch, warmup + LR-decay + early-stopping, GPU)
# re-training run(s) - same convention, another distinct log/output name.
THIRTY_EPOCH_RUNS = [
    ("004_grouped_encoder_lstm", "logs/train_30ep.log", "004 (30ep, warmup+decay) - Grouped-encoder LSTM"),
    ("010_sum_reparam_head", "logs/train.log", "010 (30ep, warmup+decay) - Sum-reparam head (t0, t0+t1)"),
    ("011_xgboost_head", "logs/train_backbone.log", "011 backbone (30ep, warmup+decay) - Encoder+2L LSTM (for XGBoost head)"),
    ("012_nonlinear_encoder", "logs/train.log", "012 (30ep, warmup+decay) - Price asinh-expanded grouped-encoder LSTM"),
    ("013_rolling_norm_encoder", "logs/train.log", "013 (30ep, warmup+decay) - Rolling EWMA-normalized grouped-encoder LSTM"),
    ("013_rolling_norm_encoder", "logs/train_priceonly.log", "013 (price-only + 2 windows, 30ep, warmup+decay) - latency-fixed retrain"),
    ("018_nonlinear_rolling_norm_encoder", "logs/train.log", "018 (30ep, warmup+decay) - 012 nonlinear expansion + 013 price-only EWMA (2 windows)"),
    ("017_nonlinear_sum_reparam", "logs/train.log", "017 (30ep, warmup+decay) - 012's encoder + sum-reparam head (t0, t0+t1)"),
    ("001_lstm_attention", "logs/train_full.log", "001 (full-scale, 30ep, warmup+decay) - d_model=128, 2-layer LSTM + windowed attention"),
    ("001_lstm_attention", "logs/train_d32.log", "001 (d_model=32, 30ep, warmup+decay) - original safe config, modern recipe"),
    ("015_nonlinear_windowed_attention", "logs/train.log", "015 (30ep, warmup+decay) - 012's encoder + 1-layer LSTM + windowed attention (512)"),
    ("016_nonlinear_sparse_attention", "logs/train.log", "016 (30ep, warmup+decay) - 012's encoder + 2-layer LSTM + dilated sparse attention"),
    ("014_sparse_attention_lstm", "logs/train.log", "014 (30ep, warmup+decay) - 001's architecture at full scale (d_model=128) + dilated sparse attention"),
]

EPOCH_RE = re.compile(r"epoch (\d+): train_loss=([\d.]+) valid_loss=([\d.]+)(?: lr=([\d.e+-]+))?")


def extract_losses(log_path: Path):
    text = log_path.read_text(errors="replace")
    matches = EPOCH_RE.findall(text)
    epochs = [int(m[0]) for m in matches]
    train_losses = [float(m[1]) for m in matches]
    valid_losses = [float(m[2]) for m in matches]
    lrs = [float(m[3]) if m[3] else None for m in matches]
    return epochs, train_losses, valid_losses, lrs


def plot_one(exp_dir_name: str, log_rel_path: str, title: str, out_name: str = "loss_curve.png") -> Path:
    exp_dir = EXPERIMENTS_DIR / exp_dir_name
    log_path = exp_dir / log_rel_path
    epochs, train_losses, valid_losses, lrs = extract_losses(log_path)
    if not epochs:
        raise ValueError(f"no epoch loss lines found in {log_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, marker="o", label="train_loss")
    ax.plot(epochs, valid_losses, marker="o", label="valid_loss")

    # Mark epochs where the LR scheduler stepped down (extended runs only).
    for i in range(1, len(lrs)):
        if lrs[i] is not None and lrs[i - 1] is not None and lrs[i] < lrs[i - 1]:
            ax.axvline(epochs[i], color="gray", linestyle="--", alpha=0.6, linewidth=1)
            ax.annotate(f"lr->{lrs[i]:.1e}", (epochs[i], max(train_losses)),
                        rotation=90, fontsize=7, va="top", ha="right", color="gray")

    best_idx = min(range(len(valid_losses)), key=lambda i: valid_losses[i])
    ax.scatter([epochs[best_idx]], [valid_losses[best_idx]], color="green", zorder=5,
               label=f"best valid_loss ({valid_losses[best_idx]:.4f})")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (mse+pearson, weighted)")
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = exp_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    for exp_dir_name, log_rel_path, title in EXPERIMENTS:
        out_path = plot_one(exp_dir_name, log_rel_path, title)
        print(f"{exp_dir_name}: saved {out_path}")


if __name__ == "__main__":
    main()
