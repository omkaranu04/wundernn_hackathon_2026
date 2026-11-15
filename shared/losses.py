"""Differentiable training losses.

Includes a differentiable proxy for the competition's Weighted Pearson (WP)
metric, so training can optimize something closer to what's actually scored
than plain MSE (WP is a *correlation*, weighted by |clipped target| - a model
that minimizes MSE is not directly minimizing 1 - WP).
"""
from __future__ import annotations

from typing import Optional

import torch

METRIC_CLIP = 2.0
EPS = 1e-8


def weighted_pearson_loss(
    target: torch.Tensor, prediction: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Negative weighted Pearson correlation, batched over the last dimension.

    Mirrors `wnn_connectome_starterpack/utils.py::weighted_pearson` (same clip,
    same |clipped target| weighting) but stays differentiable so it can be used
    directly as a training loss. `target`/`prediction` should be shaped
    (..., N) where N is the axis correlation is computed over (typically "all
    scored steps of one sequence, for one target") - any leading dims (e.g. a
    batch of sequences, or the two target indices) are kept and averaged at
    the end.

    AGGREGATION CAVEAT (metric corrected 2026-09-15/16): the per-sequence
    usage in every experiment's `train.py` (one correlation per sequence,
    then averaged with equal weight per sequence) matches the OLD,
    retired metric's aggregation. The corrected "Global WP" metric instead
    pools all scored rows across all sequences into one set of moments per
    target and computes a single correlation. Clip (+-2), the |clipped
    target| weights and the per-target/mean-of-two-targets structure are
    unchanged and still match. The practical difference: a per-sequence
    correlation cancels each sequence's own mean and scale, so it gives no
    gradient for getting a sequence's overall level/volatility right,
    whereas the global metric rewards that (between-sequence differences
    enter the pooled covariance). The MSE term, which pools rows, partly
    covers this. Every experiment through 018 was trained with the
    per-sequence form; its effect on global WP has NOT been measured. To
    approximate the global metric during training, pass the whole batch's
    scored rows flattened into a single N axis (one correlation per target)
    instead of one correlation per sequence.
    """
    y = target.clamp(-METRIC_CLIP, METRIC_CLIP)
    p = prediction.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = weight if weight is not None else y.abs()

    total = w.sum(dim=-1, keepdim=True).clamp_min(EPS)
    y_mean = (w * y).sum(dim=-1, keepdim=True) / total
    p_mean = (w * p).sum(dim=-1, keepdim=True) / total
    y_c = y - y_mean
    p_c = p - p_mean

    total_sq = total.squeeze(-1)
    cov = (w * y_c * p_c).sum(dim=-1) / total_sq
    y_std = ((w * y_c**2).sum(dim=-1) / total_sq).clamp_min(EPS).sqrt()
    p_std = ((w * p_c**2).sum(dim=-1) / total_sq).clamp_min(EPS).sqrt()

    corr = cov / (y_std * p_std)
    return -corr.mean()


def weighted_mse_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Magnitude-weighted MSE, using the same |clipped target| weighting as WP.

    Cheaper and more stable early in training than the correlation loss (which
    has a degenerate/unstable gradient when a batch's weighted variance is
    near zero); the two are meant to be combined (see `train.py`'s `--loss`
    flag), e.g. warm up with MSE then anneal towards the correlation loss.
    """
    y = target.clamp(-METRIC_CLIP, METRIC_CLIP)
    p = prediction.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs()
    return (w * (y - p) ** 2).sum() / w.sum().clamp_min(EPS)
