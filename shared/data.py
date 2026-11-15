"""Shared data-loading utilities for reading the competition's sequence Parquet files.

Re-exports the column/constant definitions from the organizer-provided
`wnn_connectome_starterpack/utils.py` so every experiment agrees on the same
feature order, rather than redefining it per experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq

_STARTERPACK = Path(__file__).resolve().parent.parent / "wnn_connectome_starterpack"
if str(_STARTERPACK) not in sys.path:
    sys.path.insert(0, str(_STARTERPACK))

from utils import (  # noqa: E402  (import after sys.path fixup, intentional)
    DataPoint,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    SEQUENCE_LENGTH,
    WARMUP,
    N_FEATURES,
)

__all__ = [
    "DataPoint", "FEATURE_COLUMNS", "TARGET_COLUMNS", "SEQUENCE_LENGTH", "WARMUP", "N_FEATURES",
    "SequenceRecord", "iter_sequences", "read_sequence", "file_columns",
]


class SequenceRecord:
    """One complete 20,000-row sequence, loaded into memory as plain numpy arrays."""

    __slots__ = ("seq_ix", "features", "targets", "need_prediction", "is_scored")

    def __init__(self, seq_ix, features, targets, need_prediction, is_scored):
        self.seq_ix = seq_ix
        self.features = features                  # float32 (SEQUENCE_LENGTH, N_FEATURES)
        self.targets = targets                    # float32 (SEQUENCE_LENGTH, 2) or None if unlabeled
        self.need_prediction = need_prediction     # bool (SEQUENCE_LENGTH,)
        self.is_scored = is_scored                 # bool (SEQUENCE_LENGTH,) or None if absent


def file_columns(parquet_file: pq.ParquetFile) -> tuple[bool, bool]:
    """Returns (has_is_scored, has_targets) for a given open ParquetFile."""
    columns = parquet_file.schema_arrow.names
    return "is_scored" in columns, TARGET_COLUMNS[0] in columns


def read_sequence(
    parquet_file: pq.ParquetFile, group: int, *, has_is_scored: bool, has_targets: bool
) -> SequenceRecord:
    """Read and convert a single row group (= one sequence) into a `SequenceRecord`.

    This is the single place that knows how to turn a raw Arrow row-group
    table into numpy arrays - both `iter_sequences` (eager, whole-file
    iteration) and any lazy per-item dataset (e.g. `ConnectomeSequenceDataset`
    in experiments/001_lstm_attention/dataset.py, which calls this directly
    per `__getitem__` instead of materializing the whole file) go through it,
    so there is exactly one implementation of the column layout to keep in
    sync with `FEATURE_COLUMNS`/`TARGET_COLUMNS`.

    `use_threads=True`: measured ~40% faster raw read per row group (114
    columns decoded in parallel) vs. the previous `use_threads=False`,
    on fresh/uncached row groups of train.parquet - ~11-14ms/seq vs.
    ~18-25ms/seq. Every experiment's DataLoader uses num_workers=0
    (synchronous), so this read time is pure overhead on top of GPU
    compute, not hidden by prefetching - a real, if modest (~3-6% of
    total per-batch time), training speedup. Does not affect the
    organizer-provided `ScorerStepByStep` (wnn_connectome_starterpack/
    utils.py, left untouched per this project's read-only convention for
    that file) - there the parquet read is <2% of total per-sequence
    scoring time, dwarfed by the 20,000-step per-row inference loop, so
    the same change wouldn't move that bottleneck at all.
    """
    table = parquet_file.read_row_group(group, use_threads=True)
    seq_ix = table["seq_ix"][0].as_py()
    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    need_prediction = table["need_prediction"].to_numpy().astype(bool)
    targets = (np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
               if has_targets else None)
    is_scored = table["is_scored"].to_numpy().astype(bool) if has_is_scored else None
    return SequenceRecord(seq_ix, features, targets, need_prediction, is_scored)


def iter_sequences(parquet_path, *, has_is_scored: Optional[bool] = None) -> Iterator[SequenceRecord]:
    """Yield one `SequenceRecord` per Parquet row group, reading them one at a
    time (not loading the whole file into memory at once).

    NOTE: an eager *caller* that does `list(iter_sequences(path))` (as
    `dataset.py` used to) still ends up holding the whole file in memory -
    see `read_sequence` if you need genuinely lazy, on-demand, random-access
    per-sequence reads (e.g. for a `torch.utils.data.Dataset.__getitem__`).
    """
    parquet_file = pq.ParquetFile(parquet_path)
    has_is_scored_detected, has_targets = file_columns(parquet_file)
    if has_is_scored is None:
        has_is_scored = has_is_scored_detected

    for group in range(parquet_file.num_row_groups):
        yield read_sequence(parquet_file, group, has_is_scored=has_is_scored, has_targets=has_targets)
