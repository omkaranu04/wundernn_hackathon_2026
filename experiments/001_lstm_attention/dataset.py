"""Training dataset: one full 20,000-step sequence per sample, read lazily
(one Parquet row group per `__getitem__`) rather than loaded eagerly.

Two independent design choices bundled here, both driven by memory:

1. **One sample = one full sequence, not a chunk.** See `model.py`'s
   `CausalSelfAttention`: windowed causal attention looks back up to
   `attention_window` steps *measured from the true start of the sequence*.
   Chunking would need that look-back carried across chunk boundaries (a
   growing key/value cache, like `inference.py`'s streaming predictor uses)
   to match inference exactly; full sequences sidestep that mismatch.

2. **Lazy, on-demand reads, not an eager `list(iter_sequences(...))`.** Each
   sequence is 20,000 x 112 float32 =~ 8.96 MB; `valid.parquet` alone is
   1,873 of them (~15.6 GiB) and `train.parquet` is 10,607 (~88.5 GiB) -
   loading either fully into a Python list, let alone both at once for
   train+valid, does not fit in RAM on a shared machine (this crashed with
   an OOM kill during the first smoke-test run, using just two eager copies
   of the smaller `valid.parquet`). Each `__getitem__` call instead opens
   the row group it needs on demand via `shared.data.read_sequence`, so
   resident memory stays roughly O(batch_size), not O(dataset size). The OS
   page cache will keep frequently-hit row groups warm across epochs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from data import file_columns, read_sequence  # noqa: E402


class ConnectomeSequenceDataset(Dataset):
    """Lazily reads sequences from a labeled Parquet file (train/valid) by row-group index.

    NOTE: not safe to share across multiple `DataLoader` worker processes as
    written (each worker would need its own `pq.ParquetFile` handle rather
    than inheriting this one). Fine as long as `DataLoader(..., num_workers=0)`
    (the default, and what `train.py` currently uses) - revisit if that changes.
    """

    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.parquet = pq.ParquetFile(parquet_path)
        self.has_is_scored, self.has_targets = file_columns(self.parquet)
        if not self.has_targets:
            raise ValueError(
                f"{parquet_path} has no target columns - use a labeled file "
                f"(train.parquet / valid.parquet), not test-format data"
            )

    def __len__(self) -> int:
        return self.parquet.num_row_groups

    def __getitem__(self, idx):
        seq = read_sequence(self.parquet, idx, has_is_scored=self.has_is_scored, has_targets=self.has_targets)
        return {
            "features": torch.from_numpy(seq.features),                 # (SEQUENCE_LENGTH, 112) float32
            "targets": torch.from_numpy(seq.targets),                   # (SEQUENCE_LENGTH, 2) float32
            "need_prediction": torch.from_numpy(seq.need_prediction),   # (SEQUENCE_LENGTH,) bool
        }


def collate(batch):
    """Every sequence already has the fixed SEQUENCE_LENGTH, so plain stacking
    is enough - no padding/sorting-by-length machinery needed."""
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "need_prediction": torch.stack([b["need_prediction"] for b in batch]),
    }
