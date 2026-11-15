"""Training/validation dataset: one full 20,000-step sequence per sample, read
lazily (one Parquet row group per `__getitem__`) - same as 004's dataset.py,
plus an `is_scored` mask.

`is_scored` is needed because the corrected (global) WP metric scores only
rows where `need_prediction AND is_scored` (see PROBLEM_STATEMENT.md section
3); 004's dataset only carried `need_prediction`. train.parquet has no
`is_scored` column, so it's filled with True there (training losses use
`need_prediction` alone, same as every earlier experiment).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from data import file_columns, read_sequence  # noqa: E402


class ConnectomeSequenceDataset(Dataset):
    def __init__(self, parquet_path, mask_probs_path=None):
        """mask_probs_path: float16 (n_seq, 20000) array of scored-probabilities in
        row-group order (experiment 025). If given, __getitem__ returns them as
        `row_weight`; otherwise `row_weight` is all ones (unweighted control)."""
        self.mask_probs = np.load(mask_probs_path, mmap_mode="r") if mask_probs_path else None
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
        need_prediction = torch.from_numpy(seq.need_prediction)
        is_scored = torch.from_numpy(seq.is_scored) if seq.is_scored is not None else torch.ones_like(need_prediction)
        if self.mask_probs is not None:
            row_weight = torch.from_numpy(np.asarray(self.mask_probs[idx], dtype=np.float32).copy())
        else:
            row_weight = torch.ones_like(need_prediction, dtype=torch.float32)
        return {
            "row_weight": row_weight,
            "features": torch.from_numpy(seq.features),
            "targets": torch.from_numpy(seq.targets),
            "need_prediction": need_prediction,
            "is_scored": is_scored,
        }


def collate(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "need_prediction": torch.stack([b["need_prediction"] for b in batch]),
        "is_scored": torch.stack([b["is_scored"] for b in batch]),
        "row_weight": torch.stack([b["row_weight"] for b in batch]),
    }
