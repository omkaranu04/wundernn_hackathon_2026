"""Sequence dataset for experiment 028 (one full 20,000-step sequence per item).

`row_weight` is what the loss multiplies each row by:
- mask_probs_path given  -> predicted scored-probabilities (experiment 025),
  float16 (n_seq, 20000) in row-group order - used for train.parquet;
- true_mask_as_weight    -> the file's TRUE `is_scored` flag as 0/1 weights
  (only valid.parquet has it) - used when validation sequences are added to
  the training set for a final train+valid model;
- neither                -> all ones (unweighted).
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
    def __init__(self, parquet_path, mask_probs_path=None, true_mask_as_weight=False):
        self.parquet_path = parquet_path
        self.parquet = pq.ParquetFile(parquet_path)
        self.has_is_scored, self.has_targets = file_columns(self.parquet)
        if not self.has_targets:
            raise ValueError(f"{parquet_path} has no target columns")
        if true_mask_as_weight and not self.has_is_scored:
            raise ValueError(f"{parquet_path} has no is_scored column")
        self.mask_probs = np.load(mask_probs_path, mmap_mode="r") if mask_probs_path else None
        self.true_mask_as_weight = true_mask_as_weight
        if self.mask_probs is not None and self.mask_probs.shape[0] != self.parquet.num_row_groups:
            raise ValueError("mask_probs rows != number of sequences in the parquet file")

    def __len__(self) -> int:
        return self.parquet.num_row_groups

    def __getitem__(self, idx):
        seq = read_sequence(self.parquet, idx, has_is_scored=self.has_is_scored, has_targets=self.has_targets)
        need_prediction = torch.from_numpy(seq.need_prediction)
        is_scored = torch.from_numpy(seq.is_scored) if seq.is_scored is not None else torch.ones_like(need_prediction)
        if self.mask_probs is not None:
            row_weight = torch.from_numpy(np.asarray(self.mask_probs[idx], dtype=np.float32).copy())
        elif self.true_mask_as_weight:
            row_weight = is_scored.float()
        else:
            row_weight = torch.ones_like(need_prediction, dtype=torch.float32)
        return {
            "features": torch.from_numpy(seq.features),
            "targets": torch.from_numpy(seq.targets),
            "need_prediction": need_prediction,
            "is_scored": is_scored,
            "row_weight": row_weight,
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
