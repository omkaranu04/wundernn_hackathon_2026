"""Training dataset: one full 20,000-step sequence per sample, read lazily
(one Parquet row group per `__getitem__`) - identical to experiment 004's
dataset.py (see 002's docstring for the OOM story that motivated lazy
loading). The nonlinear feature expansion lives in model.py's `encode()`,
not here - this dataset yields the same raw 112-dim features as every
other experiment.
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
            "features": torch.from_numpy(seq.features),
            "targets": torch.from_numpy(seq.targets),
            "need_prediction": torch.from_numpy(seq.need_prediction),
        }


def collate(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "need_prediction": torch.stack([b["need_prediction"] for b in batch]),
    }
