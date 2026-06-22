"""Causal-LM dataset backed by a :class:`BinIdxReader`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from mamba3jp.data.binidx import BinIdxReader


class MemmapCLMDataset(Dataset[dict[str, torch.Tensor]]):
    """Non-overlapping sliding window of length ``seq_len`` over a binidx corpus.

    Each item returns ``{"input_ids": Long[seq_len], "labels": Long[seq_len]}``
    where ``labels[i] = input_ids[i+1]`` (next-token prediction). The final
    label position is set to ``-100`` so it can be safely ignored by the loss.
    """

    def __init__(self, reader: BinIdxReader, seq_len: int = 2048) -> None:
        if seq_len <= 1:
            raise ValueError(f"seq_len must be > 1, got {seq_len}")
        self.reader = reader
        self.seq_len = int(seq_len)
        # We read ``seq_len + 1`` tokens per window to form (input, labels),
        # so the maximum start position is total_tokens - (seq_len + 1).
        usable = reader.total_tokens - 1
        self._length = usable // self.seq_len

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = np.asarray(self.reader[start:end], dtype=np.int64)
        input_ids = torch.from_numpy(chunk[:-1]).long()
        labels = torch.from_numpy(chunk[1:]).long()
        return {"input_ids": input_ids, "labels": labels}
