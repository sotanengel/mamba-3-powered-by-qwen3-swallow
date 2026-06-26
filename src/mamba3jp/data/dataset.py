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
    where ``labels[i] = input_ids[i+1]`` (next-token prediction).

    ``weights`` を渡すと、各窓に対してその窓の開始位置を含む文書の weight を
    ``"weight"`` キーで返す (PR-6, 重み付き蒸留学習)。weights は文書単位で長さ
    ``reader.n_documents`` の float ndarray を期待する。
    """

    def __init__(
        self,
        reader: BinIdxReader,
        seq_len: int = 2048,
        weights: np.ndarray | None = None,
    ) -> None:
        if seq_len <= 1:
            raise ValueError(f"seq_len must be > 1, got {seq_len}")
        self.reader = reader
        self.seq_len = int(seq_len)
        # We read ``seq_len + 1`` tokens per window to form (input, labels),
        # so the maximum start position is total_tokens - (seq_len + 1).
        usable = reader.total_tokens - 1
        self._length = usable // self.seq_len

        self._weights: np.ndarray | None = None
        self._doc_offsets: np.ndarray | None = None
        if weights is not None:
            if weights.shape[0] != reader.n_documents:
                raise ValueError(
                    f"weights length {weights.shape[0]} != n_documents {reader.n_documents}"
                )
            self._weights = np.asarray(weights, dtype=np.float32)
            # cumsum で各文書の終端位置を求める。窓開始の np.searchsorted(side='right') で
            # その位置を含む文書 index が得られる。
            self._doc_offsets = np.cumsum(reader.doc_lengths.astype(np.int64))

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
        out: dict[str, torch.Tensor] = {"input_ids": input_ids, "labels": labels}
        if self._weights is not None and self._doc_offsets is not None:
            doc_idx = int(np.searchsorted(self._doc_offsets, start, side="right"))
            doc_idx = min(doc_idx, self._weights.shape[0] - 1)
            out["weight"] = torch.tensor(float(self._weights[doc_idx]), dtype=torch.float32)
        return out
