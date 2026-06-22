"""TDD: tests for src/mamba3jp/data/dataset.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mamba3jp.data.binidx import BinIdxWriter

torch = pytest.importorskip("torch")

from mamba3jp.data.dataset import MemmapCLMDataset  # noqa: E402


def _make_reader(tmp_path: Path, n_tokens: int):
    from mamba3jp.data.binidx import BinIdxReader

    bin_path = tmp_path / "d.bin"
    idx_path = tmp_path / "d.idx"
    with BinIdxWriter(bin_path, idx_path) as w:
        w.add_document(np.arange(n_tokens, dtype=np.uint32))
    return BinIdxReader(bin_path, idx_path)


def test_length_matches_floor_div(tmp_path: Path) -> None:
    r = _make_reader(tmp_path, 1000)
    ds = MemmapCLMDataset(r, seq_len=16)
    assert len(ds) == (1000 - 1) // 16


def test_getitem_returns_input_ids_and_labels(tmp_path: Path) -> None:
    r = _make_reader(tmp_path, 200)
    ds = MemmapCLMDataset(r, seq_len=8)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "labels"}
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long
    assert item["input_ids"].shape == (8,)
    assert item["labels"].shape == (8,)


def test_labels_are_input_ids_shifted_by_one(tmp_path: Path) -> None:
    r = _make_reader(tmp_path, 200)
    ds = MemmapCLMDataset(r, seq_len=8)
    item = ds[3]
    # next-token prediction: labels[i] == input_ids[i+1]
    assert torch.equal(item["labels"][:-1], item["input_ids"][1:])


def test_consecutive_windows_dont_overlap(tmp_path: Path) -> None:
    r = _make_reader(tmp_path, 200)
    ds = MemmapCLMDataset(r, seq_len=8)
    a = ds[0]["input_ids"]
    b = ds[1]["input_ids"]
    assert a[-1].item() + 1 == b[0].item()


def test_out_of_range_raises(tmp_path: Path) -> None:
    r = _make_reader(tmp_path, 200)
    ds = MemmapCLMDataset(r, seq_len=8)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
