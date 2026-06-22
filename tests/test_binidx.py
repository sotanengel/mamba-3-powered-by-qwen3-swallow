"""TDD: tests for src/mamba3jp/data/binidx.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mamba3jp.data.binidx import BinIdxReader, BinIdxWriter


def test_roundtrip_single_document(tmp_path: Path) -> None:
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "data.idx"
    tokens = np.arange(100, dtype=np.uint32)

    with BinIdxWriter(bin_path, idx_path) as w:
        w.add_document(tokens)

    r = BinIdxReader(bin_path, idx_path)
    assert r.total_tokens == 100
    assert r.n_documents == 1
    np.testing.assert_array_equal(r[:], tokens)


def test_roundtrip_multiple_documents(tmp_path: Path) -> None:
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "data.idx"
    docs = [
        np.array([1, 2, 3, 4, 5], dtype=np.uint32),
        np.array([10, 20, 30], dtype=np.uint32),
        np.array([99, 100, 101, 102, 103, 104, 105], dtype=np.uint32),
    ]

    with BinIdxWriter(bin_path, idx_path) as w:
        for d in docs:
            w.add_document(d)

    r = BinIdxReader(bin_path, idx_path)
    assert r.total_tokens == sum(len(d) for d in docs)
    assert r.n_documents == len(docs)
    np.testing.assert_array_equal(r[:], np.concatenate(docs))
    # per-document recovery via doc_lengths
    cursor = 0
    for d in docs:
        np.testing.assert_array_equal(r[cursor : cursor + len(d)], d)
        cursor += len(d)


def test_slice_indexing_returns_expected_range(tmp_path: Path) -> None:
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "data.idx"
    tokens = np.arange(50, dtype=np.uint32) * 7

    with BinIdxWriter(bin_path, idx_path) as w:
        w.add_document(tokens)

    r = BinIdxReader(bin_path, idx_path)
    np.testing.assert_array_equal(r[10:20], tokens[10:20])
    np.testing.assert_array_equal(r[:5], tokens[:5])
    np.testing.assert_array_equal(r[-5:], tokens[-5:])


def test_integer_indexing_returns_scalar(tmp_path: Path) -> None:
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "data.idx"
    tokens = np.array([10, 20, 30, 40], dtype=np.uint32)

    with BinIdxWriter(bin_path, idx_path) as w:
        w.add_document(tokens)

    r = BinIdxReader(bin_path, idx_path)
    assert int(r[2]) == 30


def test_writer_rejects_wrong_dtype(tmp_path: Path) -> None:
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "data.idx"
    with BinIdxWriter(bin_path, idx_path) as w, pytest.raises(ValueError):
        w.add_document(np.array([1, 2, 3], dtype=np.int16))


def test_reader_rejects_corrupt_magic(tmp_path: Path) -> None:
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "data.idx"
    bin_path.write_bytes(b"")
    idx_path.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError):
        BinIdxReader(bin_path, idx_path)
