"""TDD: weights 対応 MemmapCLMDataset (PR-6).

文書ごとの ``final_score`` を ``weights`` ndarray として Dataset に渡し、各窓に
対してその窓の開始位置を含む文書の weight が割り当てられる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mamba3jp.data.binidx import BinIdxReader, BinIdxWriter

torch = pytest.importorskip("torch")

from mamba3jp.data.dataset import MemmapCLMDataset  # noqa: E402


def _make_reader(tmp_path: Path, doc_sizes: list[int]) -> BinIdxReader:
    bin_path = tmp_path / "d.bin"
    idx_path = tmp_path / "d.idx"
    cursor = 0
    with BinIdxWriter(bin_path, idx_path) as w:
        for sz in doc_sizes:
            w.add_document(np.arange(cursor, cursor + sz, dtype=np.uint32))
            cursor += sz
    return BinIdxReader(bin_path, idx_path)


# ---- back-compat: no weights ------------------------------------------------


def test_no_weights_yields_input_ids_and_labels_only(tmp_path: Path) -> None:
    r = _make_reader(tmp_path, [200])
    ds = MemmapCLMDataset(r, seq_len=8)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "labels"}


# ---- weights propagation ---------------------------------------------------


def test_weights_attached_per_window(tmp_path: Path) -> None:
    # 文書 0: 100 tokens, 文書 1: 100 tokens
    r = _make_reader(tmp_path, [100, 100])
    weights = np.asarray([0.3, 0.9], dtype=np.float32)
    ds = MemmapCLMDataset(r, seq_len=16, weights=weights)

    # 文書 0 範囲 (start in [0, 100)) → weight ≈ 0.3
    first = ds[0]
    assert "weight" in first
    assert isinstance(first["weight"], torch.Tensor)
    assert first["weight"].dtype == torch.float32
    assert float(first["weight"]) == pytest.approx(0.3)

    # 文書 1 範囲の窓 (start in [100, 200)) を選ぶ
    # seq_len=16 で len(ds) = (200-1)//16 = 12, doc1 が始まる index は 100//16 = 6 以降
    later = ds[7]
    assert float(later["weight"]) == pytest.approx(0.9)


def test_weight_uses_document_at_window_start(tmp_path: Path) -> None:
    """境界をまたぐ窓は開始位置の文書 weight を採用する (実装方針)。"""
    r = _make_reader(tmp_path, [9, 23])
    weights = np.asarray([0.1, 0.7], dtype=np.float32)
    ds = MemmapCLMDataset(r, seq_len=8, weights=weights)
    # 窓 0: start=0, doc=0 → weight 0.1
    assert float(ds[0]["weight"]) == pytest.approx(0.1)
    # 窓 1: start=8, doc=0 (まだ doc 0) → weight 0.1
    assert float(ds[1]["weight"]) == pytest.approx(0.1)
    # 窓 2: start=16, doc=1 → weight 0.7
    assert float(ds[2]["weight"]) == pytest.approx(0.7)


def test_weights_length_mismatch_raises(tmp_path: Path) -> None:
    r = _make_reader(tmp_path, [100, 100])
    bad = np.asarray([0.5], dtype=np.float32)  # 文書数 2 だが weights 長 1
    with pytest.raises(ValueError, match="weights"):
        MemmapCLMDataset(r, seq_len=8, weights=bad)
