"""TDD: tests for src/mamba3jp/data/ingest.py.

These tests are written *before* the implementation. ``pytest tests/test_ingest.py``
must fail until ``record_to_chatml`` and ``is_low_quality`` exist.

Coverage (mirrors the 6 records in tests/data/sample_joryu.jsonl):
1. thinking + polite     → ChatML with <think> block
2. nothinking + casual   → ChatML without <think> block
3. thinking + expert     → ChatML with <think> block
4. empty answer          → skipped (returns None)
5. unclosed <think> tag  → skipped
6. repeated characters   → skipped
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamba3jp.data.ingest import (
    IM_END,
    IM_START,
    THINK_CLOSE,
    THINK_OPEN,
    SkipReason,
    classify_skip,
    iter_records,
    record_to_chatml,
)

# ---- helpers ----------------------------------------------------------------


def _load_samples(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- 1. thinking + polite ---------------------------------------------------


def test_thinking_polite_produces_chatml_with_think_block(
    sample_joryu_path: Path,
) -> None:
    rec = _load_samples(sample_joryu_path)[0]
    out = record_to_chatml(rec)
    assert out is not None
    # all three role blocks present
    assert out.count(IM_START) == 3
    assert out.count(IM_END) == 3
    # think block present and properly closed
    assert THINK_OPEN in out
    assert THINK_CLOSE in out
    assert out.index(THINK_OPEN) < out.index(THINK_CLOSE)
    # body parts copied verbatim
    assert str(rec["system_prompt"]) in out
    assert str(rec["prompt"]) in out
    assert str(rec["answer"]) in out
    assert str(rec["thinking_trace"]) in out


def test_thinking_record_role_ordering(sample_joryu_path: Path) -> None:
    rec = _load_samples(sample_joryu_path)[0]
    out = record_to_chatml(rec)
    assert out is not None
    assert out.index("system") < out.index("user") < out.index("assistant")


# ---- 2. nothinking + casual -------------------------------------------------


def test_nothinking_omits_think_block(sample_joryu_path: Path) -> None:
    rec = _load_samples(sample_joryu_path)[1]
    out = record_to_chatml(rec)
    assert out is not None
    assert THINK_OPEN not in out
    assert THINK_CLOSE not in out
    assert str(rec["answer"]) in out


# ---- 3. thinking + expert ---------------------------------------------------


def test_thinking_expert_includes_think_block(sample_joryu_path: Path) -> None:
    rec = _load_samples(sample_joryu_path)[2]
    out = record_to_chatml(rec)
    assert out is not None
    assert THINK_OPEN in out and THINK_CLOSE in out
    assert str(rec["thinking_trace"]) in out


# ---- 4. empty answer --------------------------------------------------------


def test_empty_answer_returns_none(sample_joryu_path: Path) -> None:
    rec = _load_samples(sample_joryu_path)[3]
    assert record_to_chatml(rec) is None
    assert classify_skip(rec) is SkipReason.EMPTY_ANSWER


# ---- 5. unclosed <think> tag ------------------------------------------------


def test_unclosed_think_returns_none(sample_joryu_path: Path) -> None:
    rec = _load_samples(sample_joryu_path)[4]
    assert record_to_chatml(rec) is None
    assert classify_skip(rec) is SkipReason.UNCLOSED_THINK


# ---- 6. repeated characters -------------------------------------------------


def test_repeated_characters_returns_none(sample_joryu_path: Path) -> None:
    rec = _load_samples(sample_joryu_path)[5]
    assert record_to_chatml(rec) is None
    assert classify_skip(rec) is SkipReason.REPEATED_CHARS


# ---- include_thinking flag --------------------------------------------------


def test_include_thinking_false_omits_block_even_when_present(
    sample_joryu_path: Path,
) -> None:
    rec = _load_samples(sample_joryu_path)[0]  # thinking record
    out = record_to_chatml(rec, include_thinking=False)
    assert out is not None
    assert THINK_OPEN not in out
    assert str(rec["answer"]) in out


# ---- iter_records (file-level streaming) ------------------------------------


def test_iter_records_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "tiny.jsonl"
    p.write_text(
        '\n{"prompt":"p","answer":"a","system_prompt":"s","mode":"nothinking",'
        '"thinking_trace":null}\n\n',
        encoding="utf-8",
    )
    recs = list(iter_records(p))
    assert len(recs) == 1
    assert recs[0]["prompt"] == "p"


def test_iter_records_reads_zstd(tmp_path: Path) -> None:
    zstd = pytest.importorskip("zstandard")
    raw = (
        '{"prompt":"p","answer":"a","system_prompt":"s","mode":"nothinking",'
        '"thinking_trace":null}\n'
    )
    p = tmp_path / "tiny.jsonl.zst"
    cctx = zstd.ZstdCompressor()
    p.write_bytes(cctx.compress(raw.encode("utf-8")))
    recs = list(iter_records(p))
    assert len(recs) == 1
    assert recs[0]["prompt"] == "p"
