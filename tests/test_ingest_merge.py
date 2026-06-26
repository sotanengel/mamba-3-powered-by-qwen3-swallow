"""TDD: raw + curated バンドルのマージ ingest (PR-4).

複数 BundleSpec を受け取り ``record_hash`` でレコードを統合する。
優先順位: ``curated > export > raw``。同一 ``record_hash`` が複数 source に
存在する場合は優先 source の payload を採用し、スコアは可能な限り curated 側を
保持する。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamba3jp.data.bundles import (
    ScoreEntry,
    compute_record_hash,
    discover_bundle,
    merge_sources,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _rec(prompt: str, answer: str, *, cfg: str = "sha256-cfg", **extra) -> dict:
    base = {
        "prompt": prompt,
        "answer": answer,
        "mode": "nothinking",
        "sampling": {"temperature": 0.6, "top_p": 0.9},
        "system_prompt": "system",
        "config_hash": cfg,
        "thinking_trace": None,
        "style_id": "polite",
        "category": "テスト",
    }
    base.update(extra)
    return base


def _build_raw(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "raw.jsonl"
    _write_jsonl(p, records)
    return p


def _build_curated(tmp_path: Path, records: list[dict], scores: list[dict]) -> Path:
    d = tmp_path / "curated_x"
    _write_jsonl(d / "responses.high_quality.jsonl", records)
    _write_jsonl(d / "scores.jsonl", scores)
    return d


# ---- merge basics ----------------------------------------------------------


def test_merge_dedups_by_record_hash(tmp_path: Path) -> None:
    rec_a = _rec("AAA", "answer A", cfg="sha256-cfg1")
    rec_b = _rec("BBB", "answer B", cfg="sha256-cfg2")
    raw = discover_bundle(_build_raw(tmp_path / "r", [rec_a, rec_b]))
    # 同一の rec_a を curated にも置く
    curated = discover_bundle(
        _build_curated(
            tmp_path / "c",
            [rec_a],
            [{"record_hash": compute_record_hash(rec_a), "final_score": 0.9, "accepted": True}],
        )
    )

    merged = list(merge_sources([raw, curated]))
    hashes = [compute_record_hash(rec) for rec, _, _ in merged]
    # 同一の rec_a が重複していない
    assert len(set(hashes)) == len(hashes) == 2
    # rec_a の方は curated 由来のスコアが伝播している
    by_hash = {compute_record_hash(rec): (rec, score, kind) for rec, score, kind in merged}
    _rec_a_payload, score_a, kind_a = by_hash[compute_record_hash(rec_a)]
    assert isinstance(score_a, ScoreEntry)
    assert score_a.final_score == pytest.approx(0.9)
    assert kind_a == "curated"
    # rec_b は raw 由来 → score 未結合・kind=raw
    _rec_b_payload, score_b, kind_b = by_hash[compute_record_hash(rec_b)]
    assert score_b is None
    assert kind_b == "raw"


def test_merge_no_overlap_concatenates_all(tmp_path: Path) -> None:
    rec_a = _rec("a-only", "answer A")
    rec_b = _rec("c-only", "answer B")
    raw = discover_bundle(_build_raw(tmp_path / "r", [rec_a]))
    curated = discover_bundle(
        _build_curated(
            tmp_path / "c",
            [rec_b],
            [{"record_hash": compute_record_hash(rec_b), "final_score": 0.8, "accepted": True}],
        )
    )

    merged = list(merge_sources([raw, curated]))
    assert len(merged) == 2
    seen = {compute_record_hash(r) for r, _, _ in merged}
    assert seen == {compute_record_hash(rec_a), compute_record_hash(rec_b)}


# ---- priority order --------------------------------------------------------


def test_curated_overrides_raw_on_conflict(tmp_path: Path) -> None:
    """raw と curated で同一 record_hash が衝突したら curated 側の dict を採用する。"""
    rec_in_raw = _rec("dup", "shared answer", cfg="sha256-cfg3")
    # curated 側に追加フィールドを付ける (curated ペイロードに来ること)
    rec_in_curated = dict(rec_in_raw)
    rec_in_curated["__source_marker__"] = "curated_wins"

    raw = discover_bundle(_build_raw(tmp_path / "r", [rec_in_raw]))
    curated = discover_bundle(
        _build_curated(
            tmp_path / "c",
            [rec_in_curated],
            [
                {
                    "record_hash": compute_record_hash(rec_in_raw),
                    "final_score": 0.77,
                    "accepted": True,
                }
            ],
        )
    )
    merged = list(merge_sources([raw, curated]))
    assert len(merged) == 1
    rec, score, kind = merged[0]
    assert rec.get("__source_marker__") == "curated_wins"
    assert score is not None and score.final_score == pytest.approx(0.77)
    assert kind == "curated"


def test_merge_input_order_does_not_affect_winner(tmp_path: Path) -> None:
    """specs を逆順で渡しても curated が勝つ (kind ベースの優先順位)。"""
    rec = _rec("dup", "answer")
    raw = discover_bundle(_build_raw(tmp_path / "r", [rec]))
    rec_curated = dict(rec, __source_marker__="curated_wins")
    curated = discover_bundle(
        _build_curated(
            tmp_path / "c",
            [rec_curated],
            [{"record_hash": compute_record_hash(rec), "final_score": 0.5, "accepted": True}],
        )
    )
    merged = list(merge_sources([curated, raw]))
    assert len(merged) == 1
    rec, _, kind = merged[0]
    assert rec.get("__source_marker__") == "curated_wins"
    assert kind == "curated"
