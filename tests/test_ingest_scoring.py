"""TDD: スコア閾値フィルタ + curated バンドルの quality-filter スキップ (PR-3).

joryu の curate 段で算出された ``final_score`` を ``--min-score`` と比較し、
閾値未満のレコードを ``SkipReason.LOW_SCORE`` で除外する。curated バンドル由来の
レコードは既にフィルタ済みなので簡易品質フィルタをスキップする。
"""

from __future__ import annotations

from mamba3jp.data.ingest import SkipReason, classify_skip


def _ok_record() -> dict:
    return {
        "prompt": "テスト",
        "answer": "ok",
        "system_prompt": "",
        "mode": "nothinking",
        "thinking_trace": None,
    }


# ---- min_score filter -------------------------------------------------------


def test_min_score_drops_below_threshold() -> None:
    rec = _ok_record()
    assert classify_skip(rec, min_score=0.7, score=0.65) is SkipReason.LOW_SCORE


def test_min_score_keeps_at_threshold() -> None:
    rec = _ok_record()
    assert classify_skip(rec, min_score=0.7, score=0.7) is None


def test_min_score_keeps_above_threshold() -> None:
    rec = _ok_record()
    assert classify_skip(rec, min_score=0.7, score=0.95) is None


def test_min_score_without_score_keeps_record() -> None:
    """スコアが結合されていないレコードは min_score 指定があっても通過する。"""
    rec = _ok_record()
    assert classify_skip(rec, min_score=0.7, score=None) is None


def test_min_score_none_disables_check() -> None:
    rec = _ok_record()
    assert classify_skip(rec, min_score=None, score=0.0) is None


# ---- skip_quality_filter (for curated bundles) -----------------------------


def test_skip_quality_filter_bypasses_repeat_check() -> None:
    """curated バンドル由来は raw 段のフィルタを再適用しない。"""
    rec = {
        "prompt": "テスト",
        "answer": "ああああああああああ",  # 通常なら REPEATED_CHARS
        "system_prompt": "",
        "mode": "nothinking",
        "thinking_trace": None,
    }
    # 既存挙動: フィルタ ON では REPEATED_CHARS
    assert classify_skip(rec) is SkipReason.REPEATED_CHARS
    # skip_quality_filter=True ではスキップしない
    assert classify_skip(rec, skip_quality_filter=True) is None


def test_skip_quality_filter_still_drops_empty_answer() -> None:
    """skip_quality_filter=True でも、回復不能な EMPTY_ANSWER だけは依然として落とす。"""
    rec = _ok_record()
    rec["answer"] = ""
    assert classify_skip(rec, skip_quality_filter=True) is SkipReason.EMPTY_ANSWER


def test_skip_quality_filter_still_validates_turns() -> None:
    """skip_quality_filter=True でも、turns の整合性検証は通す。"""
    rec = _ok_record()
    rec["turns"] = [{"role": "tool", "name": "calc", "content": "1"}]
    assert classify_skip(rec, skip_quality_filter=True) is SkipReason.MALFORMED_TURNS


def test_skip_quality_filter_combined_with_min_score() -> None:
    rec = _ok_record()
    rec["answer"] = "ああああああああああ"  # raw だと REPEATED_CHARS
    # curated 由来でフィルタスキップ → score 評価のみ
    assert (
        classify_skip(rec, skip_quality_filter=True, min_score=0.5, score=0.3)
        is SkipReason.LOW_SCORE
    )
    assert classify_skip(rec, skip_quality_filter=True, min_score=0.5, score=0.8) is None
