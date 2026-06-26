"""TDD: マルチターン / tool_call レンダラのテスト (PR-2).

joryu の ``turns`` フィールドを保持して Qwen3 ChatML を生成する。tool_call は
Qwen3 ネイティブの ``<tool_call>{"name":...,"arguments":...}</tool_call>`` ブロックで
assistant content 内に埋め込み、tool 応答は ``<|im_start|>tool`` ロールで挟む。

fixture: ``tests/data/sample_joryu_multiturn.jsonl``
  - record[0]: 正常な multi-turn (assistant + tool + assistant)
  - record[1]: turns 空 → 既存単一ターン経路 (回帰ガード)
  - record[2]: tool ロールが先頭で role 順不整合 → MALFORMED_TURNS
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
    record_to_chatml,
    record_to_chatml_multiturn,
)


@pytest.fixture(scope="session")
def multiturn_records() -> list[dict]:
    p = Path(__file__).parent / "data" / "sample_joryu_multiturn.jsonl"
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- multi-turn happy path --------------------------------------------------


def test_multiturn_preserves_role_ordering(multiturn_records: list[dict]) -> None:
    rec = multiturn_records[0]
    out = record_to_chatml(rec)
    assert out is not None
    # system, user, assistant(tool-call), tool, assistant(final) の 5 ブロック
    assert out.count(IM_START) == 5
    assert out.count(IM_END) == 5
    # 順序: system < user < 1st assistant < tool < 2nd assistant
    sys_pos = out.index(f"{IM_START}system")
    user_pos = out.index(f"{IM_START}user")
    asst1_pos = out.index(f"{IM_START}assistant")
    tool_pos = out.index(f"{IM_START}tool")
    asst2_pos = out.index(f"{IM_START}assistant", asst1_pos + 1)
    assert sys_pos < user_pos < asst1_pos < tool_pos < asst2_pos


def test_multiturn_emits_tool_call_block(multiturn_records: list[dict]) -> None:
    rec = multiturn_records[0]
    out = record_to_chatml(rec)
    assert out is not None
    # Qwen3 native tool_call block
    assert "<tool_call>" in out
    assert "</tool_call>" in out
    assert '"name": "fetch_url"' in out
    assert '"url": "https://example.com/weather"' in out


def test_multiturn_tool_response_content_present(multiturn_records: list[dict]) -> None:
    rec = multiturn_records[0]
    out = record_to_chatml(rec)
    assert out is not None
    # tool ロールの content がそのまま埋め込まれている
    assert "最高気温 28 度、晴れ" in out


def test_multiturn_thinking_block_at_top_of_first_assistant(
    multiturn_records: list[dict],
) -> None:
    """thinking_trace は最初の assistant turn の先頭に <think> として現れる。"""
    rec = multiturn_records[0]
    out = record_to_chatml(rec, include_thinking=True)
    assert out is not None
    assert THINK_OPEN in out
    assert THINK_CLOSE in out
    # think は最初の assistant の中、tool より前にある
    assert out.index(THINK_OPEN) < out.index(f"{IM_START}tool")


def test_multiturn_no_think_when_include_thinking_false(
    multiturn_records: list[dict],
) -> None:
    rec = multiturn_records[0]
    out = record_to_chatml(rec, include_thinking=False)
    assert out is not None
    assert THINK_OPEN not in out


def test_multiturn_include_tool_calls_false_drops_block(
    multiturn_records: list[dict],
) -> None:
    rec = multiturn_records[0]
    out = record_to_chatml_multiturn(rec, include_thinking=False, include_tool_calls=False)
    assert out is not None
    assert "<tool_call>" not in out
    # 最終応答テキストは残る
    assert "今日の東京の最高気温は 28 度の予報です。" in out


# ---- back-compat: empty turns falls back to single-turn --------------------


def test_empty_turns_uses_single_turn_path(multiturn_records: list[dict]) -> None:
    rec = multiturn_records[1]
    out = record_to_chatml(rec)
    assert out is not None
    # 単一ターン経路では system, user, assistant の 3 ブロックのみ
    assert out.count(IM_START) == 3
    assert "<tool_call>" not in out
    assert str(rec["answer"]) in out


def test_single_turn_record_without_turns_field_works(sample_joryu_path: Path) -> None:
    """既存 sample_joryu.jsonl のレコード (turns フィールド無し) で回帰しないこと。"""
    with sample_joryu_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for rec in records:
        out = record_to_chatml(rec)
        # quality filter で None になり得るがそれは既存挙動どおり
        if out is None:
            continue
        # turns 経路に滑り落ちて 3 ブロックを破壊していないこと
        assert out.count(IM_START) == 3


# ---- malformed turns --------------------------------------------------------


def test_malformed_turns_tool_before_assistant_skipped(
    multiturn_records: list[dict],
) -> None:
    rec = multiturn_records[2]
    # multiturn 単独関数では None を返す
    assert record_to_chatml_multiturn(rec, include_thinking=False) is None
    # classify_skip でも MALFORMED_TURNS と分類される
    assert classify_skip(rec) is SkipReason.MALFORMED_TURNS


# ---- direct multiturn API ---------------------------------------------------


def test_direct_multiturn_api_returns_none_for_empty_turns(
    multiturn_records: list[dict],
) -> None:
    rec = multiturn_records[1]  # turns == []
    out = record_to_chatml_multiturn(rec, include_thinking=False)
    assert out is None  # 明示的に空 turns はマルチターン責務外


def test_classify_skip_returns_none_for_valid_multiturn(
    multiturn_records: list[dict],
) -> None:
    rec = multiturn_records[0]
    assert classify_skip(rec) is None
