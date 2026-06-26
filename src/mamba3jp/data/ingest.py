"""Convert joryu-pipline records into Qwen3-style ChatML strings.

A joryu record looks like::

    {
        "prompt": "...",
        "category": "...",
        "style_id": "polite" | "casual" | "expert",
        "mode": "thinking" | "nothinking",
        "system_prompt": "...",
        "sampling": {...},
        "thinking_trace": str | null,
        "reasoning": str,
        "answer": "...",
        "model": "Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
        "config_hash": "...",
        "created_at": "..."
    }

The produced ChatML text is what we feed into the Qwen3 tokenizer for binidx
construction (issue #4). The thinking trace, when present, is re-inserted at the
top of the assistant turn — this lets the student model learn the teacher's
reasoning trace under causal LM training (per the user-confirmed plan).
"""

from __future__ import annotations

import enum
import io
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Qwen3 ChatML special tokens (verbatim strings, not token ids).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Records whose answer is shorter than this (after strip) are dropped.
MIN_ANSWER_CHARS = 1

# Same char repeated this many times in a row marks the record as low quality.
# Requirement: "同一トークンの5回以上連続繰り返し" — we use char-level here since
# tokenization happens downstream.
MAX_REPEAT_RUN = 5
_REPEAT_RE = re.compile(rf"(.)\1{{{MAX_REPEAT_RUN - 1},}}", flags=re.UNICODE)


class SkipReason(enum.Enum):
    """Why a joryu record was excluded from the training data."""

    EMPTY_ANSWER = "empty_answer"
    UNCLOSED_THINK = "unclosed_think"
    REPEATED_CHARS = "repeated_chars"
    MALFORMED_TURNS = "malformed_turns"
    LOW_SCORE = "low_score"


def classify_skip(
    rec: dict[str, Any],
    *,
    min_score: float | None = None,
    score: float | None = None,
    skip_quality_filter: bool = False,
) -> SkipReason | None:
    """Return the first applicable skip reason for ``rec`` or ``None`` if it passes.

    Order matters: cheaper checks first. EMPTY_ANSWER と turns 整合性は常に検査するが、
    ``skip_quality_filter=True`` (curated バンドル由来) では UNCLOSED_THINK /
    REPEATED_CHARS の再適用を省く。``min_score`` と ``score`` を併用すると閾値未満を
    LOW_SCORE で落とす。score が ``None`` の場合 (= スコア未結合) は通過させる。
    """
    answer = str(rec.get("answer") or "").strip()
    if len(answer) < MIN_ANSWER_CHARS:
        return SkipReason.EMPTY_ANSWER

    if not skip_quality_filter:
        # Unclosed <think> in either the trace or the answer body.
        for field in ("thinking_trace", "answer"):
            text = rec.get(field)
            if isinstance(text, str) and _has_unclosed_think(text):
                return SkipReason.UNCLOSED_THINK

        if _REPEAT_RE.search(answer):
            return SkipReason.REPEATED_CHARS
        trace = rec.get("thinking_trace")
        if isinstance(trace, str) and _REPEAT_RE.search(trace):
            return SkipReason.REPEATED_CHARS

    turns = rec.get("turns")
    if isinstance(turns, list) and turns and not _turns_well_formed(turns):
        return SkipReason.MALFORMED_TURNS

    if min_score is not None and score is not None and score < min_score:
        return SkipReason.LOW_SCORE

    return None


def _turns_well_formed(turns: list[Any]) -> bool:
    """``turns`` のロール順序を粗く検証する。

    - 最初の要素は ``assistant`` でなければならない (tool が先頭に来ない)
    - 各 ``tool`` turn は直前の ``assistant`` turn が ``tool_calls`` を持つこと
    """
    expect_tool_continuation = False
    for turn in turns:
        if not isinstance(turn, dict):
            return False
        role = turn.get("role")
        if role == "assistant":
            expect_tool_continuation = bool(turn.get("tool_calls"))
        elif role == "tool":
            if not expect_tool_continuation:
                return False
            # 連続する tool turn は許容するが、tool_call の対応が無いなら次の tool は許可しない
        else:
            return False
    return True


def _has_unclosed_think(text: str) -> bool:
    return text.count(THINK_OPEN) > text.count(THINK_CLOSE)


def record_to_chatml(
    rec: dict[str, Any],
    *,
    include_thinking: bool = True,
    include_tool_calls: bool = True,
    min_score: float | None = None,
    score: float | None = None,
    skip_quality_filter: bool = False,
) -> str | None:
    """Render a single joryu record as a Qwen3 ChatML string.

    Returns ``None`` if the record fails the quality filter.

    When ``include_thinking`` is True and the record is a thinking-mode answer
    with a non-empty ``thinking_trace``, the trace is wrapped in ``<think>...</think>``
    and emitted at the top of the (first) assistant turn.

    ``turns`` が非空の場合、:func:`record_to_chatml_multiturn` にディスパッチする。
    """
    if (
        classify_skip(
            rec,
            min_score=min_score,
            score=score,
            skip_quality_filter=skip_quality_filter,
        )
        is not None
    ):
        return None

    turns = rec.get("turns")
    if isinstance(turns, list) and turns:
        return record_to_chatml_multiturn(
            rec, include_thinking=include_thinking, include_tool_calls=include_tool_calls
        )

    system_prompt = str(rec.get("system_prompt") or "").strip()
    prompt = str(rec.get("prompt") or "").strip()
    answer = str(rec.get("answer") or "").strip()

    assistant_body = _build_assistant_body(rec, answer, include_thinking=include_thinking)

    parts = [
        f"{IM_START}system\n{system_prompt}{IM_END}\n",
        f"{IM_START}user\n{prompt}{IM_END}\n",
        f"{IM_START}assistant\n{assistant_body}{IM_END}\n",
    ]
    return "".join(parts)


def _build_assistant_body(rec: dict[str, Any], answer: str, *, include_thinking: bool) -> str:
    if not include_thinking:
        return answer
    if rec.get("mode") != "thinking":
        return answer
    trace = rec.get("thinking_trace")
    if not isinstance(trace, str) or not trace.strip():
        return answer
    return f"{THINK_OPEN}\n{trace.strip()}\n{THINK_CLOSE}\n\n{answer}"


# -- multi-turn rendering -----------------------------------------------------


def record_to_chatml_multiturn(
    rec: dict[str, Any],
    *,
    include_thinking: bool = True,
    include_tool_calls: bool = True,
) -> str | None:
    """マルチターン (Tool Loop) レコードを Qwen3 ChatML 文字列にレンダリングする。

    ``turns`` が空 / 形式不正の場合は ``None`` を返す。tool_call は Qwen3 ネイティブの
    ``<tool_call>{"name":...,"arguments":...}</tool_call>`` ブロックで assistant content に
    埋め込む。tool role は ``<|im_start|>tool`` で挟む。
    """
    turns = rec.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    if not _turns_well_formed(turns):
        return None

    system_prompt = str(rec.get("system_prompt") or "").strip()
    prompt = str(rec.get("prompt") or "").strip()

    parts = [
        f"{IM_START}system\n{system_prompt}{IM_END}\n",
        f"{IM_START}user\n{prompt}{IM_END}\n",
    ]
    first_assistant_emitted = False
    for turn in turns:
        role = turn.get("role")
        if role == "assistant":
            body = _build_assistant_turn_body(
                turn,
                rec,
                include_thinking=include_thinking and not first_assistant_emitted,
                include_tool_calls=include_tool_calls,
            )
            parts.append(f"{IM_START}assistant\n{body}{IM_END}\n")
            first_assistant_emitted = True
        elif role == "tool":
            tool_content = str(turn.get("content") or "")
            parts.append(f"{IM_START}tool\n{tool_content}{IM_END}\n")
        else:  # pragma: no cover — guarded by _turns_well_formed
            return None
    return "".join(parts)


def _build_assistant_turn_body(
    turn: dict[str, Any],
    rec: dict[str, Any],
    *,
    include_thinking: bool,
    include_tool_calls: bool,
) -> str:
    content = str(turn.get("content") or "").strip()
    blocks: list[str] = []
    if include_thinking and rec.get("mode") == "thinking":
        trace = rec.get("thinking_trace")
        if isinstance(trace, str) and trace.strip():
            blocks.append(f"{THINK_OPEN}\n{trace.strip()}\n{THINK_CLOSE}\n")
    if content:
        blocks.append(content)
    if include_tool_calls:
        for call in turn.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            args = call.get("arguments")
            if not isinstance(name, str):
                continue
            payload = json.dumps(
                {"name": name, "arguments": args if isinstance(args, dict) else {}},
                ensure_ascii=False,
                sort_keys=True,
            )
            blocks.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(blocks)


# -- file streaming -----------------------------------------------------------


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream joryu records from a plain ``.jsonl`` or zstd-compressed ``.jsonl.zst``.

    Empty lines are skipped. Each record is decoded with ``json.loads``.
    """
    path = Path(path)
    if path.suffix == ".zst":
        import zstandard as zstd

        with path.open("rb") as raw:
            dctx = zstd.ZstdDecompressor()
            stream = io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8")
            yield from _iter_jsonl_lines(stream)
    else:
        with path.open(encoding="utf-8") as f:
            yield from _iter_jsonl_lines(f)


def _iter_jsonl_lines(stream: io.TextIOBase) -> Iterator[dict[str, Any]]:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)
