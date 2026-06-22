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


def classify_skip(rec: dict[str, Any]) -> SkipReason | None:
    """Return the first applicable skip reason for ``rec`` or ``None`` if it passes.

    Order matters: cheaper checks first.
    """
    answer = str(rec.get("answer") or "").strip()
    if len(answer) < MIN_ANSWER_CHARS:
        return SkipReason.EMPTY_ANSWER

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

    return None


def _has_unclosed_think(text: str) -> bool:
    return text.count(THINK_OPEN) > text.count(THINK_CLOSE)


def record_to_chatml(rec: dict[str, Any], *, include_thinking: bool = True) -> str | None:
    """Render a single joryu record as a Qwen3 ChatML string.

    Returns ``None`` if the record fails the quality filter.

    When ``include_thinking`` is True and the record is a thinking-mode answer
    with a non-empty ``thinking_trace``, the trace is wrapped in ``<think>...</think>``
    and emitted at the top of the assistant turn.
    """
    if classify_skip(rec) is not None:
        return None

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


def _build_assistant_body(
    rec: dict[str, Any], answer: str, *, include_thinking: bool
) -> str:
    if not include_thinking:
        return answer
    if rec.get("mode") != "thinking":
        return answer
    trace = rec.get("thinking_trace")
    if not isinstance(trace, str) or not trace.strip():
        return answer
    return f"{THINK_OPEN}\n{trace.strip()}\n{THINK_CLOSE}\n\n{answer}"


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
