"""Ingest joryu-pipline responses.jsonl(.zst) into ChatML intermediate JSONL.

Usage (inside container):
    python scripts/ingest_joryu.py \
        --input /data/joryu/responses.jsonl \
        --output data/intermediate/chatml.jsonl \
        --stats logs/ingest_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mamba3jp.data.ingest import classify_skip, iter_records, record_to_chatml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="joryu responses.jsonl(.zst)")
    p.add_argument("--output", type=Path, required=True, help="ChatML JSONL output")
    p.add_argument(
        "--stats",
        type=Path,
        default=Path("logs/ingest_stats.json"),
        help="ingest statistics JSON output",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="omit <think> blocks even when thinking_trace is present",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    skip_by_reason: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()

    include_thinking = not args.no_thinking
    with args.output.open("w", encoding="utf-8") as out_f:
        for rec in iter_records(args.input):
            n_in += 1
            chatml = record_to_chatml(rec, include_thinking=include_thinking)
            if chatml is None:
                reason = classify_skip(rec)
                skip_by_reason[reason.value if reason else "unknown"] += 1
                continue
            n_out += 1
            by_category[str(rec.get("category", "unknown"))] += 1
            by_mode[str(rec.get("mode", "unknown"))] += 1
            out_f.write(json.dumps({"text": chatml}, ensure_ascii=False) + "\n")

    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "records_in": n_in,
        "records_out": n_out,
        "skip_rate": (n_in - n_out) / n_in if n_in else 0.0,
        "skip_by_reason": dict(skip_by_reason),
        "by_category": dict(by_category),
        "by_mode": dict(by_mode),
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] wrote {n_out}/{n_in} records → {args.output}")
    print(f"[ingest] stats → {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
