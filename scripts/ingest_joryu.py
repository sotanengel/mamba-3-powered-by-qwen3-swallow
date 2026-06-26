"""Ingest joryu-pipline 出力 (raw / export / curated) を ChatML 中間 JSONL に変換する。

Usage:
    # 単一ファイル (従来通り)
    python scripts/ingest_joryu.py \
        --input /data/joryu/responses.jsonl \
        --output data/intermediate/chatml.jsonl

    # export バンドル (sha256 検証 + scores.jsonl 結合)
    python scripts/ingest_joryu.py \
        --bundle /data/joryu/exports/20260621T000000Z \
        --output data/intermediate/chatml.jsonl \
        --min-score 0.7

    # curated バンドル (raw フィルタを自動スキップ)
    python scripts/ingest_joryu.py \
        --bundle /data/joryu/curated/latest \
        --output data/intermediate/chatml.jsonl

出力 JSONL の各行: ``{"text": "<ChatML>", "meta": {...}}``。
provenance は ``<output親>/manifest_partial.json`` に書き出す。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mamba3jp.data.bundles import (
    BundleSpec,
    discover_bundle,
    iter_bundle,
    load_scores,
    merge_sources,
    verify_sha256,
)
from mamba3jp.data.ingest import classify_skip, iter_records, record_to_chatml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="raw joryu responses.jsonl(.zst)")
    src.add_argument(
        "--bundle",
        type=Path,
        help="export ディレクトリ または curated ディレクトリ。auto-discover される",
    )
    src.add_argument(
        "--sources-yaml",
        type=Path,
        help="複数 source をマージする (yaml の sources: リスト)。raw+curated を curated 優先で結合",
    )

    p.add_argument("--output", type=Path, required=True, help="ChatML 中間 JSONL 出力")
    p.add_argument(
        "--stats", type=Path, default=Path("logs/ingest_stats.json"), help="統計 JSON 出力"
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="thinking_trace を <think> ブロックに含めない",
    )
    p.add_argument(
        "--no-tool-calls",
        action="store_true",
        help="multi-turn 出力時に <tool_call> ブロックを含めない",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="scores.jsonl の final_score 閾値 (未満を LOW_SCORE で除外)",
    )
    p.add_argument(
        "--skip-quality-filter",
        action="store_true",
        help="UNCLOSED_THINK / REPEATED_CHARS のフィルタを省く (curated 由来は auto-on)",
    )
    p.add_argument(
        "--verify-sha256",
        dest="verify_sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export バンドルの SHA256SUMS を検証する (既定 ON)",
    )
    return p.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_meta(rec: dict[str, Any], source_kind: str, score: float | None) -> dict[str, Any]:
    return {
        "style_id": rec.get("style_id"),
        "mode": rec.get("mode"),
        "category": rec.get("category"),
        "config_hash": rec.get("config_hash"),
        "final_score": score,
        "source_kind": source_kind,
    }


def _resolve_spec(args: argparse.Namespace) -> BundleSpec:
    target = args.bundle if args.bundle is not None else args.input
    return discover_bundle(target)


def _load_sources_yaml(path: Path) -> list[BundleSpec]:
    """``sources: [{kind, path}, ...]`` を読み、各 path を discover する。

    yaml 側の ``kind`` は注釈的扱いで、実際の判別は :func:`discover_bundle` が行う。
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: top-level 'sources:' list is required")
    specs: list[BundleSpec] = []
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"{path}: each source needs a 'path' key")
        specs.append(discover_bundle(Path(str(entry["path"]))))
    return specs


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    if args.sources_yaml is not None:
        specs = _load_sources_yaml(args.sources_yaml)
    else:
        specs = [_resolve_spec(args)]

    if args.verify_sha256:
        for s in specs:
            if s.kind == "export":
                verify_sha256(s)

    # 各レコードがどの kind 由来かを保持して、curated 由来なら品質フィルタを自動スキップ。
    include_thinking = not args.no_thinking
    include_tool_calls = not args.no_tool_calls

    n_in = n_out = 0
    skip_by_reason: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    by_style: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    config_hashes: set[str] = set()

    iterator: Any
    if len(specs) > 1:
        iterator = merge_sources(specs)
    else:
        spec = specs[0]
        scores = load_scores(spec.scores_path) if spec.scores_path else {}
        if spec.scores_path is not None or spec.kind != "raw":
            iterator = ((rec, score, spec.kind) for rec, score in iter_bundle(spec, scores=scores))
        else:
            iterator = ((rec, None, spec.kind) for rec in iter_records(spec.responses_path))

    with args.output.open("w", encoding="utf-8") as out_f:
        for rec, score_entry, src_kind in iterator:
            n_in += 1
            score_val = score_entry.final_score if score_entry is not None else None
            skip_quality_filter = args.skip_quality_filter or src_kind == "curated"
            chatml = record_to_chatml(
                rec,
                include_thinking=include_thinking,
                include_tool_calls=include_tool_calls,
                min_score=args.min_score,
                score=score_val,
                skip_quality_filter=skip_quality_filter,
            )
            if chatml is None:
                reason = classify_skip(
                    rec,
                    min_score=args.min_score,
                    score=score_val,
                    skip_quality_filter=skip_quality_filter,
                )
                skip_by_reason[reason.value if reason else "unknown"] += 1
                continue
            n_out += 1
            meta = _build_meta(rec, src_kind, score_val)
            by_category[str(meta["category"] or "unknown")] += 1
            by_mode[str(meta["mode"] or "unknown")] += 1
            by_style[str(meta["style_id"] or "unknown")] += 1
            by_kind[src_kind] += 1
            if isinstance(meta["config_hash"], str):
                config_hashes.add(meta["config_hash"])
            out_f.write(
                json.dumps({"text": chatml, "meta": meta}, ensure_ascii=False) + "\n"
            )

    manifest_partial = {
        "sources": [
            {
                "kind": s.kind,
                "responses_path": str(s.responses_path),
                "scores_path": str(s.scores_path) if s.scores_path else None,
                "meta_path": str(s.meta_path) if s.meta_path else None,
                "sha256_path": str(s.sha256_path) if s.sha256_path else None,
                "responses_sha256": _sha256_file(s.responses_path),
            }
            for s in specs
        ],
        "cli_args": {
            "input": str(args.input) if args.input else None,
            "bundle": str(args.bundle) if args.bundle else None,
            "sources_yaml": str(args.sources_yaml) if args.sources_yaml else None,
            "min_score": args.min_score,
            "skip_quality_filter": args.skip_quality_filter,
            "no_thinking": args.no_thinking,
            "no_tool_calls": args.no_tool_calls,
            "verify_sha256": args.verify_sha256,
        },
        "config_hashes": sorted(config_hashes),
        "records_in": n_in,
        "records_out": n_out,
        "by_style": dict(by_style),
        "by_mode": dict(by_mode),
        "by_category": dict(by_category),
        "by_source_kind": dict(by_kind),
    }
    (args.output.parent / "manifest_partial.json").write_text(
        json.dumps(manifest_partial, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "source_kinds": [s.kind for s in specs],
        "inputs": [str(s.responses_path) for s in specs],
        "output": str(args.output),
        "records_in": n_in,
        "records_out": n_out,
        "skip_rate": (n_in - n_out) / n_in if n_in else 0.0,
        "skip_by_reason": dict(skip_by_reason),
        "by_category": dict(by_category),
        "by_mode": dict(by_mode),
        "by_style": dict(by_style),
        "by_source_kind": dict(by_kind),
        "config_hashes_seen": len(config_hashes),
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    kinds = "+".join(s.kind for s in specs)
    print(f"[ingest] kinds={kinds} wrote {n_out}/{n_in} records → {args.output}")
    print(f"[ingest] stats → {args.stats}")
    print(f"[ingest] manifest_partial → {args.output.parent / 'manifest_partial.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
