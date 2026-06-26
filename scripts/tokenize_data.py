"""Tokenize ChatML JSONL into train/val binidx files using the teacher tokenizer.

Usage (inside container):
    python scripts/tokenize_data.py \
        --input data/intermediate/chatml.jsonl \
        --out-dir data/tokenized \
        --tokenizer tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4 \
        --val-ratio 0.05

Each document is tokenized independently, gets an EOS token appended
(``<|endoftext|>`` = 151643 in Qwen3's vocab), and is appended to the binidx
stream. The val split is done at the document level, not the token level, so a
single document never spans both splits.

PR-5 で追加:
- ``(style_id, mode)`` バケットによる **層化分割**
- 各文書の ``final_score`` を ``train.weights.npy`` / ``val.weights.npy`` に同梱
- ``ingest_partial`` を引き継いだ統合 ``manifest.json`` を出力
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from mamba3jp.data.binidx import BinIdxWriter
from mamba3jp.data.manifest import write_manifest

# 層化バケットがこの件数を下回るとフォールバック (グローバル random Bernoulli)。
MIN_STRATIFY_BUCKET = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="ChatML JSONL")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--tokenizer",
        type=str,
        default="tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
    )
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stats",
        type=Path,
        default=Path("logs/tokenize_stats.json"),
    )
    p.add_argument(
        "--ingest-manifest",
        type=Path,
        default=None,
        help="ingest_joryu.py が書く manifest_partial.json (provenance を伝播する)",
    )
    return p.parse_args()


def _iter_text(path: Path) -> Iterator[str]:
    """text-only ストリーム (後方互換)。"""
    for text, _meta in _iter_documents(path):
        yield text


def _iter_documents(path: Path) -> Iterator[tuple[str, dict]]:
    """``{"text":..., "meta":{...}}`` 形式と純テキスト ``{"text":...}`` の両方を読む。

    text のみの旧形式では meta=={} を yield する。
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                text = obj.get("text")
                raw_meta = obj.get("meta")
                meta = raw_meta if isinstance(raw_meta, dict) else {}
            else:
                text = obj
                meta = {}
            if isinstance(text, str) and text:
                yield text, meta


def stratified_assign(
    metas: list[dict],
    *,
    val_ratio: float,
    rng: random.Random,
) -> list[Literal["train", "val"]]:
    """``(style_id, mode)`` バケットごとに ``ceil(val_ratio * |bucket|)`` を val に割り当てる。

    バケットの件数が :data:`MIN_STRATIFY_BUCKET` 未満ならグローバル Bernoulli
    にフォールバック。meta 欠落の旧形式は ``("unknown", "unknown")`` バケットに集約。
    """
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, meta in enumerate(metas):
        key = (str(meta.get("style_id") or "unknown"), str(meta.get("mode") or "unknown"))
        buckets[key].append(i)

    assigns: list[Literal["train", "val"]] = ["train"] * len(metas)
    fallback_indices: list[int] = []
    for indices in buckets.values():
        if len(indices) < MIN_STRATIFY_BUCKET:
            fallback_indices.extend(indices)
            continue
        n_val = max(1, math.ceil(len(indices) * val_ratio))
        n_val = min(n_val, len(indices))
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for idx in shuffled[:n_val]:
            assigns[idx] = "val"

    if fallback_indices:
        rng.shuffle(fallback_indices)
        n_val_global = max(1, round(len(fallback_indices) * val_ratio))
        n_val_global = min(n_val_global, len(fallback_indices))
        # 端数次第では 0 になる可能性があるが、最低 1 件は val に出す。
        if val_ratio <= 0.0:
            n_val_global = 0
        for idx in fallback_indices[:n_val_global]:
            assigns[idx] = "val"
    return assigns


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer  # heavy import; do it inside main

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
    if eos_id is None:
        raise RuntimeError("could not locate an EOS token id in the tokenizer")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    train_bin = args.out_dir / "train.bin"
    train_idx = args.out_dir / "train.idx"
    val_bin = args.out_dir / "val.bin"
    val_idx = args.out_dir / "val.idx"

    # 1) 全文書の text と meta を 1 パス目で読み込む。stratified_assign は
    #    全件のバケット分布を見てから割り当てるため、これは仕様上不可避。
    documents: list[tuple[str, dict]] = list(_iter_documents(args.input))
    metas = [m for _t, m in documents]
    rng = random.Random(args.seed)
    assigns = stratified_assign(metas, val_ratio=args.val_ratio, rng=rng)

    # 2) 2 パス目でトークナイズ + binidx + weight を書く。
    train_weights: list[float] = []
    val_weights: list[float] = []
    tok_train = tok_val = 0
    bucket_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    config_hashes: set[str] = set()
    by_source_kind: Counter[str] = Counter()

    with BinIdxWriter(train_bin, train_idx) as tw, BinIdxWriter(val_bin, val_idx) as vw:
        for (text, meta), label in tqdm(
            zip(documents, assigns, strict=True), desc="tokenize", total=len(documents)
        ):
            ids = tok.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint32)
            score = meta.get("final_score")
            weight = float(score) if isinstance(score, (int, float)) else 1.0
            bucket_label = f"{meta.get('style_id') or 'unknown'}/{meta.get('mode') or 'unknown'}"
            cfg = meta.get("config_hash")
            if isinstance(cfg, str):
                config_hashes.add(cfg)
            kind = meta.get("source_kind")
            if isinstance(kind, str):
                by_source_kind[kind] += 1
            if label == "val":
                vw.add_document(arr)
                val_weights.append(weight)
                tok_val += int(arr.size)
                bucket_counts[bucket_label]["val"] += 1
            else:
                tw.add_document(arr)
                train_weights.append(weight)
                tok_train += int(arr.size)
                bucket_counts[bucket_label]["train"] += 1

    np.save(args.out_dir / "train.weights.npy", np.asarray(train_weights, dtype=np.float32))
    np.save(args.out_dir / "val.weights.npy", np.asarray(val_weights, dtype=np.float32))

    stats = {
        "tokenizer": args.tokenizer,
        "vocab_size": getattr(tok, "vocab_size", None) or len(tok),
        "eos_token_id": int(eos_id),
        "val_ratio": args.val_ratio,
        "docs_train": len(train_weights),
        "docs_val": len(val_weights),
        "tokens_train": tok_train,
        "tokens_val": tok_val,
        "tokens_total": tok_train + tok_val,
        "by_bucket": dict(bucket_counts),
        "config_hashes_seen": len(config_hashes),
        "by_source_kind": dict(by_source_kind),
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    ingest_partial: dict[str, Any] | None = None
    if args.ingest_manifest is not None and args.ingest_manifest.exists():
        ingest_partial = json.loads(args.ingest_manifest.read_text(encoding="utf-8"))
    elif (args.input.parent / "manifest_partial.json").exists():
        ingest_partial = json.loads(
            (args.input.parent / "manifest_partial.json").read_text(encoding="utf-8")
        )

    manifest_parts: list[dict[str, Any]] = []
    if ingest_partial is not None:
        manifest_parts.append({"ingest": ingest_partial})
    manifest_parts.append(
        {
            "tokenizer": args.tokenizer,
            "vocab_size": getattr(tok, "vocab_size", None) or len(tok),
            "eos_token_id": int(eos_id),
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "train": {"docs": len(train_weights), "tokens": tok_train},
            "val": {"docs": len(val_weights), "tokens": tok_val},
            "by_bucket": dict(bucket_counts),
            "by_source_kind": dict(by_source_kind),
            "config_hashes": sorted(config_hashes),
        }
    )
    write_manifest(args.out_dir, manifest_parts)

    print(f"[tokenize] train: {len(train_weights)} docs / {tok_train:,} tokens")
    print(f"[tokenize] val:   {len(val_weights)} docs / {tok_val:,} tokens")
    print(f"[tokenize] stats → {args.stats}")
    print(f"[tokenize] manifest → {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
