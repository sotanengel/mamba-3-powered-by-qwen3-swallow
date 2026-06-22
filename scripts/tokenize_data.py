"""Tokenize ChatML JSONL into train/val binidx files using the Qwen3 tokenizer.

Usage (inside container):
    python scripts/tokenize_data.py \
        --input data/intermediate/chatml.jsonl \
        --out-dir data/tokenized \
        --tokenizer Qwen/Qwen3-8B \
        --val-ratio 0.05

Each document is tokenized independently, gets an EOS token appended
(``<|endoftext|>`` = 151643 in Qwen3's vocab), and is appended to the binidx
stream. The val split is done at the document level, not the token level, so a
single document never spans both splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mamba3jp.data.binidx import BinIdxWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="ChatML JSONL")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stats",
        type=Path,
        default=Path("logs/tokenize_stats.json"),
    )
    return p.parse_args()


def _iter_text(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") if isinstance(obj, dict) else obj
            if isinstance(text, str) and text:
                yield text


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

    rng = random.Random(args.seed)
    n_train = n_val = 0
    tok_train = tok_val = 0

    with BinIdxWriter(train_bin, train_idx) as tw, BinIdxWriter(val_bin, val_idx) as vw:
        for text in tqdm(_iter_text(args.input), desc="tokenize"):
            ids = tok.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint32)
            if rng.random() < args.val_ratio:
                vw.add_document(arr)
                n_val += 1
                tok_val += int(arr.size)
            else:
                tw.add_document(arr)
                n_train += 1
                tok_train += int(arr.size)

    stats = {
        "tokenizer": args.tokenizer,
        "vocab_size": getattr(tok, "vocab_size", None) or len(tok),
        "eos_token_id": int(eos_id),
        "val_ratio": args.val_ratio,
        "docs_train": n_train,
        "docs_val": n_val,
        "tokens_train": tok_train,
        "tokens_val": tok_val,
        "tokens_total": tok_train + tok_val,
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[tokenize] train: {n_train} docs / {tok_train:,} tokens")
    print(f"[tokenize] val:   {n_val} docs / {tok_val:,} tokens")
    print(f"[tokenize] stats → {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
