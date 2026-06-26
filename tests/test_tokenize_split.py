"""TDD: tokenize_data.py の層化分割 / manifest / weights 出力 (PR-5).

stratified_assign: (style_id, mode) バケットごとに ceil(val_ratio * |bucket|) を val へ。
バケット <20 件 はグローバル random フォールバック。

各文書の weight は ``meta.final_score`` を採用、欠落時は 1.0。
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tokenize_data import stratified_assign

from mamba3jp.data.manifest import write_manifest


def _meta(style_id: str, mode: str, *, score: float | None = None, cfg: str | None = None) -> dict:
    return {
        "style_id": style_id,
        "mode": mode,
        "final_score": score,
        "config_hash": cfg or "sha256-cfg",
    }


# ---- stratified_assign ------------------------------------------------------


def test_stratified_split_preserves_per_bucket_ratio() -> None:
    metas: list[dict] = []
    # bucket (polite, thinking) x 60
    metas += [_meta("polite", "thinking") for _ in range(60)]
    # bucket (casual, nothinking) x 40
    metas += [_meta("casual", "nothinking") for _ in range(40)]

    assigns = stratified_assign(metas, val_ratio=0.1, rng=random.Random(0))
    assert len(assigns) == len(metas)

    by_bucket: dict[tuple, Counter] = {}
    for meta, label in zip(metas, assigns, strict=True):
        key = (meta["style_id"], meta["mode"])
        by_bucket.setdefault(key, Counter())[label] += 1

    # bucket (polite, thinking): val = ceil(60 * 0.1) = 6
    assert by_bucket[("polite", "thinking")]["val"] == 6
    assert by_bucket[("polite", "thinking")]["train"] == 54
    # bucket (casual, nothinking): val = ceil(40 * 0.1) = 4
    assert by_bucket[("casual", "nothinking")]["val"] == 4
    assert by_bucket[("casual", "nothinking")]["train"] == 36


def test_small_buckets_fall_back_to_random() -> None:
    """20 件未満のバケットは個別層化せず、グローバル Bernoulli にフォールバック。"""
    metas = [_meta("polite", "thinking") for _ in range(5)]
    metas += [_meta("casual", "nothinking") for _ in range(10)]
    assigns = stratified_assign(metas, val_ratio=0.5, rng=random.Random(42))
    # フォールバック適用なので 5+10=15 件のうち val がほぼ半分。
    n_val = sum(1 for a in assigns if a == "val")
    assert 4 <= n_val <= 11  # 50% 中心の幅広い許容


def test_stratified_split_deterministic_under_same_seed() -> None:
    metas = [_meta("polite", "thinking") for _ in range(30)]
    metas += [_meta("casual", "nothinking") for _ in range(30)]
    a1 = stratified_assign(metas, val_ratio=0.1, rng=random.Random(7))
    a2 = stratified_assign(metas, val_ratio=0.1, rng=random.Random(7))
    assert a1 == a2


def test_stratified_split_handles_empty_meta() -> None:
    """meta 欠落の旧形式 ({}) では unknown バケットに集約され、グローバルにフォールバック。"""
    metas = [{} for _ in range(15)]
    assigns = stratified_assign(metas, val_ratio=0.2, rng=random.Random(0))
    assert len(assigns) == 15
    assert set(assigns) <= {"train", "val"}


# ---- manifest ---------------------------------------------------------------


def test_write_manifest_serializes_expected_keys(tmp_path: Path) -> None:
    out_dir = tmp_path / "tokenized"
    out_dir.mkdir()
    write_manifest(
        out_dir,
        parts=[
            {
                "ingest": {
                    "sources": [{"kind": "raw", "responses_path": "/x/r.jsonl"}],
                    "config_hashes": ["sha256-cfgA", "sha256-cfgB"],
                },
                "tokenizer": "Qwen/Qwen3",
                "vocab_size": 151672,
                "train": {"docs": 50, "tokens": 12345},
                "val": {"docs": 5, "tokens": 600},
                "by_bucket": {"polite/thinking": {"train": 30, "val": 3}},
            }
        ],
    )
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["tokenizer"] == "Qwen/Qwen3"
    assert data["vocab_size"] == 151672
    assert data["train"]["docs"] == 50
    assert "sha256-cfgA" in data["ingest"]["config_hashes"]


def test_write_manifest_attaches_git_sha_when_available(tmp_path: Path) -> None:
    """git sha が取れる環境では manifest.git_sha に文字列が入る。取れなくても落ちない。"""
    out_dir = tmp_path / "tokenized"
    out_dir.mkdir()
    write_manifest(out_dir, parts=[{"tokenizer": "t"}])
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    # 取れた場合は str、取れなかった場合は None。どちらでも OK。
    assert "git_sha" in data
    assert data["git_sha"] is None or isinstance(data["git_sha"], str)


# ---- pytest marker (slow integration test for full tokenize round-trip) ---


@pytest.mark.slow
def test_tokenize_round_trip_writes_weights_and_manifest(tmp_path: Path) -> None:
    """transformers が無くてもスキップ。weights.npy / manifest.json の存在のみ確認。"""
    pytest.importorskip("transformers")
    import numpy as np
    from scripts.tokenize_data import main as tokenize_main

    intermediate = tmp_path / "chatml.jsonl"
    docs = []
    for i in range(8):
        text = (
            "<|im_start|>system\nx<|im_end|>\n"
            f"<|im_start|>user\nq{i}<|im_end|>\n"
            f"<|im_start|>assistant\na{i}<|im_end|>\n"
        )
        docs.append({
            "text": text,
            "meta": {
                "style_id": "polite",
                "mode": "thinking",
                "final_score": 0.9 if i % 2 == 0 else None,
                "config_hash": "sha256-cfgX",
            },
        })
    with intermediate.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    out_dir = tmp_path / "tokenized"
    sys.argv = [
        "tokenize_data.py",
        "--input",
        str(intermediate),
        "--out-dir",
        str(out_dir),
        "--tokenizer",
        "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
        "--val-ratio",
        "0.25",
    ]
    tokenize_main()

    assert (out_dir / "train.weights.npy").exists()
    assert (out_dir / "val.weights.npy").exists()
    assert (out_dir / "manifest.json").exists()
    train_w = np.load(out_dir / "train.weights.npy")
    val_w = np.load(out_dir / "val.weights.npy")
    assert train_w.size + val_w.size == 8
