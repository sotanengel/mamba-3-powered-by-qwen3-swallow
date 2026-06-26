"""TDD: tests for src/mamba3jp/data/bundles.py.

joryu パイプラインの 3 種のデータ供給形態を抽象化する BundleSpec / 探索 /
sha256 検証 / scores.jsonl ロードを担う。実装前に書かれており、
``pytest tests/test_bundles.py`` は実装が揃うまで失敗する。

サポートする 3 形態:
1. ``raw``     : 単一の ``responses.jsonl`` または ``responses.jsonl.zst``
2. ``export``  : ``responses.jsonl.zst`` + ``meta.json`` + ``SHA256SUMS`` を含むディレクトリ
3. ``curated`` : ``responses.high_quality.jsonl`` (+ ``scores.jsonl``) を含むディレクトリ

scores.jsonl の結合キーは joryu の ``record_hash`` (sha256-prefixed)。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mamba3jp.data.bundles import (
    BundleIntegrityError,
    BundleSpec,
    ScoreEntry,
    compute_record_hash,
    discover_bundle,
    load_scores,
    verify_sha256,
)

# ---- helpers ----------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_zstd_jsonl(path: Path, records: list[dict]) -> None:
    import zstandard as zstd

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor().compress(raw))


def _write_sha256sums(directory: Path, filenames: list[str]) -> None:
    lines = []
    for fname in filenames:
        digest = hashlib.sha256((directory / fname).read_bytes()).hexdigest()
        lines.append(f"{digest}  {fname}")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_records() -> list[dict]:
    return [
        {
            "prompt": "日本の四季について教えて。",
            "answer": "春夏秋冬があります。",
            "mode": "thinking",
            "sampling": {"temperature": 0.6, "top_p": 0.95},
            "system_prompt": "丁寧に答えてください。",
            "config_hash": "sha256-cfg01",
            "thinking_trace": "簡単にまとめる。",
            "style_id": "polite",
            "category": "知識",
        },
        {
            "prompt": "ラーメンの作り方は？",  # noqa: RUF001 — Japanese full-width question mark
            "answer": "麺を茹でてスープに乗せます。",
            "mode": "nothinking",
            "sampling": {"temperature": 0.8, "top_p": 0.9},
            "system_prompt": "フランクに。",
            "config_hash": "sha256-cfg02",
            "thinking_trace": None,
            "style_id": "casual",
            "category": "実用",
        },
    ]


def _build_export_bundle(directory: Path, records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _write_zstd_jsonl(directory / "responses.jsonl.zst", records)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "record_count": len(records),
                "model": "Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
                "time_range": ["2026-06-21T00:00:00Z", "2026-06-21T01:00:00Z"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_sha256sums(directory, ["responses.jsonl.zst", "meta.json"])
    return directory


def _build_curated_bundle(
    directory: Path, accepted: list[dict], rejected: list[dict], scores: list[dict]
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "responses.high_quality.jsonl", accepted)
    _write_jsonl(directory / "responses.rejected.jsonl", rejected)
    _write_jsonl(directory / "scores.jsonl", scores)
    return directory


# ---- compute_record_hash ----------------------------------------------------


def test_compute_record_hash_is_deterministic() -> None:
    rec = _sample_records()[0]
    h1 = compute_record_hash(rec)
    h2 = compute_record_hash(dict(rec))
    assert h1 == h2
    assert h1.startswith("sha256-")
    # 64 hex chars after prefix
    assert len(h1) == len("sha256-") + 64


def test_compute_record_hash_changes_with_answer() -> None:
    rec = _sample_records()[0]
    other = dict(rec)
    other["answer"] = rec["answer"] + "!"
    assert compute_record_hash(rec) != compute_record_hash(other)


def test_compute_record_hash_includes_thinking_trace_only_in_thinking_mode() -> None:
    base = dict(_sample_records()[0])  # mode == thinking
    # 思考モードでは thinking_trace の差がハッシュを変える
    other = dict(base)
    other["thinking_trace"] = "別の思考"
    assert compute_record_hash(base) != compute_record_hash(other)

    # nothinking モードでは thinking_trace を無視する
    base_nt = dict(base, mode="nothinking", thinking_trace=None)
    other_nt = dict(base_nt, thinking_trace="無視されるはず")
    assert compute_record_hash(base_nt) == compute_record_hash(other_nt)


def test_compute_record_hash_sampling_normalized_by_key_order() -> None:
    rec = _sample_records()[0]
    permuted = dict(rec)
    # dict key 順序を変えても安定なはず
    permuted["sampling"] = {"top_p": 0.95, "temperature": 0.6}
    assert compute_record_hash(rec) == compute_record_hash(permuted)


# ---- discover_bundle: raw ---------------------------------------------------


def test_discover_raw_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "responses.jsonl"
    _write_jsonl(p, _sample_records())
    spec = discover_bundle(p)
    assert isinstance(spec, BundleSpec)
    assert spec.kind == "raw"
    assert spec.responses_path == p
    assert spec.scores_path is None
    assert spec.meta_path is None
    assert spec.sha256_path is None


def test_discover_raw_jsonl_zst(tmp_path: Path) -> None:
    p = tmp_path / "responses.jsonl.zst"
    _write_zstd_jsonl(p, _sample_records())
    spec = discover_bundle(p)
    assert spec.kind == "raw"
    assert spec.responses_path == p


# ---- discover_bundle: export -------------------------------------------------


def test_discover_export_directory(tmp_path: Path) -> None:
    bundle = _build_export_bundle(tmp_path / "20260621T000000Z", _sample_records())
    spec = discover_bundle(bundle)
    assert spec.kind == "export"
    assert spec.responses_path == bundle / "responses.jsonl.zst"
    assert spec.meta_path == bundle / "meta.json"
    assert spec.sha256_path == bundle / "SHA256SUMS"
    assert spec.scores_path is None


# ---- discover_bundle: curated -----------------------------------------------


def test_discover_curated_directory_with_scores(tmp_path: Path) -> None:
    recs = _sample_records()
    score_rows = [
        {
            "record_hash": compute_record_hash(recs[0]),
            "final_score": 0.85,
            "accepted": True,
            "rejected_by": [],
        }
    ]
    bundle = _build_curated_bundle(
        tmp_path / "20260622_000000", accepted=[recs[0]], rejected=[recs[1]], scores=score_rows
    )
    spec = discover_bundle(bundle)
    assert spec.kind == "curated"
    assert spec.responses_path == bundle / "responses.high_quality.jsonl"
    assert spec.scores_path == bundle / "scores.jsonl"
    assert spec.meta_path is None
    assert spec.sha256_path is None


def test_discover_curated_without_scores_optional(tmp_path: Path) -> None:
    recs = _sample_records()
    bundle = tmp_path / "20260622_111111"
    bundle.mkdir(parents=True)
    _write_jsonl(bundle / "responses.high_quality.jsonl", [recs[0]])
    # scores.jsonl 不在でも curated と認識できる
    spec = discover_bundle(bundle)
    assert spec.kind == "curated"
    assert spec.scores_path is None


# ---- discover_bundle: error paths -------------------------------------------


def test_discover_unrecognized_path_raises(tmp_path: Path) -> None:
    odd = tmp_path / "mystery"
    odd.mkdir()
    with pytest.raises(ValueError, match="not recognized"):
        discover_bundle(odd)


def test_discover_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_bundle(tmp_path / "does_not_exist.jsonl")


# ---- verify_sha256 ----------------------------------------------------------


def test_verify_sha256_matches(tmp_path: Path) -> None:
    bundle = _build_export_bundle(tmp_path / "20260621T010000Z", _sample_records())
    spec = discover_bundle(bundle)
    result = verify_sha256(spec)
    assert "responses.jsonl.zst" in result
    assert "meta.json" in result
    # returned hashes must be 64-hex sha256 digests
    for hex_digest in result.values():
        assert len(hex_digest) == 64
        int(hex_digest, 16)


def test_verify_sha256_raises_on_tamper(tmp_path: Path) -> None:
    bundle = _build_export_bundle(tmp_path / "20260621T020000Z", _sample_records())
    # tamper with the payload after SHA256SUMS is computed
    (bundle / "responses.jsonl.zst").write_bytes(b"corrupted-bytes")
    spec = discover_bundle(bundle)
    with pytest.raises(BundleIntegrityError, match=r"responses\.jsonl\.zst"):
        verify_sha256(spec)


def test_verify_sha256_noop_for_raw(tmp_path: Path) -> None:
    p = tmp_path / "responses.jsonl"
    _write_jsonl(p, _sample_records())
    spec = discover_bundle(p)
    # No SHA256SUMS for raw; verify_sha256 must return {} without raising.
    assert verify_sha256(spec) == {}


def test_verify_sha256_noop_for_curated_without_sha256sums(tmp_path: Path) -> None:
    recs = _sample_records()
    bundle = _build_curated_bundle(
        tmp_path / "20260622_222222", accepted=[recs[0]], rejected=[], scores=[]
    )
    spec = discover_bundle(bundle)
    assert verify_sha256(spec) == {}


# ---- load_scores -----------------------------------------------------------


def test_load_scores_keyed_by_record_hash(tmp_path: Path) -> None:
    recs = _sample_records()
    score_rows = [
        {
            "record_hash": compute_record_hash(recs[0]),
            "final_score": 0.92,
            "accepted": True,
            "rejected_by": [],
            "signal_scores": {"LEN-A": 1.0},
        },
        {
            "record_hash": compute_record_hash(recs[1]),
            "final_score": 0.41,
            "accepted": False,
            "rejected_by": ["LEN-A"],
            "signal_scores": {"LEN-A": 0.0},
        },
    ]
    p = tmp_path / "scores.jsonl"
    _write_jsonl(p, score_rows)
    scores = load_scores(p)
    assert set(scores.keys()) == {compute_record_hash(recs[0]), compute_record_hash(recs[1])}
    entry = scores[compute_record_hash(recs[0])]
    assert isinstance(entry, ScoreEntry)
    assert entry.final_score == pytest.approx(0.92)
    assert entry.accepted is True
    assert entry.rejected_by == []


def test_load_scores_skips_malformed_rows(tmp_path: Path) -> None:
    p = tmp_path / "scores.jsonl"
    # 1 行目: 正常、2 行目: record_hash 欠落で skip
    p.write_text(
        json.dumps({"record_hash": "sha256-ok", "final_score": 0.5, "accepted": True}) + "\n"
        + json.dumps({"final_score": 0.5}) + "\n",
        encoding="utf-8",
    )
    scores = load_scores(p)
    assert list(scores.keys()) == ["sha256-ok"]
