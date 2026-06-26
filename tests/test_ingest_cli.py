"""TDD: scripts/ingest_joryu.py の subprocess 統合テスト (PR-3).

curated / export / raw 各形態を一回ずつ subprocess で実行し、
- 中間 JSONL に meta が乗ること
- manifest_partial.json が出ること
- --min-score でフィルタが効くこと
を確認する。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mamba3jp.data.bundles import compute_record_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ingest_joryu.py"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_zstd(path: Path, records: list[dict]) -> None:
    import zstandard as zstd

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor().compress(raw))


def _samples() -> list[dict]:
    return [
        {
            "prompt": "日本の四季を教えて。",
            "answer": "春夏秋冬があります。",
            "mode": "thinking",
            "sampling": {"temperature": 0.6},
            "system_prompt": "丁寧に。",
            "config_hash": "sha256-cfgA",
            "thinking_trace": "簡単に整理する。",
            "style_id": "polite",
            "category": "知識",
        },
        {
            "prompt": "もう一つの質問。",
            "answer": "もう一つの回答。",
            "mode": "nothinking",
            "sampling": {"temperature": 0.8},
            "system_prompt": "フランクに。",
            "config_hash": "sha256-cfgB",
            "thinking_trace": None,
            "style_id": "casual",
            "category": "実用",
        },
    ]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env={**env, **dict(__import__("os").environ)},
    )


# ---- raw -------------------------------------------------------------------


def test_raw_jsonl_writes_meta_enriched_output(tmp_path: Path) -> None:
    src = tmp_path / "responses.jsonl"
    _write_jsonl(src, _samples())
    out = tmp_path / "out" / "chatml.jsonl"
    stats = tmp_path / "out" / "stats.json"
    _run(["--input", str(src), "--output", str(out), "--stats", str(stats)])

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    for obj in lines:
        assert "text" in obj and "meta" in obj
        assert obj["meta"]["source_kind"] == "raw"
        assert obj["meta"]["final_score"] is None
        assert obj["meta"]["style_id"] in {"polite", "casual"}

    manifest = json.loads((out.parent / "manifest_partial.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["kind"] == "raw"
    assert manifest["records_in"] == 2
    assert manifest["records_out"] == 2
    assert set(manifest["config_hashes"]) == {"sha256-cfgA", "sha256-cfgB"}


# ---- export ----------------------------------------------------------------


def test_export_bundle_verifies_sha256_and_filters_by_score(tmp_path: Path) -> None:
    bundle = tmp_path / "20260621T010000Z"
    recs = _samples()
    _write_zstd(bundle / "responses.jsonl.zst", recs)
    score_rows = [
        {"record_hash": compute_record_hash(recs[0]), "final_score": 0.95, "accepted": True},
        {"record_hash": compute_record_hash(recs[1]), "final_score": 0.30, "accepted": False},
    ]
    _write_jsonl(bundle / "scores.jsonl", score_rows)
    (bundle / "meta.json").write_text(json.dumps({"record_count": 2}), encoding="utf-8")
    payload = (bundle / "responses.jsonl.zst").read_bytes()
    meta_payload = (bundle / "meta.json").read_bytes()
    scores_payload = (bundle / "scores.jsonl").read_bytes()
    (bundle / "SHA256SUMS").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  responses.jsonl.zst\n"
        f"{hashlib.sha256(meta_payload).hexdigest()}  meta.json\n"
        f"{hashlib.sha256(scores_payload).hexdigest()}  scores.jsonl\n",
        encoding="utf-8",
    )

    out = tmp_path / "out" / "chatml.jsonl"
    stats = tmp_path / "out" / "stats.json"
    _run(
        [
            "--bundle",
            str(bundle),
            "--output",
            str(out),
            "--stats",
            str(stats),
            "--min-score",
            "0.7",
        ]
    )

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    # 低スコアのレコードはドロップされ、高スコアのみ残る
    assert len(lines) == 1
    assert lines[0]["meta"]["final_score"] == pytest.approx(0.95)
    assert lines[0]["meta"]["source_kind"] == "export"

    stat = json.loads(stats.read_text(encoding="utf-8"))
    assert stat["skip_by_reason"].get("low_score") == 1


# ---- curated ---------------------------------------------------------------


def test_curated_bundle_auto_skips_quality_filter(tmp_path: Path) -> None:
    bundle = tmp_path / "20260622_000000"
    # curated は通常なら REPEATED_CHARS で落ちるレコードを accepted として持つ
    high_rec = {
        "prompt": "繰り返し許容のテスト。",
        "answer": "ああああああああああ",  # raw フィルタなら REPEATED_CHARS
        "mode": "nothinking",
        "sampling": {"temperature": 0.6},
        "system_prompt": "",
        "config_hash": "sha256-cfgC",
        "thinking_trace": None,
        "style_id": "polite",
        "category": "テスト",
    }
    _write_jsonl(bundle / "responses.high_quality.jsonl", [high_rec])
    _write_jsonl(
        bundle / "scores.jsonl",
        [{"record_hash": compute_record_hash(high_rec), "final_score": 0.85, "accepted": True}],
    )

    out = tmp_path / "out" / "chatml.jsonl"
    stats = tmp_path / "out" / "stats.json"
    _run(["--bundle", str(bundle), "--output", str(out), "--stats", str(stats)])

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["meta"]["source_kind"] == "curated"
    assert lines[0]["meta"]["final_score"] == pytest.approx(0.85)


# ---- sources-yaml merge ----------------------------------------------------


def test_sources_yaml_merges_raw_and_curated(tmp_path: Path) -> None:
    """raw に 2 件、curated に 1 件 (raw と同一) が入っているとき、
    マージ結果は 2 件で、curated 由来の方は final_score を持つこと。"""
    recs = _samples()
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, recs)
    curated_dir = tmp_path / "curated_x"
    _write_jsonl(curated_dir / "responses.high_quality.jsonl", [recs[0]])
    _write_jsonl(
        curated_dir / "scores.jsonl",
        [{"record_hash": compute_record_hash(recs[0]), "final_score": 0.88, "accepted": True}],
    )

    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        "sources:\n"
        f"  - {{ kind: raw, path: {raw_path.as_posix()} }}\n"
        f"  - {{ kind: curated, path: {curated_dir.as_posix()} }}\n",
        encoding="utf-8",
    )

    out = tmp_path / "out" / "chatml.jsonl"
    stats = tmp_path / "out" / "stats.json"
    _run(["--sources-yaml", str(sources_yaml), "--output", str(out), "--stats", str(stats)])

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    # 2 records merged (rec_0 from curated, rec_1 from raw)
    assert len(lines) == 2
    by_kind = {obj["meta"]["source_kind"]: obj for obj in lines}
    assert set(by_kind) == {"curated", "raw"}
    assert by_kind["curated"]["meta"]["final_score"] == pytest.approx(0.88)
    assert by_kind["raw"]["meta"]["final_score"] is None

    manifest = json.loads((out.parent / "manifest_partial.json").read_text(encoding="utf-8"))
    kinds = {s["kind"] for s in manifest["sources"]}
    assert kinds == {"raw", "curated"}
    assert manifest["by_source_kind"] == {"raw": 1, "curated": 1}


# ---- sha256 mismatch -------------------------------------------------------


def test_export_bundle_sha256_mismatch_fails(tmp_path: Path) -> None:
    bundle = tmp_path / "20260621T020000Z"
    recs = _samples()
    _write_zstd(bundle / "responses.jsonl.zst", recs)
    (bundle / "meta.json").write_text("{}", encoding="utf-8")
    (bundle / "SHA256SUMS").write_text(
        "0" * 64 + "  responses.jsonl.zst\n" + hashlib.sha256(b"{}").hexdigest() + "  meta.json\n",
        encoding="utf-8",
    )
    out = tmp_path / "out" / "chatml.jsonl"
    stats = tmp_path / "out" / "stats.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--bundle",
            str(bundle),
            "--output",
            str(out),
            "--stats",
            str(stats),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode != 0
    assert "SHA256SUMS mismatch" in (proc.stderr + proc.stdout)
