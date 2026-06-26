"""joryu パイプライン出力 (raw / export / curated) を抽象化する Bundle 層。

joryu は 3 種の形態でデータを提供する:

- ``raw``     : 単一の ``responses.jsonl`` または ``responses.jsonl.zst``
- ``export``  : ``responses.jsonl.zst`` + ``meta.json`` + ``SHA256SUMS`` のディレクトリ
- ``curated`` : ``responses.high_quality.jsonl`` (+ ``scores.jsonl``) のディレクトリ

本モジュールはどの形態か判別 (``discover_bundle``) し、必要に応じて整合性検証
(``verify_sha256``) とスコア結合 (``load_scores``) を提供する。

スコア結合キーは joryu の ``record_hash`` (sha256-prefix 付き)。同一定義の
``compute_record_hash`` を本リポジトリ側でも持ち、curated 由来でない raw 側
レコードからも同じハッシュを算出して結合できるようにする。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BundleKind = Literal["raw", "export", "curated"]


class BundleIntegrityError(RuntimeError):
    """SHA256SUMS と実ファイルのハッシュが食い違ったときに送出される。"""


@dataclass(frozen=True)
class BundleSpec:
    """データ供給形態の正規化された記述子。"""

    kind: BundleKind
    responses_path: Path
    scores_path: Path | None
    meta_path: Path | None
    sha256_path: Path | None
    source_root: Path


@dataclass(frozen=True)
class ScoreEntry:
    """``scores.jsonl`` 1 行を 1 オブジェクトとして保持する。"""

    record_hash: str
    final_score: float
    accepted: bool
    rejected_by: list[str]
    signal_scores: dict[str, float]


# ---- record hash (joryu の compute_record_hash と互換) ----------------------


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def compute_record_hash(record: dict[str, Any]) -> str:
    """joryu の ``src/joryu/curate/record_hash.py`` と bit-identical なハッシュ。

    ``sha256(prompt || answer || mode || sampling || system_prompt || config_hash)``
    に thinking モード時は ``thinking_trace`` を追加する。
    """
    payload: dict[str, Any] = {
        "prompt": record.get("prompt", ""),
        "answer": record.get("answer", ""),
        "mode": record.get("mode", ""),
        "sampling": _normalize(record.get("sampling", {})),
        "system_prompt": record.get("system_prompt", ""),
        "config_hash": record.get("config_hash", ""),
    }
    if record.get("mode") == "thinking":
        payload["thinking_trace"] = record.get("thinking_trace", "") or record.get("reasoning", "")
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "sha256-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- discovery --------------------------------------------------------------


_RAW_SUFFIXES = (".jsonl", ".jsonl.zst")


def _has_raw_suffix(p: Path) -> bool:
    name = p.name.lower()
    return any(name.endswith(s) for s in _RAW_SUFFIXES)


def discover_bundle(path: Path) -> BundleSpec:
    """``path`` が raw / export / curated のいずれであるかを判別する。

    判別不能なディレクトリは ``ValueError`` を送出する。存在しないパスは
    ``FileNotFoundError`` を送出する。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"bundle path does not exist: {path}")

    if path.is_file():
        if not _has_raw_suffix(path):
            raise ValueError(f"bundle file not recognized (need .jsonl or .jsonl.zst): {path}")
        return BundleSpec(
            kind="raw",
            responses_path=path,
            scores_path=None,
            meta_path=None,
            sha256_path=None,
            source_root=path,
        )

    # directory case
    sha = path / "SHA256SUMS"
    meta = path / "meta.json"
    export_payload = path / "responses.jsonl.zst"
    curated_payload = path / "responses.high_quality.jsonl"
    scores = path / "scores.jsonl"

    if sha.exists() and export_payload.exists():
        return BundleSpec(
            kind="export",
            responses_path=export_payload,
            scores_path=scores if scores.exists() else None,
            meta_path=meta if meta.exists() else None,
            sha256_path=sha,
            source_root=path,
        )

    if curated_payload.exists():
        return BundleSpec(
            kind="curated",
            responses_path=curated_payload,
            scores_path=scores if scores.exists() else None,
            meta_path=meta if meta.exists() else None,
            sha256_path=sha if sha.exists() else None,
            source_root=path,
        )

    raise ValueError(f"bundle directory not recognized (no SHA256SUMS / curated payload): {path}")


# ---- sha256 verification ----------------------------------------------------


def _parse_sha256sums(text: str) -> dict[str, str]:
    """coreutils 互換の ``<hex>  <filename>`` 行を辞書に変換する。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # filename 側に空白が含まれる可能性は実運用で無いが split は安全側に倒す
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, fname = parts[0], parts[1].lstrip("*").strip()
        out[fname] = digest.lower()
    return out


def verify_sha256(spec: BundleSpec) -> dict[str, str]:
    """``SHA256SUMS`` を読み、列挙された各ファイルのハッシュ一致を検証する。

    不一致時には :class:`BundleIntegrityError` を送出。``sha256_path`` が未設定
    の bundle (raw / sha256sums なしの curated) では ``{}`` を返す (no-op)。
    """
    if spec.sha256_path is None or not spec.sha256_path.exists():
        return {}
    expected = _parse_sha256sums(spec.sha256_path.read_text(encoding="utf-8"))
    if not expected:
        return {}

    root = spec.source_root if spec.source_root.is_dir() else spec.source_root.parent
    mismatches: list[str] = []
    for fname, want in expected.items():
        target = root / fname
        if not target.exists():
            mismatches.append(f"{fname}: missing")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != want:
            mismatches.append(f"{fname}: expected {want[:12]}..., got {actual[:12]}...")
    if mismatches:
        raise BundleIntegrityError("SHA256SUMS mismatch: " + "; ".join(mismatches))
    return expected


# ---- scores -----------------------------------------------------------------


def iter_bundle(
    spec: BundleSpec,
    scores: dict[str, ScoreEntry] | None = None,
) -> Iterator[tuple[dict[str, Any], ScoreEntry | None]]:
    """Bundle 内の各レコードに ``ScoreEntry`` を左結合して順に yield する。

    結合キーは :func:`compute_record_hash`。``scores`` が ``None`` の場合は
    spec.scores_path から自動ロードする (存在しない場合は ``{}``)。
    """
    from mamba3jp.data.ingest import iter_records  # 遅延 import で循環回避

    if scores is None:
        scores = load_scores(spec.scores_path) if spec.scores_path else {}
    for rec in iter_records(spec.responses_path):
        rec_hash = compute_record_hash(rec)
        yield rec, scores.get(rec_hash)


# Bundle 種別の優先度 (大きいほど優先)。同一 record_hash の衝突時は高優先側を採用。
_KIND_PRIORITY: dict[BundleKind, int] = {"raw": 0, "export": 1, "curated": 2}


def merge_sources(
    specs: list[BundleSpec],
) -> Iterator[tuple[dict[str, Any], ScoreEntry | None, BundleKind]]:
    """複数 BundleSpec を ``record_hash`` で統合し、各レコードを一度だけ yield する。

    優先順位: ``curated > export > raw``。同一 record_hash が複数 source に存在
    する場合は優先度の高い source のレコード dict と kind を採用し、スコアも同じ
    source 側を優先する (curated にスコアがあればそれを、無ければ低優先側のスコアを保持)。

    yield 順は specs 引数の出現順に従って初出を保つ。yield 値は
    ``(record, score_or_None, winning_kind)`` の 3-tuple。
    """
    chosen: dict[str, tuple[int, BundleKind, dict[str, Any], ScoreEntry | None]] = {}
    order: list[str] = []
    for spec in specs:
        prio = _KIND_PRIORITY.get(spec.kind, 0)
        for rec, score in iter_bundle(spec):
            rec_hash = compute_record_hash(rec)
            if rec_hash not in chosen:
                chosen[rec_hash] = (prio, spec.kind, rec, score)
                order.append(rec_hash)
                continue
            existing_prio, existing_kind, existing_rec, existing_score = chosen[rec_hash]
            if prio > existing_prio:
                new_score = score if score is not None else existing_score
                chosen[rec_hash] = (prio, spec.kind, rec, new_score)
            elif prio == existing_prio:
                new_score = existing_score if existing_score is not None else score
                chosen[rec_hash] = (existing_prio, existing_kind, existing_rec, new_score)
            elif existing_score is None and score is not None:
                chosen[rec_hash] = (existing_prio, existing_kind, existing_rec, score)
    for rec_hash in order:
        _, kind, rec, score = chosen[rec_hash]
        yield rec, score, kind


def load_scores(path: Path) -> dict[str, ScoreEntry]:
    """``scores.jsonl`` を ``record_hash`` をキーとする辞書に展開する。

    ``record_hash`` を欠く行は静かに無視する (curate 側の旧バージョン互換)。
    """
    out: dict[str, ScoreEntry] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_hash = row.get("record_hash")
            if not isinstance(rec_hash, str):
                continue
            try:
                final_score = float(row.get("final_score", 0.0))
            except (TypeError, ValueError):
                continue
            rejected_by_raw = row.get("rejected_by") or []
            rejected_by = [str(x) for x in rejected_by_raw] if isinstance(rejected_by_raw, list) else []
            signal_raw = row.get("signal_scores") or {}
            signal_scores = (
                {str(k): float(v) for k, v in signal_raw.items() if isinstance(v, (int, float))}
                if isinstance(signal_raw, dict)
                else {}
            )
            out[rec_hash] = ScoreEntry(
                record_hash=rec_hash,
                final_score=final_score,
                accepted=bool(row.get("accepted", False)),
                rejected_by=rejected_by,
                signal_scores=signal_scores,
            )
    return out
