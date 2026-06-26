"""Dataset 成果物 (train.bin / val.bin / *.weights.npy) の同梱 manifest を書く。

manifest.json は学習スクリプト & 再現実験のためのデータセット版管理。
ingest 段の ``manifest_partial.json`` と tokenize 段の追加情報をマージしたものを
出力する想定。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _git_sha(repo: Path | None = None) -> str | None:
    """``git rev-parse HEAD`` の出力を返す。取得失敗時は ``None``。"""
    cwd = str(repo) if repo else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def write_manifest(out_dir: Path, parts: list[dict[str, Any]]) -> Path:
    """``out_dir/manifest.json`` を書き出して書き込み先パスを返す。

    複数の dict 断片を浅く merge する (キー衝突時は後勝ち)。git sha が取得できれば
    付与し、取得不能なら ``None`` を入れる。
    """
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    merged.setdefault("git_sha", _git_sha(out_dir))
    target = out_dir / "manifest.json"
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
