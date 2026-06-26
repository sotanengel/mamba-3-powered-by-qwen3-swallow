"""Regression: CI lint must run the same pre-commit hooks developers use locally.

PR #25 failed CI on ``ruff format --check`` while pre-commit config already
included ``ruff-format``. Root cause was opt-in local hooks never running on
the authoring machine — not a missing hook. CI must invoke pre-commit so the
check path is identical and cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"


def _load_pre_commit_hooks() -> list[str]:
    data = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    hook_ids: list[str] = []
    for repo in data.get("repos", []):
        for hook in repo.get("hooks", []):
            stages = hook.get("stages")
            if stages == ["manual"]:
                continue
            hook_ids.append(hook["id"])
    return hook_ids


def test_pre_commit_config_includes_ruff_format() -> None:
    """Local hooks must cover formatting, not just lint fixes."""
    assert "ruff" in _load_pre_commit_hooks()
    assert "ruff-format" in _load_pre_commit_hooks()


def test_ci_lint_job_runs_pre_commit_all_files() -> None:
    """Lint job must delegate to pre-commit (single source of truth)."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    lint_job = workflow["jobs"]["lint"]
    steps_text = "\n".join(
        step.get("run", "") for step in lint_job["steps"] if isinstance(step, dict)
    )
    assert "pre-commit run --all-files" in steps_text, (
        "CI lint job should run `pre-commit run --all-files` so it mirrors "
        "local hooks and catches unformatted files before merge."
    )


def test_ci_lint_does_not_duplicate_ruff_invocation() -> None:
    """Avoid parallel ruff CLI paths that can drift from pre-commit rev pins."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    lint_job = workflow["jobs"]["lint"]
    steps_text = "\n".join(
        step.get("run", "") for step in lint_job["steps"] if isinstance(step, dict)
    )
    assert "ruff check" not in steps_text
    assert "ruff format" not in steps_text
