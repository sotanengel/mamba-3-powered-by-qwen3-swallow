"""Verify pytest markers behave as documented in conftest."""

from __future__ import annotations

import pytest


def test_default_collection_runs() -> None:
    assert True


@pytest.mark.gpu
def test_gpu_marker_skipped_by_default() -> None:  # pragma: no cover - skipped
    raise AssertionError("this test must be skipped without --run-gpu")


@pytest.mark.slow
def test_slow_marker_skipped_by_default() -> None:  # pragma: no cover - skipped
    raise AssertionError("this test must be skipped without --run-slow")
