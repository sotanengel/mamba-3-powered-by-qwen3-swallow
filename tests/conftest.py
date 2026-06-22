"""Shared pytest configuration.

GPU-bound tests are marked ``@pytest.mark.gpu`` and skipped by default. Pass
``--run-gpu`` to opt in (typically inside the docker container with a GPU).
Likewise for ``slow`` tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.gpu",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu to enable")
    skip_slow = pytest.mark.skip(reason="needs --run-slow to enable")
    run_gpu = config.getoption("--run-gpu")
    run_slow = config.getoption("--run-slow")
    for item in items:
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def sample_joryu_path() -> Path:
    """Path to the bundled tiny joryu-format JSONL used by ingest tests."""
    return Path(__file__).parent / "data" / "sample_joryu.jsonl"


@pytest.fixture
def sample_joryu_records(sample_joryu_path: Path) -> Iterator[dict[str, object]]:
    """Yield each record from the bundled sample JSONL."""
    with sample_joryu_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
