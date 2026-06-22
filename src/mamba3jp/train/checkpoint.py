"""Atomic checkpoint save/load with full RNG capture and step-* rotation."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_STEP_RE = re.compile(r"^step-(\d+)\.pt$")


def _capture_rng() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def save_ckpt(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    step: int,
    val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a checkpoint to ``path``.

    The on-disk format is a plain ``torch.save`` dict::

        {
            "model": state_dict,
            "optimizer": state_dict,
            "scheduler": state_dict | None,
            "step": int,
            "val_loss": float | None,
            "rng": {"torch": ..., "numpy": ..., "python": ...},
            "extra": dict,
        }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "rng": _capture_rng(),
        "extra": dict(extra) if extra else {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_ckpt(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint produced by :func:`save_ckpt`.

    Returns the raw payload dict so the caller can restore state into whatever
    model/optimizer/scheduler instances they have.
    """
    # ``weights_only=False`` is required because we serialize the RNG state and
    # python ``random`` state, which torch.load 2.6+ refuses to unpickle in the
    # default weights-only mode.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def rotate(directory: str | Path, *, keep: int = 3, keep_best: bool = True) -> list[Path]:
    """Delete the oldest ``step-*.pt`` files in ``directory`` until ``keep`` remain.

    ``last.pt`` and ``best.pt`` are always preserved if they exist.
    Returns the list of removed paths.
    """
    directory = Path(directory)
    candidates: list[tuple[int, Path]] = []
    for p in directory.glob("step-*.pt"):
        m = _STEP_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    candidates.sort(key=lambda kv: kv[0])

    removed: list[Path] = []
    while len(candidates) > keep:
        _, victim = candidates.pop(0)
        victim.unlink()
        removed.append(victim)

    # ``last.pt`` and ``best.pt`` are never touched by this routine; documented
    # here so future readers don't add them to the glob above.
    _ = keep_best  # kept for API stability; logic above is the implementation
    return removed
