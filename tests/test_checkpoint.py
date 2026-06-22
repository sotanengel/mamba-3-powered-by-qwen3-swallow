"""TDD: tests for src/mamba3jp/train/checkpoint.py.

All tests run on CPU with a tiny ``nn.Linear`` so the suite stays fast.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mamba3jp.train.checkpoint import (  # noqa: E402
    load_ckpt,
    rotate,
    save_ckpt,
)


def _make_tiny_state() -> tuple[torch.nn.Module, torch.optim.Optimizer, object]:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # take a step so opt state is populated
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    return model, opt, sched


def test_save_and_load_roundtrip_model_state(tmp_path: Path) -> None:
    model, opt, sched = _make_tiny_state()
    path = tmp_path / "ckpt.pt"
    save_ckpt(path, model=model, optimizer=opt, scheduler=sched, step=100, val_loss=1.23)

    ck = load_ckpt(path)

    new_model = torch.nn.Linear(4, 4)
    new_model.load_state_dict(ck["model"])

    for k, v in model.state_dict().items():
        assert torch.equal(v, new_model.state_dict()[k]), f"mismatch on {k}"


def test_save_and_load_roundtrip_optimizer_state(tmp_path: Path) -> None:
    model, opt, sched = _make_tiny_state()
    path = tmp_path / "ckpt.pt"
    save_ckpt(path, model=model, optimizer=opt, scheduler=sched, step=100)

    ck = load_ckpt(path)
    new_model = torch.nn.Linear(4, 4)
    new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_opt.load_state_dict(ck["optimizer"])
    assert new_opt.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_save_and_load_roundtrip_step_and_val_loss(tmp_path: Path) -> None:
    model, opt, sched = _make_tiny_state()
    path = tmp_path / "ckpt.pt"
    save_ckpt(path, model=model, optimizer=opt, scheduler=sched, step=42, val_loss=2.5)
    ck = load_ckpt(path)
    assert ck["step"] == 42
    assert ck["val_loss"] == 2.5


def test_rng_roundtrip(tmp_path: Path) -> None:
    model, opt, sched = _make_tiny_state()
    path = tmp_path / "ckpt.pt"

    # Save with the *current* RNGs.
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    save_ckpt(path, model=model, optimizer=opt, scheduler=sched, step=0)

    expected = (
        torch.rand(3).tolist(),
        np.random.rand(3).tolist(),
        [random.random() for _ in range(3)],
    )

    # Mutate RNGs, then load and check we got back the same draws.
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    ck = load_ckpt(path)

    torch.set_rng_state(ck["rng"]["torch"])
    np.random.set_state(ck["rng"]["numpy"])
    random.setstate(ck["rng"]["python"])
    got = (torch.rand(3).tolist(), np.random.rand(3).tolist(), [random.random() for _ in range(3)])
    assert got == expected


def test_atomic_write_leaves_no_tmp_files(tmp_path: Path) -> None:
    model, opt, sched = _make_tiny_state()
    path = tmp_path / "ckpt.pt"
    save_ckpt(path, model=model, optimizer=opt, scheduler=sched, step=0)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"atomic save left tmp files: {leftovers}"
    assert path.exists()


def test_rotate_keeps_newest_n(tmp_path: Path) -> None:
    model, opt, sched = _make_tiny_state()
    for step in (10, 20, 30, 40, 50):
        save_ckpt(
            tmp_path / f"step-{step}.pt",
            model=model,
            optimizer=opt,
            scheduler=sched,
            step=step,
        )
    rotate(tmp_path, keep=3)
    kept = sorted(p.name for p in tmp_path.glob("step-*.pt"))
    assert kept == ["step-30.pt", "step-40.pt", "step-50.pt"]


def test_rotate_preserves_last_and_best(tmp_path: Path) -> None:
    model, opt, sched = _make_tiny_state()
    for step in (10, 20, 30):
        save_ckpt(
            tmp_path / f"step-{step}.pt",
            model=model,
            optimizer=opt,
            scheduler=sched,
            step=step,
        )
    save_ckpt(tmp_path / "last.pt", model=model, optimizer=opt, scheduler=sched, step=30)
    save_ckpt(tmp_path / "best.pt", model=model, optimizer=opt, scheduler=sched, step=20)

    rotate(tmp_path, keep=1)

    remaining = {p.name for p in tmp_path.glob("*.pt")}
    assert "last.pt" in remaining
    assert "best.pt" in remaining
    assert "step-30.pt" in remaining
    assert "step-20.pt" not in remaining
    assert "step-10.pt" not in remaining
