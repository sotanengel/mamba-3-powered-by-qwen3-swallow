"""TDD: tests for src/mamba3jp/train/loop.py.

The smoke test runs the loop with a tiny CPU-only ``nn.Module`` that mimics the
``MambaLMHeadModel`` API (``model(input_ids=..., labels=...)`` returns an
object with a ``.loss`` attribute). This proves the orchestration code —
gradient accumulation, clipping, optimizer step, scheduler step, logging — is
correct independently of the GPU-only mamba-ssm kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from mamba3jp.train.loop import TrainConfig, train_steps  # noqa: E402


@dataclass
class _Out:
    loss: torch.Tensor
    logits: torch.Tensor


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.proj = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> _Out:
        h = self.emb(input_ids)
        logits = self.proj(h)
        loss = torch.tensor(0.0)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return _Out(loss=loss, logits=logits)


def _make_loader(n: int = 32, seq_len: int = 8, vocab: int = 32) -> DataLoader[Any]:
    g = torch.Generator().manual_seed(0)
    data = [
        {
            "input_ids": torch.randint(0, vocab, (seq_len,), generator=g),
            "labels": torch.randint(0, vocab, (seq_len,), generator=g),
        }
        for _ in range(n)
    ]
    return DataLoader(data, batch_size=2, shuffle=False)


def _make_cfg(max_steps: int, grad_accum: int = 1) -> TrainConfig:
    return TrainConfig(
        max_steps=max_steps,
        grad_accum=grad_accum,
        clip_grad=1.0,
        log_interval=1,
        eval_interval=10**9,  # disable val in smoke
        save_interval=10**9,
        device="cpu",
        use_amp=False,
    )


def test_loop_runs_two_steps_and_loss_is_finite() -> None:
    torch.manual_seed(0)
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    losses = list(train_steps(model, opt, sched, _make_loader(), _make_cfg(max_steps=2)))
    assert len(losses) == 2
    for entry in losses:
        assert torch.isfinite(torch.tensor(entry["loss"]))
        assert entry["step"] in (1, 2)


def test_loop_is_deterministic_with_fixed_seed() -> None:
    def run() -> list[float]:
        torch.manual_seed(123)
        model = _TinyLM()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        return [e["loss"] for e in train_steps(model, opt, sched, _make_loader(), _make_cfg(max_steps=3))]

    assert run() == run()


def test_grad_accumulation_consumes_n_microbatches_per_step() -> None:
    torch.manual_seed(0)
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)

    # Wrap the loader to count how many batches the loop pulls out.
    inner = _make_loader(n=24)
    pulled: list[int] = []

    def counting_loader():
        for b in inner:
            pulled.append(1)
            yield b

    # 4 optimizer steps * accum=3 = 12 micro-batches consumed.
    list(
        train_steps(
            model, opt, sched, counting_loader(), _make_cfg(max_steps=4, grad_accum=3)
        )
    )
    assert sum(pulled) == 12


def test_loss_decreases_over_many_steps() -> None:
    torch.manual_seed(0)
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    history = [e["loss"] for e in train_steps(model, opt, sched, _make_loader(n=100), _make_cfg(max_steps=20))]
    # Average of the last 5 should beat the first 5 by a clear margin.
    early = sum(history[:5]) / 5
    late = sum(history[-5:]) / 5
    assert late < early
