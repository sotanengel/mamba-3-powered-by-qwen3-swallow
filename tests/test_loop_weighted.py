"""TDD: train_steps の重み付き loss モード (PR-6).

``cfg.weighted_loss=True`` のとき、バッチに含まれる ``weight`` で per-sample 重み付け
平均 (weighted mean) を取る。weight が無い (= バッチが None weight) か weighted_loss=False の
ときは従来挙動 (均一 mean) と bit-equivalent。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mamba3jp.train.loop import TrainConfig, train_steps  # noqa: E402


@dataclass
class _Out:
    loss: torch.Tensor
    logits: torch.Tensor


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.proj = torch.nn.Linear(dim, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> _Out:
        h = self.emb(input_ids)
        logits = self.proj(h)
        loss = torch.tensor(0.0)
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return _Out(loss=loss, logits=logits)


def _make_weighted_loader(weights: list[float]) -> DataLoader[Any]:
    g = torch.Generator().manual_seed(0)
    data = [
        {
            "input_ids": torch.randint(0, 16, (4,), generator=g),
            "labels": torch.randint(0, 16, (4,), generator=g),
            "weight": torch.tensor(w, dtype=torch.float32),
        }
        for w in weights
    ]
    return DataLoader(data, batch_size=2, shuffle=False)


def _cfg(*, weighted: bool, steps: int = 1) -> TrainConfig:
    return TrainConfig(
        max_steps=steps,
        grad_accum=1,
        clip_grad=1.0,
        log_interval=1,
        eval_interval=10**9,
        save_interval=10**9,
        device="cpu",
        use_amp=False,
        weighted_loss=weighted,
    )


# ---- back-compat -----------------------------------------------------------


def test_weighted_loss_false_matches_existing_behavior() -> None:
    """weighted_loss=False では従来挙動 (.loss を使う) と bit-equivalent。"""
    torch.manual_seed(0)
    model_a = _TinyLM()
    opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
    sched_a = torch.optim.lr_scheduler.ConstantLR(opt_a, factor=1.0)
    losses_a = list(
        train_steps(model_a, opt_a, sched_a, _make_weighted_loader([1.0] * 8), _cfg(weighted=False))
    )

    torch.manual_seed(0)
    model_b = _TinyLM()
    opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)
    sched_b = torch.optim.lr_scheduler.ConstantLR(opt_b, factor=1.0)
    # weight 無しの DataLoader
    g = torch.Generator().manual_seed(0)
    plain = [
        {
            "input_ids": torch.randint(0, 16, (4,), generator=g),
            "labels": torch.randint(0, 16, (4,), generator=g),
        }
        for _ in range(8)
    ]
    losses_b = list(
        train_steps(model_b, opt_b, sched_b, DataLoader(plain, batch_size=2), _cfg(weighted=False))
    )
    assert losses_a[0]["loss"] == pytest.approx(losses_b[0]["loss"])


# ---- weighted mean correctness --------------------------------------------


def test_weighted_loss_matches_manual_computation() -> None:
    """weighted_loss=True の場合、loss が手計算の weighted mean と一致する。"""
    torch.manual_seed(0)
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # 学習させずに評価
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)

    weights = [2.0, 0.5]  # 2 samples のバッチ
    loader = _make_weighted_loader(weights)
    [log] = list(train_steps(model, opt, sched, loader, _cfg(weighted=True)))

    # 手計算: 同じローダから得たデータを使う (DataLoader collate と同等の順序)
    torch.manual_seed(0)
    ref_model = _TinyLM()
    # 再度ローダを作って 1 バッチ取り出す → 学習時と同じ collation
    [batch] = list(_make_weighted_loader(weights))
    with torch.no_grad():
        out = ref_model(input_ids=batch["input_ids"], labels=batch["labels"])
        per_token = F.cross_entropy(
            out.logits.reshape(-1, out.logits.size(-1)),
            batch["labels"].reshape(-1),
            reduction="none",
        ).reshape(2, 4)
        per_sample = per_token.mean(dim=1)
        w = batch["weight"]
        expected = float((per_sample * w).sum() / w.sum())
    assert log["loss"] == pytest.approx(expected, rel=1e-4)


def test_weighted_loss_is_finite_and_logs_unweighted_too() -> None:
    torch.manual_seed(0)
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    [log] = list(
        train_steps(
            model, opt, sched, _make_weighted_loader([0.5, 1.5, 0.1, 2.0]), _cfg(weighted=True)
        )
    )
    assert torch.isfinite(torch.tensor(log["loss"]))
    # 重み付きと無印の両方をログに出す
    assert "loss_unweighted" in log
    assert torch.isfinite(torch.tensor(log["loss_unweighted"]))
