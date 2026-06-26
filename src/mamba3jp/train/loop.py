"""Causal-LM training loop.

The loop is deliberately framework-light: it consumes any optimizer / scheduler
/ DataLoader, and any ``model`` whose ``forward(input_ids, labels)`` returns an
object with a ``.loss`` attribute. That contract is satisfied by both the real
``MambaLMHeadModel`` and our tiny test stub, which is what lets the smoke test
run on CPU without mamba-ssm installed.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    max_steps: int
    grad_accum: int = 1
    clip_grad: float = 1.0
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    device: str = "cuda"
    # bf16 autocast on CUDA; disabled on CPU (and for unit tests).
    use_amp: bool = True
    # PR-6: バッチに含まれる "weight" を per-sample 重みとして使い、
    # weighted-mean CE loss を計算する。weight が無いバッチや False のときは
    # 既存挙動 (model.loss をそのまま使う) と bit-equivalent。
    weighted_loss: bool = False


def _autocast_ctx(cfg: TrainConfig) -> Any:
    if not cfg.use_amp or cfg.device == "cpu":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _move(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _microbatches(loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("DataLoader produced no batches")


def train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    loader: DataLoader[Any] | Iterable[dict[str, torch.Tensor]],
    cfg: TrainConfig,
) -> Iterator[dict[str, float]]:
    """Run ``cfg.max_steps`` optimizer steps and yield one log dict per step.

    Yielding (instead of looping internally + calling wandb) is what makes this
    function easy to unit-test: the smoke test just collects the iterator. The
    higher-level ``scripts/train.py`` wraps it with wandb logging, eval, and
    checkpointing.
    """
    model.to(cfg.device).train()
    micro_iter = _microbatches(loader)
    step = 0
    while step < cfg.max_steps:
        step += 1
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        loss_unweighted_accum = 0.0
        n_tokens = 0
        t0 = time.perf_counter()
        for _ in range(cfg.grad_accum):
            batch = _move(next(micro_iter), cfg.device)
            weight = batch.get("weight") if cfg.weighted_loss else None
            loss: torch.Tensor
            with _autocast_ctx(cfg):
                out = model(input_ids=batch["input_ids"], labels=batch["labels"])
                if weight is not None:
                    loss_weighted, loss_unweighted = _weighted_loss(
                        out.logits, batch["labels"], weight
                    )
                    loss = loss_weighted / cfg.grad_accum
                    loss_unweighted_accum += float(loss_unweighted.detach())
                else:
                    loss = out.loss / cfg.grad_accum
                    loss_unweighted_accum += float(out.loss.detach())
            loss.backward()  # type: ignore[no-untyped-call]
            loss_accum += float(loss.detach()) * cfg.grad_accum
            n_tokens += batch["input_ids"].numel()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad))
        optimizer.step()
        scheduler.step()

        elapsed = time.perf_counter() - t0
        yield {
            "step": step,
            "loss": loss_accum / cfg.grad_accum,
            "loss_unweighted": loss_unweighted_accum / cfg.grad_accum,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "tokens": n_tokens,
            "tok_per_sec": n_tokens / max(elapsed, 1e-9),
        }


def _weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """per-sample weighted mean CE loss と参考用の unweighted mean を返す。

    ``logits`` shape (B, T, V), ``labels`` shape (B, T), ``weight`` shape (B,)。
    label が ``-100`` の位置は CE 側で無視される (``F.cross_entropy(reduction="none")``)。
    """
    b, t, v = logits.shape
    per_token = F.cross_entropy(
        logits.reshape(-1, v), labels.reshape(-1), reduction="none", ignore_index=-100
    ).reshape(b, t)
    # ignore_index の位置を 0 にしてサンプル内平均を取る (mask を per-sample で割る)。
    mask = (labels != -100).to(per_token.dtype)
    per_sample_sum = (per_token * mask).sum(dim=1)
    per_sample_count = mask.sum(dim=1).clamp_min(1.0)
    per_sample = per_sample_sum / per_sample_count

    w = weight.to(per_sample.dtype)
    denom = w.sum().clamp_min(1e-9)
    weighted = (per_sample * w).sum() / denom
    unweighted = per_sample.mean()
    return weighted, unweighted
