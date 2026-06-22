"""Optimizer / scheduler factories.

We default to bitsandbytes' 8-bit Adam to halve the optimizer state memory
footprint (~260 MB at 130M vs ~1 GB for fp32 Adam). The function falls back to
``torch.optim.AdamW`` when bitsandbytes is unavailable so the loop stays
testable without the CUDA dependency.
"""

from __future__ import annotations

import math

import torch


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0,
    eight_bit: bool = True,
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if eight_bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.Adam8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)
        except ImportError:
            pass  # fall through to AdamW
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_pct: float = 0.02,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup → cosine decay to ``peak_lr * min_lr_ratio``."""
    warmup_steps = max(1, int(total_steps * warmup_pct))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
