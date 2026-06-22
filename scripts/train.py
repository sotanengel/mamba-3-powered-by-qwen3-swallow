"""End-to-end training entrypoint.

Wires together:
- ``mamba3jp.model.builder.build_model_from_yaml`` (#5)
- ``mamba3jp.data.dataset.MemmapCLMDataset`` (#4)
- ``mamba3jp.train.loop.train_steps`` (#7)
- ``mamba3jp.train.checkpoint`` (#8) — last/best/step-N rotation + ``--resume``

Usage (inside container):
    python scripts/train.py \
        --model configs/model_130m.yaml \
        --train configs/train.yaml \
        --data  configs/data.yaml \
        --out   /checkpoints/mamba3-130m

    # Resume:
    python scripts/train.py ... --resume /checkpoints/mamba3-130m/last.pt
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mamba3jp.data.binidx import BinIdxReader
from mamba3jp.data.dataset import MemmapCLMDataset
from mamba3jp.model.builder import build_model_from_yaml, count_parameters
from mamba3jp.train.checkpoint import load_ckpt, rotate, save_ckpt
from mamba3jp.train.loop import TrainConfig, train_steps
from mamba3jp.train.optim import build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="checkpoint output directory")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model: torch.nn.Module, loader: DataLoader, cfg: TrainConfig) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            total += float(out.loss) * batch["input_ids"].numel()
            n += batch["input_ids"].numel()
    model.train()
    return total / max(1, n)


def main() -> int:
    args = parse_args()
    train_cfg_d = _load_yaml(args.train)
    data_cfg_d = _load_yaml(args.data)

    _seed(int(train_cfg_d["seed"]))

    # ---- data ----------
    tokenized_dir = Path(data_cfg_d["tokenized_dir"])
    train_reader = BinIdxReader(tokenized_dir / "train.bin", tokenized_dir / "train.idx")
    val_reader = BinIdxReader(tokenized_dir / "val.bin", tokenized_dir / "val.idx")
    train_ds = MemmapCLMDataset(train_reader, seq_len=int(train_cfg_d["seq_len"]))
    val_ds = MemmapCLMDataset(val_reader, seq_len=int(train_cfg_d["seq_len"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg_d["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg_d["num_workers"]),
        pin_memory=True,
        persistent_workers=int(train_cfg_d["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg_d["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg_d["num_workers"]),
        pin_memory=True,
    )

    # ---- model ----------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(data_cfg_d["tokenizer"], trust_remote_code=True)
    raw_vocab = getattr(tok, "vocab_size", None) or len(tok)
    model = build_model_from_yaml(
        args.model,
        vocab_size=raw_vocab,
        dtype=torch.bfloat16,
        device="cuda",
    )
    total, _ = count_parameters(model)
    print(
        f"[train] model={args.model.stem} params={total:,} mixer={type(model.backbone.layers[0].mixer).__name__}"
    )

    # gradient checkpointing for mamba layers (the upstream MambaLMHeadModel
    # doesn't expose ``gradient_checkpointing_enable`` so we wrap manually).
    if train_cfg_d.get("gradient_checkpointing", True):
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        layers = model.backbone.layers

        def make_forward(orig):
            def fwd(*a, **kw):
                return ckpt_fn(orig, *a, use_reentrant=False, **kw)

            return fwd

        for layer in layers:
            layer.forward = make_forward(layer.forward)  # type: ignore[assignment]

    # ---- optim ----------
    opt = build_optimizer(
        model,
        lr=float(train_cfg_d["lr"]),
        betas=tuple(train_cfg_d["betas"]),
        weight_decay=float(train_cfg_d["weight_decay"]),
        eight_bit=bool(train_cfg_d.get("eight_bit_adam", True)),
    )
    sched = build_scheduler(
        opt,
        total_steps=int(train_cfg_d["max_steps"]),
        warmup_pct=float(train_cfg_d["warmup_pct"]),
        min_lr_ratio=float(train_cfg_d["min_lr_ratio"]),
    )

    start_step = 0
    if args.resume and args.resume.exists():
        ck = load_ckpt(args.resume, map_location="cuda")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scheduler") is not None:
            sched.load_state_dict(ck["scheduler"])
        if "rng" in ck:
            torch.set_rng_state(ck["rng"]["torch"])
            np.random.set_state(ck["rng"]["numpy"])
            random.setstate(ck["rng"]["python"])
        start_step = int(ck["step"])
        print(f"[train] resumed from {args.resume} at step={start_step}")

    # ---- wandb (optional) ----------
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            run_name = train_cfg_d.get("wandb_run_name") or f"{args.model.stem}-{int(time.time())}"
            wandb_run = wandb.init(
                project=train_cfg_d["wandb_project"],
                entity=train_cfg_d.get("wandb_entity"),
                name=run_name,
                config={**train_cfg_d, "model_config": str(args.model)},
                resume="allow",
            )
        except Exception as e:  # pragma: no cover
            print(f"[train] wandb disabled: {e}")
            wandb_run = None

    # ---- loop ----------
    tcfg = TrainConfig(
        max_steps=int(train_cfg_d["max_steps"]),
        grad_accum=int(train_cfg_d["grad_accum"]),
        clip_grad=float(train_cfg_d["clip_grad"]),
        log_interval=int(train_cfg_d["log_interval"]),
        eval_interval=int(train_cfg_d["eval_interval"]),
        save_interval=int(train_cfg_d["save_interval"]),
        device="cuda",
        use_amp=bool(train_cfg_d.get("use_amp", True)),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    for log in train_steps(model, opt, sched, train_loader, tcfg):
        # offset by start_step for resume
        log["step"] = log["step"] + start_step
        if wandb_run is not None and log["step"] % tcfg.log_interval == 0:
            import wandb

            wandb_run.log(
                {
                    "train/loss": log["loss"],
                    "train/lr": log["lr"],
                    "train/grad_norm": log["grad_norm"],
                    "train/tok_per_sec": log["tok_per_sec"],
                },
                step=log["step"],
            )

        if log["step"] % tcfg.eval_interval == 0:
            val_loss = _evaluate(model, val_loader, tcfg)
            print(
                f"[train] step={log['step']} train_loss={log['loss']:.4f} val_loss={val_loss:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss}, step=log["step"])
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(
                    args.out / "best.pt",
                    model=model,
                    optimizer=opt,
                    scheduler=sched,
                    step=log["step"],
                    val_loss=val_loss,
                )

        if log["step"] % tcfg.save_interval == 0:
            save_ckpt(
                args.out / f"step-{log['step']}.pt",
                model=model,
                optimizer=opt,
                scheduler=sched,
                step=log["step"],
            )
            save_ckpt(
                args.out / "last.pt",
                model=model,
                optimizer=opt,
                scheduler=sched,
                step=log["step"],
            )
            rotate(args.out, keep=int(train_cfg_d.get("keep_ckpt", 3)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
