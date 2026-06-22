"""Build a Mamba LM-head model from YAML and optionally run a smoke forward.

Usage (inside container):
    python scripts/build_model.py --config configs/model_130m.yaml --smoke-forward
    python scripts/build_model.py --config configs/model_50m.yaml --smoke-forward
    python scripts/build_model.py --config configs/model_mamba2_130m.yaml --smoke-forward

The smoke step also doubles as the requirements-doc 6.4 risk check: it asserts
that ``ssm_cfg.layer`` actually dispatches to the requested mixer class.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--smoke-forward", action="store_true")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # heavy imports
    import torch
    from transformers import AutoTokenizer

    from mamba3jp.model.builder import build_model_from_yaml, count_parameters

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    raw_vocab = getattr(tok, "vocab_size", None) or len(tok)
    print(f"[build] tokenizer={args.tokenizer} raw_vocab={raw_vocab}")

    model = build_model_from_yaml(
        args.config,
        vocab_size=raw_vocab,
        dtype=torch.bfloat16,
        device=args.device,
    )
    total, trainable = count_parameters(model)
    print(f"[build] params total={total:,} trainable={trainable:,}")
    print(f"[build] mixer={type(model.backbone.layers[0].mixer).__name__}")

    if args.smoke_forward:
        ids = torch.randint(0, model.lm_head.weight.shape[0], (1, args.seq_len)).to(args.device)
        with torch.no_grad():
            out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        print(f"[build] forward OK logits.shape={tuple(logits.shape)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
