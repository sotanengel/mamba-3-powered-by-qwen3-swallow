"""Evaluate a checkpoint: val perplexity + lm-evaluation-harness benchmarks.

Usage (inside container):
    python scripts/evaluate.py --ckpt /checkpoints/mamba3-130m/best.pt \
        --model configs/model_130m.yaml \
        --data configs/data.yaml \
        --tasks lambada_openai,jcommonsenseqa
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument(
        "--tasks",
        type=str,
        default="lambada_openai,hellaswag,piqa,arc_easy,winogrande,jcommonsenseqa",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("logs/eval"))
    return p.parse_args()


def compute_val_perplexity(args: argparse.Namespace) -> float:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from mamba3jp.data.binidx import BinIdxReader
    from mamba3jp.data.dataset import MemmapCLMDataset
    from mamba3jp.model.builder import build_model_from_yaml
    from mamba3jp.train.checkpoint import load_ckpt

    with args.data.open(encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    tokenized_dir = Path(data_cfg["tokenized_dir"])
    reader = BinIdxReader(tokenized_dir / "val.bin", tokenized_dir / "val.idx")
    ds = MemmapCLMDataset(reader, seq_len=2048)
    loader = DataLoader(ds, batch_size=1, num_workers=2, pin_memory=True)

    tok = AutoTokenizer.from_pretrained(data_cfg["tokenizer"], trust_remote_code=True)
    raw_vocab = getattr(tok, "vocab_size", None) or len(tok)
    model = build_model_from_yaml(
        args.model, vocab_size=raw_vocab, dtype=torch.bfloat16, device="cuda"
    )
    ck = load_ckpt(args.ckpt, map_location="cuda")
    model.load_state_dict(ck["model"])
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            total_loss += float(out.loss) * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()
    avg = total_loss / max(1, total_tokens)
    return math.exp(avg)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ppl = compute_val_perplexity(args)
    print(f"[eval] val_perplexity={ppl:.4f}")

    # lm-evaluation-harness — pretrained must be a directory; we point at the
    # checkpoint's parent for now (assumes the trainer writes a HF-compatible
    # snapshot alongside .pt files in the future). For initial integration we
    # just record the command we would have run.
    harness_cmd = [
        "lm_eval",
        "--model",
        "mamba_ssm",
        "--model_args",
        f"pretrained={args.ckpt.parent}",
        "--tasks",
        args.tasks,
        "--device",
        "cuda",
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(args.out / f"{args.ckpt.stem}_lmeval.json"),
    ]
    print(f"[eval] running: {' '.join(harness_cmd)}")
    rc = subprocess.run(harness_cmd, check=False).returncode

    summary = {
        "ckpt": str(args.ckpt),
        "val_perplexity": ppl,
        "lm_eval_returncode": rc,
        "lm_eval_tasks": args.tasks.split(","),
    }
    (args.out / f"{args.ckpt.stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
