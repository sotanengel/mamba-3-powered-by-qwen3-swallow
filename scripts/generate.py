"""Generate text from a trained Mamba-3-JP checkpoint.

Usage (inside container):
    python scripts/generate.py \
        --ckpt /checkpoints/mamba3-130m/best.pt \
        --model configs/model_130m.yaml \
        --prompt "日本の四季について簡潔に教えてください。" \
        --max-new-tokens 512 --temperature 0.7 --top-p 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--system", type=str, default="あなたは丁寧で正確な日本語アシスタントです。")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--thinking", action="store_true", help="prefill <think> to encourage CoT")
    return p.parse_args()


def _build_prompt(system: str, user: str, *, thinking: bool) -> str:
    parts = [
        f"<|im_start|>system\n{system}<|im_end|>\n",
        f"<|im_start|>user\n{user}<|im_end|>\n",
        "<|im_start|>assistant\n",
    ]
    if thinking:
        parts.append("<think>\n")
    return "".join(parts)


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    from mamba3jp.model.builder import build_model_from_yaml
    from mamba3jp.train.checkpoint import load_ckpt

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    raw_vocab = getattr(tok, "vocab_size", None) or len(tok)
    model = build_model_from_yaml(
        args.model, vocab_size=raw_vocab, dtype=torch.bfloat16, device="cuda"
    )
    ck = load_ckpt(args.ckpt, map_location="cuda")
    model.load_state_dict(ck["model"])
    model.eval()

    prompt = _build_prompt(args.system, args.prompt, thinking=args.thinking)
    input_ids = tok.encode(prompt, return_tensors="pt").to("cuda")

    eos_id = tok.convert_tokens_to_ids("<|im_end|>") or tok.eos_token_id

    # mamba-ssm provides a generate utility on the LM head model.
    out = model.generate(
        input_ids=input_ids,
        max_length=input_ids.shape[1] + args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=eos_id,
        cg=True,
    )
    text = tok.decode(out.sequences[0], skip_special_tokens=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
