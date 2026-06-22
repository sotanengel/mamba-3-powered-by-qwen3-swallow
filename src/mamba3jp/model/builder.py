"""Build Mamba-3 / Mamba-2 ``MambaLMHeadModel`` from a YAML config.

The config-shaping helpers (:func:`load_yaml`, :func:`pad_vocab`,
:func:`build_mamba_config`) are deliberately importable without mamba-ssm, so
they can be unit-tested on CPU-only environments (e.g. CI). The model
construction itself (:func:`build_model_from_yaml`) requires mamba-ssm + CUDA
and is only exercised inside the docker container or on the dev machine's GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import torch
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


@dataclass
class MambaConfigSpec:
    """Plain mirror of ``mamba_ssm.models.config_mamba.MambaConfig``.

    Kept in our codebase as a dataclass so we can validate / serialize / test
    without importing the GPU-only package.
    """

    d_model: int
    n_layer: int
    vocab_size: int
    ssm_cfg: dict[str, Any] = field(default_factory=dict)
    rms_norm: bool = True
    fused_add_norm: bool = True
    residual_in_fp32: bool = True
    tie_embeddings: bool = True


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data)}")
    return data


def pad_vocab(n: int, multiple: int = 8) -> int:
    """Round ``n`` up to the next multiple of ``multiple``.

    Required because mamba-ssm's fused kernels prefer 8-aligned vocab sizes and
    Qwen3's 151,643-token vocab is not aligned out of the box.
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    if n == 0:
        return 0
    remainder = n % multiple
    if remainder == 0:
        return n
    return n + (multiple - remainder)


def build_mamba_config(
    cfg: dict[str, Any], vocab_size: int, vocab_multiple: int = 8
) -> MambaConfigSpec:
    """Convert a parsed YAML dict + raw vocab size into a :class:`MambaConfigSpec`."""
    required = ("d_model", "n_layer", "ssm_cfg")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"model config missing required keys: {missing}")
    if "layer" not in cfg["ssm_cfg"]:
        raise ValueError("ssm_cfg.layer is required (e.g. 'Mamba3', 'Mamba2')")

    return MambaConfigSpec(
        d_model=int(cfg["d_model"]),
        n_layer=int(cfg["n_layer"]),
        vocab_size=pad_vocab(int(vocab_size), vocab_multiple),
        ssm_cfg=dict(cfg["ssm_cfg"]),
        rms_norm=bool(cfg.get("rms_norm", True)),
        fused_add_norm=bool(cfg.get("fused_add_norm", True)),
        residual_in_fp32=bool(cfg.get("residual_in_fp32", True)),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
    )


def build_model_from_yaml(
    yaml_path: str | Path,
    vocab_size: int,
    dtype: torch.dtype | None = None,
    device: str = "cuda",
) -> MambaLMHeadModel:
    """Construct a ``MambaLMHeadModel`` from one of ``configs/model_*.yaml``.

    Requires mamba-ssm + CUDA. Verifies that the chosen mixer class is actually
    instantiated (the requirements doc flags this as a risk because the
    ``ssm_cfg.layer='Mamba3'`` dispatch is not loudly documented upstream).
    """
    import torch
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    spec = build_mamba_config(load_yaml(yaml_path), vocab_size)
    mamba_cfg = MambaConfig(
        d_model=spec.d_model,
        n_layer=spec.n_layer,
        vocab_size=spec.vocab_size,
        ssm_cfg=spec.ssm_cfg,
        rms_norm=spec.rms_norm,
        fused_add_norm=spec.fused_add_norm,
        residual_in_fp32=spec.residual_in_fp32,
        tie_embeddings=spec.tie_embeddings,
    )

    model = MambaLMHeadModel(
        mamba_cfg,
        dtype=dtype if dtype is not None else torch.bfloat16,
        device=device,
    )

    expected = spec.ssm_cfg["layer"]
    actual = type(model.backbone.layers[0].mixer).__name__
    if actual != expected:
        raise RuntimeError(
            f"mamba-ssm dispatched {actual!r} but YAML asked for {expected!r}; "
            "ssm_cfg.layer routing in mixer_seq_simple may have changed"
        )

    if spec.tie_embeddings:
        tie_weights(model)

    return model


def tie_weights(model: MambaLMHeadModel) -> None:
    """Share the embedding and lm_head weight tensors.

    The Qwen3 vocab of ~151k means the un-tied embedding+head dwarfs the rest
    of a 130M model. Tying halves that overhead and matches Mamba's published
    recipe.
    """
    model.lm_head.weight = model.backbone.embedding.weight


def count_parameters(model: MambaLMHeadModel) -> tuple[int, int]:
    """Return ``(total_params, trainable_params)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
