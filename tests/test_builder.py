"""TDD: tests for src/mamba3jp/model/builder.py.

CPU-only tests cover config loading and vocab padding logic. The actual model
construction and forward pass are gated behind ``@pytest.mark.gpu`` because the
upstream ``mamba-ssm`` kernels expect CUDA.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mamba3jp.model.builder import build_mamba_config, load_yaml, pad_vocab

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "configs"


# ---- CPU tests --------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "multiple", "expected"),
    [
        (151_643, 8, 151_648),
        (151_648, 8, 151_648),
        (1, 8, 8),
        (0, 8, 0),
        (100, 64, 128),
    ],
)
def test_pad_vocab(n: int, multiple: int, expected: int) -> None:
    assert pad_vocab(n, multiple) == expected


def test_load_yaml_parses_mamba3_130m() -> None:
    cfg = load_yaml(CFG_DIR / "model_130m.yaml")
    assert cfg["d_model"] == 768
    assert cfg["n_layer"] == 24
    assert cfg["ssm_cfg"]["layer"] == "Mamba3"
    assert cfg["tie_embeddings"] is True


def test_load_yaml_parses_mamba3_50m() -> None:
    cfg = load_yaml(CFG_DIR / "model_50m.yaml")
    assert cfg["d_model"] == 512
    assert cfg["n_layer"] == 16
    assert cfg["ssm_cfg"]["layer"] == "Mamba3"


def test_load_yaml_parses_mamba2_baseline() -> None:
    cfg = load_yaml(CFG_DIR / "model_mamba2_130m.yaml")
    assert cfg["ssm_cfg"]["layer"] == "Mamba2"


def test_build_mamba_config_propagates_ssm_layer() -> None:
    cfg = load_yaml(CFG_DIR / "model_130m.yaml")
    mc = build_mamba_config(cfg, vocab_size=151_648)
    assert mc.d_model == 768
    assert mc.n_layer == 24
    assert mc.vocab_size == 151_648
    assert mc.ssm_cfg["layer"] == "Mamba3"
    assert mc.tie_embeddings is True


def test_build_mamba_config_pads_vocab_when_misaligned() -> None:
    cfg = load_yaml(CFG_DIR / "model_130m.yaml")
    mc = build_mamba_config(cfg, vocab_size=151_643, vocab_multiple=8)
    assert mc.vocab_size == 151_648


# ---- GPU tests (gated) ------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("yaml_name", ["model_130m.yaml", "model_50m.yaml"])
def test_build_model_dispatches_mamba3(yaml_name: str) -> None:
    import torch

    from mamba3jp.model.builder import build_model_from_yaml

    model = build_model_from_yaml(
        CFG_DIR / yaml_name,
        vocab_size=151_648,
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert model.backbone.layers[0].mixer.__class__.__name__ == "Mamba3"


@pytest.mark.gpu
def test_build_model_dispatches_mamba2() -> None:
    import torch

    from mamba3jp.model.builder import build_model_from_yaml

    model = build_model_from_yaml(
        CFG_DIR / "model_mamba2_130m.yaml",
        vocab_size=151_648,
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert model.backbone.layers[0].mixer.__class__.__name__ == "Mamba2"


@pytest.mark.gpu
def test_tie_weights_shares_storage() -> None:
    import torch

    from mamba3jp.model.builder import build_model_from_yaml

    model = build_model_from_yaml(
        CFG_DIR / "model_50m.yaml",
        vocab_size=151_648,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # The embedding and lm_head weight tensors share storage.
    assert model.lm_head.weight.data_ptr() == model.backbone.embedding.weight.data_ptr()


@pytest.mark.gpu
def test_smoke_forward_logits_shape() -> None:
    import torch

    from mamba3jp.model.builder import build_model_from_yaml

    model = build_model_from_yaml(
        CFG_DIR / "model_50m.yaml",
        vocab_size=151_648,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ids = torch.randint(0, 151_648, (1, 64), device="cuda")
    out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    assert logits.shape == (1, 64, 151_648)
