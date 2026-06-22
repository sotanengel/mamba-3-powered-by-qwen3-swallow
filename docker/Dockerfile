# syntax=docker/dockerfile:1.7
# Mamba-3-JP base image: CUDA 12.4 + PyTorch 2.4.1 + mamba-ssm (source built)
#                        + triton 3.5 (required by mamba3_siso_fwd at import)
#
# Build:
#   docker compose -f docker/docker-compose.yml build train
#
# Smoke test:
#   docker compose -f docker/docker-compose.yml run --rm train \
#       python -c "from mamba_ssm import Mamba3; print('OK')"

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    MAMBA_FORCE_BUILD=TRUE

# System packages + Python 3.11 from deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
        git curl wget build-essential ninja-build \
        tzdata \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (cu124) — 2.3.1 wheels are no longer published on the cu124 index;
# 2.4.1+cu124 is the closest still-available release.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# packaging / ninja are required by mamba-ssm build_ext.
# einops / transformers are mamba-ssm runtime imports — install them here so the
# next `pip install --no-deps` for mamba-ssm doesn't leave them missing.
RUN pip install packaging ninja wheel setuptools einops transformers

# causal-conv1d (mamba-ssm dependency)
RUN pip install causal-conv1d>=1.4.0 --no-build-isolation

# mamba-ssm: build from source.
# `--no-deps` is critical: upstream `@main` declares tilelang / quack-kernels /
# triton>=3.5 as hard deps, which force pip to upgrade torch to 2.12+cu130
# and break the just-built selective_scan_cuda extension (ABI mismatch).
# We skip those optional accelerators — Mamba3 still runs via the source-built
# CUDA kernels against torch 2.4.1+cu124.
RUN pip install --no-build-isolation --no-deps \
        git+https://github.com/state-spaces/mamba.git@main

# Project dependencies (non-GPU; the GPU ones are torch/bitsandbytes already in)
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Upgrade triton LAST. mamba3_siso_fwd.py calls `triton.set_allocator` at import
# time, which only exists in triton >= 3.4. mamba-ssm's own metadata asks for
# triton>=3.5. Must run after requirements.txt because lm-eval pulls torch's pin
# (triton==3.0.0) and would otherwise downgrade us.
RUN pip install --upgrade "triton==3.5.0"

# Strip build-only artifacts that are now baked into installed wheels.
# (build-essential / ninja-build are kept; users may want to rebuild kernels
# from a devcontainer shell. software-properties-common is only needed during
# the deadsnakes PPA setup above.)
RUN apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /root/.cache/pip /tmp/pip-* /tmp/* /var/tmp/* /var/cache/apt/archives/*.deb

# Non-root user
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} mamba \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash mamba \
    && mkdir -p /workspace /checkpoints /logs \
    && chown -R mamba:mamba /workspace /checkpoints /logs

USER mamba
WORKDIR /workspace

CMD ["bash"]
