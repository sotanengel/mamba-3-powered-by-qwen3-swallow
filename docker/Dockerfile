# syntax=docker/dockerfile:1.7
# Mamba-3-JP base image: CUDA 12.4 + PyTorch 2.3 + mamba-ssm (source built)
#
# Build:
#   docker compose build train
#
# Smoke test:
#   docker compose run --rm train python -c "from mamba_ssm import Mamba3; print('OK')"

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

# PyTorch (cu124) — pinned to match mamba-ssm expectations
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# packaging / ninja are required by mamba-ssm build_ext
RUN pip install packaging ninja wheel setuptools

# causal-conv1d (mamba-ssm dependency)
RUN pip install causal-conv1d>=1.4.0 --no-build-isolation

# mamba-ssm: build from source (this is the expensive layer; pinned by commit for reproducibility)
RUN pip install --no-build-isolation \
        git+https://github.com/state-spaces/mamba.git@main

# Project dependencies (non-GPU; the GPU ones are torch/bitsandbytes already in)
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

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
