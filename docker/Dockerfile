# =============================================================================
# scDNAm-GPM environment image
#
# Reproduces the exact Python environment used in the manuscript:
#   Python 3.10 / PyTorch 2.7.0 (CUDA 12.8) / mamba-ssm 2.3.0 /
#   causal-conv1d 1.6.0 / flash-attn 2.8.3 / transformers 4.57.6 / DeepSpeed 0.18.0
#
# Base image is a CUDA *devel* image on purpose: DeepSpeed JIT-compiles its ops
# on first use and needs nvcc (CUDA_HOME) to be present.
#
# NOTE on the CUDA-extension wheels:
#   - flash-attn is fetched from the official Dao-AILab release.
#   - mamba-ssm 2.3.0 / causal-conv1d 1.6.0 have no official wheels for
#     torch 2.7; prebuilt wheels are hosted on this repository's GitHub
#     Release (tag "env-v1"). See docker/README.md for one-time upload steps.
# =============================================================================

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        git \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV PIP_NO_CACHE_DIR=1

# --- 1) PyTorch (CUDA 12.8 build) -------------------------------------------
RUN pip3 install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 \
        torchvision==0.22.0

# --- 2) Prebuilt CUDA-extension wheels (must match torch/CUDA/python ABI) ----
ARG MAMBA_SSM_WHL="https://github.com/ChaoqiLiang/scDNAm-GPM/releases/download/env-v1/mamba_ssm-2.3.0%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
ARG CAUSAL_CONV1D_WHL="https://github.com/ChaoqiLiang/scDNAm-GPM/releases/download/env-v1/causal_conv1d-1.6.0%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
ARG FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
RUN pip3 install "${MAMBA_SSM_WHL}" "${CAUSAL_CONV1D_WHL}" "${FLASH_ATTN_WHL}"

# --- 3) Remaining pinned environment (276 packages, exact versions) ----------
COPY docker/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip3 install -r /tmp/requirements-lock.txt

# --- 4) Repository code ------------------------------------------------------
COPY . /workspace/scDNAm-GPM
WORKDIR /workspace/scDNAm-GPM
ENV PYTHONPATH=/workspace/scDNAm-GPM

# Smoke-test the environment. Strict: every critical package must be installed
# at the right version (importlib.metadata reads installed metadata without
# loading native extensions). Lenient: actual imports of the CUDA extensions
# are attempted but only reported — deepspeed/mamba/flash-attn require an
# NVIDIA driver at import time and this image is often built on GPU-less CI.
RUN python3 - <<'EOF'
from importlib.metadata import version
core = {"torch": "2.7.0+cu128", "transformers": "4.57.6", "mamba-ssm": None,
        "causal-conv1d": None, "flash-attn": None, "deepspeed": "0.18.0", "scanpy": None}
for pkg, expect in core.items():
    v = version(pkg)
    assert expect is None or v == expect, f"{pkg}: expected {expect}, got {v}"
    print(f"OK {pkg} {v}")
import torch, transformers
print("pure-python imports OK: torch", torch.__version__, "/ transformers", transformers.__version__)
for m in ("mamba_ssm", "causal_conv1d", "flash_attn", "deepspeed"):
    try:
        __import__(m)
        print(f"import OK: {m}")
    except Exception as e:
        print(f"NOTE: {m} import deferred to GPU runtime (no driver on builder): {type(e).__name__}")
print("scDNAm-GPM environment OK")
EOF

CMD ["bash"]
