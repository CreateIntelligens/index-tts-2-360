# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/checkpoints/hf_cache \
    HF_HUB_CACHE=/app/checkpoints/hf_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/checkpoints/hf_cache \
    HF_HUB_CACHE=/app/checkpoints/hf_cache \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip uv \
    && python3 -m venv /opt/venv

WORKDIR /app

COPY pyproject.toml README.md LICENSE LICENSE_ZH.txt DISCLAIMER MANIFEST.in ./
COPY indextts ./indextts
COPY tools ./tools
COPY trainers ./trainers
COPY assets ./assets
COPY examples ./examples
COPY checkpoints/config.yaml ./checkpoints/config.yaml
COPY inference_script.py webui.py webui_parallel.py ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --python /opt/venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --no-build-isolation-package flash-attn \
    --editable ".[cu128,webui,vllm]"

# Ensure the HuggingFace CLI + Xet accelerator are available for the runtime
# entrypoint's model download step.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python "huggingface-hub[cli,hf_xet]"

FROM base AS runner

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/checkpoints /app/outputs

EXPOSE 7860

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "webui_parallel.py", "--host", "0.0.0.0", "--port", "7860"]
