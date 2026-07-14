# =============================================================================
# electronic-lora — ACE-Step 1.5 LoKr TRAINING image (Lambda / any CUDA host)
# =============================================================================
# A pinned, reproducible training environment. Mirrors ACE-Step's own inference
# Dockerfile (nvidia/cuda 12.8, Python 3.11, uv --frozen) but is built for
# headless LoKr training with the two runtime patches BAKED IN at build time —
# no live requirements-filtering or meta-tensor patching per session.
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t electronic-lora-train:latest docker/
#
# Run: see docker/README.md (Lambda workflow) and docker/train.env.example.
# =============================================================================

ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.11
# Pin ACE-Step-1.5 to the exact commit the baked patches were verified against.
# Bump deliberately (and re-verify apply_patches.py) — do NOT float to main.
ARG ACESTEP_REF=6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ---- system packages (mirror ACE-Step's Dockerfile) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common build-essential git curl wget \
        libsndfile1 libsndfile1-dev ffmpeg libffi-dev libssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# ---- uv (fast, reproducible installs) ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- clone ACE-Step-1.5 at the pinned commit ----
ARG ACESTEP_REF
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /app \
    && cd /app && git checkout ${ACESTEP_REF} \
    && echo "${ACESTEP_REF}" > /app/.acestep_ref
WORKDIR /app

# ---- project deps from the lockfile (reproducible) ----
RUN uv sync --frozen --no-dev --python python3.11

# ---- training-only deps not in the inference lockfile ----
# (loguru / lycoris-lora / vector_quantize_pytorch were installed ad hoc in the
#  Colab notebook; wandb + tensorboard drive experiment tracking.)
RUN uv pip install \
        "wandb>=0.17" "tensorboard>=2.16" \
        loguru lycoris-lora vector_quantize_pytorch hf_transfer

# ---- bake the two runtime patches (meta-tensor guard + model_loader) ----
# These were live-patched every Colab session; here they are permanent and
# deterministic because ACESTEP_REF and the dep versions are pinned.
COPY apply_patches.py /opt/apply_patches.py
RUN uv run python /opt/apply_patches.py --verify

# ---- training launcher + entrypoint ----
COPY wandb_launch.py /opt/wandb_launch.py
COPY entrypoint.sh   /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Everything reads/writes under /workspace (bind-mounted persistent storage).
# Running with cwd=/workspace also satisfies the trainer's safe_path guard.
ENV WORKDIR=/workspace
WORKDIR /workspace

ENTRYPOINT ["/opt/entrypoint.sh"]
