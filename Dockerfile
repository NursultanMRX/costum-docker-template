FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── System deps ───────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python alias ──────────────────────────────────────────────
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# ── llama-cpp-python (CUDA bilan) ────────────────────────────
ENV CMAKE_ARGS="-DLLAMA_CUDA=on"
ENV FORCE_CMAKE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App kodi ──────────────────────────────────────────────────
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN chmod +x scripts/*.sh

# ── Models papkasi ────────────────────────────────────────────
RUN mkdir -p /app/models

EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["bash", "scripts/start.sh"]
