FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install llama-cpp-python with CUDA support first (plain pip gives CPU-only build)
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir "llama-cpp-python>=0.3.0"

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN chmod +x scripts/*.sh

# -----------------------------------------------------------------------
# Bake the model into the image at build time.
# This eliminates runtime downloads entirely: workers start in seconds,
# register with RunPod immediately, and never crash due to a missing
# HF_TOKEN or a download timeout on cold start.
#
# Pass the token via BuildKit secret (never stored in a layer):
#   docker build --secret id=HF_TOKEN,env=HF_TOKEN .
# GitHub Actions: add  secrets: | HF_TOKEN=${{ secrets.HF_TOKEN }}
# -----------------------------------------------------------------------
RUN mkdir -p /app/models

RUN --mount=type=secret,id=HF_TOKEN \
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 /app/scripts/download_model.py

CMD ["python", "-u", "src/handler.py"]
