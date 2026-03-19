FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install llama-cpp-python with CUDA support first (plain pip gives CPU-only build)
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir "llama-cpp-python>=0.3.0"

# Install remaining dependencies (pip will skip llama-cpp-python since it is already satisfied)
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN chmod +x scripts/*.sh

RUN mkdir -p /app/models

CMD ["python", "-u", "src/handler.py"]
