"""
Downloads the model from HuggingFace at Docker build time.
Run via: python3 /app/scripts/download_model.py
The HF_TOKEN is injected as a BuildKit secret at /run/secrets/HF_TOKEN.
"""
import os
from huggingface_hub import hf_hub_download

secret_path = "/run/secrets/HF_TOKEN"
token = open(secret_path).read().strip() if os.path.exists(secret_path) else None

print("Downloading model into image...", flush=True)
hf_hub_download(
    repo_id="nickoo004/gemma3-4b-karakalpak-GGUF",
    filename="gemma3-4b-karakalpak-q4_k_m.gguf",
    local_dir="/app/models",
    token=token,
)
print("Model baked into image successfully!", flush=True)
