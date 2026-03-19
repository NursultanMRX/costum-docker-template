"""
Downloads the model from HuggingFace at Docker build time.
Run via: python3 /app/scripts/download_model.py
The HF_TOKEN is injected as a BuildKit secret at /run/secrets/HF_TOKEN.
"""
import os
import sys

from huggingface_hub import hf_hub_download

secret_path = "/run/secrets/HF_TOKEN"
token = None
if os.path.exists(secret_path):
    token = open(secret_path).read().strip() or None

if token is None:
    print(
        "ERROR: HF_TOKEN secret is missing or empty.\n"
        "Add HF_TOKEN to your repository secrets:\n"
        "  Settings → Secrets and variables → Actions → New repository secret",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

print("Downloading model into image...", flush=True)
try:
    path = hf_hub_download(
        repo_id="nickoo004/gemma3-4b-karakalpak-GGUF",
        filename="gemma3-4b-karakalpak-q4_k_m.gguf",
        local_dir="/app/models",
        token=token,
    )
    print(f"Model saved to {path}", flush=True)
    print("Model baked into image successfully!", flush=True)
except Exception as e:
    print(f"ERROR: Failed to download model: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
