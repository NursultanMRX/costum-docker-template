#!/bin/bash
# HuggingFace dan model yuklab olish

set -e  # xato bo'lsa to'xta

MODEL_DIR="/app/models"
MODEL_REPO="nickoo004/gemma3-4b-karakalpak-GGUF"
MODEL_FILE="gemma3-4b-karakalpak-Q4_K_M.gguf"

mkdir -p $MODEL_DIR

# Model allaqachon bor bo'lsa yuklamaymiz
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✅ Model already exists, skipping download"
    exit 0
fi

echo "⏳ Downloading model from HuggingFace..."

pip install -q huggingface_hub hf_transfer

HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import hf_hub_download
import os

hf_hub_download(
    repo_id='$MODEL_REPO',
    filename='$MODEL_FILE',
    local_dir='$MODEL_DIR',
    token=os.environ['HF_TOKEN'],
)
print('✅ Model downloaded!')
"
