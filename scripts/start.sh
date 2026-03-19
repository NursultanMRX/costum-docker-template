#!/bin/bash
set -e

echo "🚀 Starting Gemma3 Karakalpak Server..."

# Model yuklab olish
bash /app/scripts/download_model.sh

# Server ishga tushirish
exec uvicorn src.server:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --loop asyncio \
    --log-level info
