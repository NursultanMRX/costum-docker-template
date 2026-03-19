# src/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ── HuggingFace ──────────────────────────────
    HF_TOKEN: str
    MODEL_REPO: str = "nickoo004/gemma3-4b-karakalpak-GGUF"
    MODEL_FILENAME: str = "gemma3-4b-karakalpak-Q4_K_M.gguf"
    MODEL_DIR: str = "/app/models"
    
    # ── llama.cpp server ─────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # GPU layers — VRAM ga qarab
    # RTX 3090 (24GB) + Q4_K_M (2.49GB) → hammasi GPU da!
    N_GPU_LAYERS: int = 999   # 999 = barcha layerlar GPU da
    N_CTX: int = 8192         # Gemma3 max context
    N_BATCH: int = 512        # parallel batch tokens
    N_THREADS: int = 4        # CPU threads (GPU bo'lsa kam kerak)
    
    # ── Generation defaults ───────────────────────
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 50
    DEFAULT_REPEAT_PENALTY: float = 1.1
    
    # ── Server ───────────────────────────────────
    MAX_CONCURRENT_REQUESTS: int = 8  # 24GB VRAM uchun
    REQUEST_TIMEOUT: int = 300        # 5 daqiqa
    
    @property
    def model_path(self) -> str:
        return f"{self.MODEL_DIR}/{self.MODEL_FILENAME}"
    
    class Config:
        env_file = ".env"

settings = Settings()
