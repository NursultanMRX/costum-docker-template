# config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ── HuggingFace ──────────────────────────────
    HF_TOKEN: str                          # Modal secret dan keladi
    MODEL_REPO: str = "nickoo004/gemma3-4b-karakalpak-merged"
    MODEL_REVISION: str = "main"
    
    # ── vLLM Engine ──────────────────────────────
    MAX_MODEL_LEN: int = 8192             # Gemma3 max context
    GPU_MEMORY_UTILIZATION: float = 0.90  # A10G 24GB dan 90% ishlatamiz
    MAX_NUM_SEQS: int = 16                # parallel batch requests
    TENSOR_PARALLEL_SIZE: int = 1         # 1 GPU
    DTYPE: str = "bfloat16"              # Gemma3 uchun optimal
    
    # ── Generation defaults ───────────────────────
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 50
    DEFAULT_REPETITION_PENALTY: float = 1.1
    
    # ── Server ───────────────────────────────────
    PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
