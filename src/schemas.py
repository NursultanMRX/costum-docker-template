# src/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

# ── Single Generate ───────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    repeat_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False
    stop: Optional[List[str]] = None   # stop tokens

class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str = "gemma3-4b-karakalpak-Q4_K_M"

# ── Batch Generate ────────────────────────────────────────────
class BatchItem(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: Optional[str] = None    # xato bo'lsa shu yerda

class BatchGenerateRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=32)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repeat_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    stop: Optional[List[str]] = None

class BatchGenerateResponse(BaseModel):
    results: List[BatchItem]
    total_prompts: int
    successful: int
    failed: int
    model: str = "gemma3-4b-karakalpak-Q4_K_M"

# ── Chat (OpenAI compatible) ──────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repeat_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False
    stop: Optional[List[str]] = None

class ChatResponse(BaseModel):
    message: ChatMessage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str = "gemma3-4b-karakalpak-Q4_K_M"
