# schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

# ── Single Request ────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=100)
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# ── Batch Request ─────────────────────────────────────────────
class BatchGenerateRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=32)  # max 32 ta bir vaqtda
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)

class BatchGenerateResponse(BaseModel):
    results: List[GenerateResponse]
    total_prompts: int
    failed: int = 0

# ── Chat Request (OpenAI compatible) ─────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False
