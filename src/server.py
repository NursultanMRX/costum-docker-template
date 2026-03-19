# src/server.py
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

from config import settings
from schemas import (
    GenerateRequest, GenerateResponse,
    BatchGenerateRequest, BatchGenerateResponse, BatchItem,
    ChatRequest, ChatResponse, ChatMessage
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Global model instance ─────────────────────────────────────
llm: Llama = None

# ── Semaphore: parallel requestlarni cheklash ─────────────────
semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

# ── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    log.info(f"⏳ Loading model: {settings.model_path}")
    
    llm = Llama(
        model_path=settings.model_path,
        n_gpu_layers=settings.N_GPU_LAYERS,  # hammasi GPU da
        n_ctx=settings.N_CTX,
        n_batch=settings.N_BATCH,
        n_threads=settings.N_THREADS,
        verbose=False,
        chat_format="gemma",   # Gemma3 chat template
    )
    
    log.info("✅ Model loaded! Server ready.")
    yield
    
    log.info("🛑 Shutting down...")
    del llm

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Gemma3 Karakalpak LLM Server",
    version="1.0.0",
    lifespan=lifespan
)

# ── Helpers ───────────────────────────────────────────────────
def build_params(req) -> dict:
    return {
        "max_tokens": req.max_tokens or settings.DEFAULT_MAX_TOKENS,
        "temperature": req.temperature or settings.DEFAULT_TEMPERATURE,
        "top_p": req.top_p or settings.DEFAULT_TOP_P,
        "top_k": req.top_k if hasattr(req, "top_k") and req.top_k else settings.DEFAULT_TOP_K,
        "repeat_penalty": req.repeat_penalty or settings.DEFAULT_REPEAT_PENALTY,
        "stop": req.stop or ["<end_of_turn>", "<eos>"],
    }

# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.MODEL_FILENAME,
        "gpu_layers": settings.N_GPU_LAYERS,
        "ctx": settings.N_CTX,
    }

# ── Single Generate ───────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    async with semaphore:
        try:
            params = build_params(req)
            
            if req.stream:
                # Streaming response
                async def stream_gen():
                    for chunk in llm(req.prompt, stream=True, **params):
                        token = chunk["choices"][0]["text"]
                        yield f"data: {json.dumps({'text': token})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(
                    stream_gen(),
                    media_type="text/event-stream"
                )
            
            # Normal response
            output = llm(req.prompt, **params)
            choice = output["choices"][0]
            usage = output["usage"]
            
            return GenerateResponse(
                text=choice["text"].strip(),
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )
        
        except Exception as e:
            log.error(f"Generate error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# ── Batch Generate ────────────────────────────────────────────
@app.post("/batch", response_model=BatchGenerateResponse)
async def batch_generate(req: BatchGenerateRequest):
    async with semaphore:
        params = build_params(req)
        results = []
        successful = 0
        failed = 0
        
        # llama.cpp parallel batch
        tasks = []
        for prompt in req.prompts:
            tasks.append(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda p=prompt: llm(p, **params)
                )
            )
        
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        for output in outputs:
            if isinstance(output, Exception):
                failed += 1
                results.append(BatchItem(
                    text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    error=str(output)
                ))
            else:
                successful += 1
                usage = output["usage"]
                results.append(BatchItem(
                    text=output["choices"][0]["text"].strip(),
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                ))
        
        return BatchGenerateResponse(
            results=results,
            total_prompts=len(req.prompts),
            successful=successful,
            failed=failed,
        )

# ── Chat ──────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    async with semaphore:
        try:
            params = build_params(req)
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            
            if req.stream:
                async def stream_gen():
                    for chunk in llm.create_chat_completion(
                        messages=messages,
                        stream=True,
                        **params
                    ):
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield f"data: {json.dumps({'text': delta['content']})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(
                    stream_gen(),
                    media_type="text/event-stream"
                )
            
            output = llm.create_chat_completion(
                messages=messages,
                **params
            )
            
            choice = output["choices"][0]["message"]
            usage = output["usage"]
            
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=choice["content"].strip()
                ),
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )
        
        except Exception as e:
            log.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
