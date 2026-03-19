# modal_app.py
import modal
import asyncio
from typing import AsyncGenerator

# ── Modal App ─────────────────────────────────────────────────
app = modal.App("gemma3-karakalpak-llm")

# ── Docker image ──────────────────────────────────────────────
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "vllm==0.4.3",
        "huggingface_hub",
        "hf_transfer",       # HF dan tez download
        "fastapi",
        "pydantic>=2.0",
        "pydantic-settings",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # tez download
)

# ── Model Cache Volume ────────────────────────────────────────
# Model har safar yuklanmasligi uchun persistent volume
model_cache = modal.Volume.from_name(
    "gemma3-karakalpak-cache",
    create_if_missing=True
)
CACHE_DIR = "/model_cache"

# ── GPU va Secrets ────────────────────────────────────────────
@app.cls(
    image=vllm_image,
    gpu=modal.gpu.A10G(),              # 24GB VRAM — 8.64GB model uchun yetarli
    secrets=[modal.Secret.from_name("gemma3-secrets")],
    volumes={CACHE_DIR: model_cache},  # model cache
    timeout=600,                        # 10 daqiqa timeout
    allow_concurrent_inputs=16,         # bir vaqtda 16 request
    container_idle_timeout=300,         # 5 daqiqa idle → container o'chadi
)
class GemmaModel:
    
    @modal.enter()
    def load_model(self):
        """Container ishga tushganda bir marta chaqiriladi"""
        import os
        from vllm import LLM, SamplingParams
        from config import settings
        
        print("⏳ Loading Gemma3 model...")
        
        # HF dan model yuklab olish (volume cache ishlatadi)
        self.llm = LLM(
            model=settings.MODEL_REPO,
            revision=settings.MODEL_REVISION,
            tokenizer=settings.MODEL_REPO,
            dtype=settings.DTYPE,
            max_model_len=settings.MAX_MODEL_LEN,
            gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
            max_num_seqs=settings.MAX_NUM_SEQS,
            tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,    # Gemma3 custom code uchun
            download_dir=CACHE_DIR,    # volume ga save
            tokenizer_mode="auto",
        )
        
        self.SamplingParams = SamplingParams
        print("✅ Model loaded successfully!")
    
    def _build_sampling_params(self, req) -> "SamplingParams":
        from config import settings
        return self.SamplingParams(
            max_tokens=req.max_tokens or settings.DEFAULT_MAX_TOKENS,
            temperature=req.temperature or settings.DEFAULT_TEMPERATURE,
            top_p=req.top_p or settings.DEFAULT_TOP_P,
            top_k=req.top_k or settings.DEFAULT_TOP_K,
            repetition_penalty=req.repetition_penalty or settings.DEFAULT_REPETITION_PENALTY,
        )
    
    def _apply_chat_template(self, messages: list) -> str:
        """Gemma3 chat template"""
        tokenizer = self.llm.get_tokenizer()
        formatted = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted
    
    # ── Single Generate ───────────────────────────────────────
    @modal.method()
    def generate(self, request_dict: dict) -> dict:
        from schemas import GenerateRequest
        req = GenerateRequest(**request_dict)
        
        sampling_params = self._build_sampling_params(req)
        outputs = self.llm.generate([req.prompt], sampling_params)
        
        output = outputs[0]
        return {
            "text": output.outputs[0].text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        }
    
    # ── Batch Generate ────────────────────────────────────────
    @modal.method()
    def batch_generate(self, request_dict: dict) -> dict:
        from schemas import BatchGenerateRequest
        req = BatchGenerateRequest(**request_dict)
        
        sampling_params = self._build_sampling_params(req)
        
        # vLLM batch — hammasi parallel!
        outputs = self.llm.generate(req.prompts, sampling_params)
        
        results = []
        failed = 0
        for output in outputs:
            try:
                results.append({
                    "text": output.outputs[0].text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                })
            except Exception as e:
                failed += 1
                results.append({
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                })
        
        return {
            "results": results,
            "total_prompts": len(req.prompts),
            "failed": failed,
        }
    
    # ── Chat ──────────────────────────────────────────────────
    @modal.method()
    def chat(self, request_dict: dict) -> dict:
        from schemas import ChatRequest
        req = ChatRequest(**request_dict)
        
        prompt = self._apply_chat_template(req.messages)
        sampling_params = self._build_sampling_params(req)
        outputs = self.llm.generate([prompt], sampling_params)
        
        output = outputs[0]
        return {
            "text": output.outputs[0].text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        }
    
    # ── Streaming ─────────────────────────────────────────────
    @modal.method()
    async def generate_stream(self, request_dict: dict) -> AsyncGenerator[str, None]:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from schemas import GenerateRequest
        req = GenerateRequest(**request_dict)
        
        sampling_params = self._build_sampling_params(req)
        
        # vLLM async streaming
        async for output in self.llm.generate(
            req.prompt,
            sampling_params,
            request_id="stream-1"
        ):
            if output.outputs:
                yield output.outputs[0].text

# ── FastAPI Web Endpoint ───────────────────────────────────────
@app.function(
    image=vllm_image,
    secrets=[modal.Secret.from_name("gemma3-secrets")],
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from schemas import (
        GenerateRequest, GenerateResponse,
        BatchGenerateRequest, BatchGenerateResponse,
        ChatRequest
    )
    import json
    
    web = FastAPI(
        title="Gemma3 Karakalpak LLM API",
        version="1.0.0"
    )
    model = GemmaModel()
    
    @web.get("/health")
    async def health():
        return {"status": "ok", "model": "gemma3-4b-karakalpak"}
    
    @web.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        try:
            result = model.generate.remote(req.model_dump())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web.post("/batch", response_model=BatchGenerateResponse)
    async def batch_generate(req: BatchGenerateRequest):
        try:
            result = model.batch_generate.remote(req.model_dump())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web.post("/chat")
    async def chat(req: ChatRequest):
        if req.stream:
            async def stream_gen():
                async for chunk in model.generate_stream.remote_gen.aio(req.model_dump()):
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_gen(), media_type="text/event-stream")
        
        try:
            result = model.chat.remote(req.model_dump())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web
