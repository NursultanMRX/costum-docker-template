# src/handler.py
import runpod
import os
import json
import subprocess
from huggingface_hub import hf_hub_download

# ── Global model ──────────────────────────────────────────────
llm = None
MODEL_PATH = "/app/models/gemma3-4b-karakalpak-Q4_K_M.gguf"

def load_model():
    global llm
    if llm is not None:
        return
    
    # Model yo'q bo'lsa yuklab olish
    if not os.path.exists(MODEL_PATH):
        print("⏳ Downloading model...")
        os.makedirs("/app/models", exist_ok=True)
        hf_hub_download(
            repo_id="nickoo004/gemma3-4b-karakalpak-GGUF",
            filename="gemma3-4b-karakalpak-Q4_K_M.gguf",
            local_dir="/app/models",
            token=os.environ["HF_TOKEN"],
        )
        print("✅ Model downloaded!")
    
    # llama-cpp-python ni runtime da yuklaymiz
    try:
        from llama_cpp import Llama
    except ImportError:
        print("⏳ Installing llama-cpp-python with CUDA...")
        subprocess.run([
            "pip", "install", "--no-cache-dir",
            "llama-cpp-python",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cu121"
        ], check=True)
        from llama_cpp import Llama
    
    print("⏳ Loading model into GPU...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=999,
        n_ctx=8192,
        n_batch=512,
        verbose=False,
        chat_format="gemma",
    )
    print("✅ Model ready!")

# ── Handler ───────────────────────────────────────────────────
def handler(job):
    job_input = job["input"]
    
    # Model yuklanganligini tekshirish
    load_model()
    
    job_type = job_input.get("type", "generate")
    
    try:
        # ── Single generate ───────────────────────────────────
        if job_type == "generate":
            prompt = job_input.get("prompt", "")
            if not prompt:
                return {"error": "prompt is required"}
            
            output = llm(
                prompt,
                max_tokens=job_input.get("max_tokens", 1024),
                temperature=job_input.get("temperature", 0.7),
                top_p=job_input.get("top_p", 0.9),
                top_k=job_input.get("top_k", 50),
                repeat_penalty=job_input.get("repeat_penalty", 1.1),
                stop=job_input.get("stop", ["<end_of_turn>", "<eos>"]),
            )
            
            return {
                "text": output["choices"][0]["text"].strip(),
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"],
                "total_tokens": output["usage"]["total_tokens"],
            }
        
        # ── Batch generate ────────────────────────────────────
        elif job_type == "batch":
            prompts = job_input.get("prompts", [])
            if not prompts:
                return {"error": "prompts is required"}
            if len(prompts) > 32:
                return {"error": "max 32 prompts per batch"}
            
            params = {
                "max_tokens": job_input.get("max_tokens", 1024),
                "temperature": job_input.get("temperature", 0.7),
                "top_p": job_input.get("top_p", 0.9),
                "repeat_penalty": job_input.get("repeat_penalty", 1.1),
                "stop": job_input.get("stop", ["<end_of_turn>", "<eos>"]),
            }
            
            results = []
            failed = 0
            
            for prompt in prompts:
                try:
                    output = llm(prompt, **params)
                    results.append({
                        "text": output["choices"][0]["text"].strip(),
                        "prompt_tokens": output["usage"]["prompt_tokens"],
                        "completion_tokens": output["usage"]["completion_tokens"],
                        "total_tokens": output["usage"]["total_tokens"],
                        "error": None,
                    })
                except Exception as e:
                    failed += 1
                    results.append({
                        "text": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "error": str(e),
                    })
            
            return {
                "results": results,
                "total_prompts": len(prompts),
                "successful": len(prompts) - failed,
                "failed": failed,
            }
        
        # ── Chat ──────────────────────────────────────────────
        elif job_type == "chat":
            messages = job_input.get("messages", [])
            if not messages:
                return {"error": "messages is required"}
            
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=job_input.get("max_tokens", 1024),
                temperature=job_input.get("temperature", 0.7),
                top_p=job_input.get("top_p", 0.9),
                repeat_penalty=job_input.get("repeat_penalty", 1.1),
                stop=job_input.get("stop", ["<end_of_turn>", "<eos>"]),
            )
            
            return {
                "message": {
                    "role": "assistant",
                    "content": output["choices"][0]["message"]["content"].strip(),
                },
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"],
                "total_tokens": output["usage"]["total_tokens"],
            }
        
        else:
            return {"error": f"Unknown type: {job_type}. Use: generate, batch, chat"}
    
    except Exception as e:
        return {"error": str(e)}

# ── Start ─────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
```

---

### 4. `scripts/start.sh` — o'zgartirish shart emas, lekin soddalashtirish mumkin

`CMD ["python", "-u", "src/handler.py"]` — Dockerfile da to'g'ridan belgilandi, `start.sh` shart emas.

---

## RunPod da sozlash
```
Environment Variables:
  HF_TOKEN = hf_your_token (Secret qilib saqlang!)

Container Disk: 15GB
GPU: RTX 3090 / 4090
