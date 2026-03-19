import runpod
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_PATH = "/app/models/gemma3-4b-karakalpak-Q4_K_M.gguf"
llm = None

def load_model():
    global llm
    if llm is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("⏳ Downloading model...")
        os.makedirs("/app/models", exist_ok=True)
        hf_hub_download(
            repo_id="nickoo004/gemma3-4b-karakalpak-GGUF",
            filename="gemma3-4b-karakalpak-Q4_K_M.gguf",
            local_dir="/app/models",
            token=os.environ["HF_TOKEN"],
        )
        print("✅ Downloaded!")

    print("⏳ Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=999,
        n_ctx=8192,
        n_batch=512,
        verbose=True,       # ← True qiling, log ko'rish uchun
        chat_format="gemma",
    )
    print("✅ Model ready!")

def handler(job):
    try:
        load_model()
        job_input = job["input"]
        job_type  = job_input.get("type", "generate")

        if job_type == "generate":
            output = llm(
                job_input.get("prompt", ""),
                max_tokens=job_input.get("max_tokens", 512),
                temperature=job_input.get("temperature", 0.7),
                stop=["<end_of_turn>", "<eos>"],
            )
            return {
                "text": output["choices"][0]["text"].strip(),
                "tokens": output["usage"]["total_tokens"],
            }

        elif job_type == "chat":
            output = llm.create_chat_completion(
                messages=job_input.get("messages", []),
                max_tokens=job_input.get("max_tokens", 512),
                temperature=job_input.get("temperature", 0.7),
            )
            return {
                "text": output["choices"][0]["message"]["content"].strip(),
                "tokens": output["usage"]["total_tokens"],
            }

        elif job_type == "batch":
            results = []
            for prompt in job_input.get("prompts", []):
                out = llm(prompt, max_tokens=job_input.get("max_tokens", 512))
                results.append(out["choices"][0]["text"].strip())
            return {"results": results}

        else:
            return {"error": f"Unknown type: {job_type}"}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
