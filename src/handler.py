import runpod
import os
import sys
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_PATH = "/app/models/gemma3-4b-karakalpak-Q4_K_M.gguf"
llm = None


def load_model():
    global llm
    if llm is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("Downloading model from HuggingFace...", flush=True)
        os.makedirs("/app/models", exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("WARNING: HF_TOKEN not set. Download may fail for private repos.", flush=True)
        hf_hub_download(
            repo_id="nickoo004/gemma3-4b-karakalpak-GGUF",
            filename="gemma3-4b-karakalpak-Q4_K_M.gguf",
            local_dir="/app/models",
            token=hf_token,
        )
        print("Model downloaded!", flush=True)

    print("Loading model into GPU...", flush=True)
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=999,
        n_ctx=8192,
        n_batch=512,
        n_threads=4,
        verbose=True,
        chat_format="gemma3",
    )
    print("Model ready!", flush=True)


def handler(job):
    try:
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


# Load model at startup so the first request does not time out waiting for download+load
print("Initializing handler — loading model at startup...", flush=True)
try:
    load_model()
except Exception as e:
    import traceback
    print(f"FATAL: model failed to load: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

runpod.serverless.start({"handler": handler})
