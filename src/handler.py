import runpod
import os
from llama_cpp import Llama

# Must match the filename used in the Dockerfile hf_hub_download call exactly
MODEL_PATH = "/app/models/gemma3-4b-karakalpak-q4_k_m.gguf"
llm = None


def load_model():
    global llm
    if llm is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "The model must be baked into the Docker image at build time."
        )

    print(f"Loading model from {MODEL_PATH} ...", flush=True)
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=99,       # all layers on GPU (matches local entrypoint.sh)
        n_ctx=32768,           # 32K context (matches local entrypoint.sh)
        n_batch=512,
        n_ubatch=256,          # micro-batch size (matches local entrypoint.sh)
        n_threads=4,
        flash_attn=True,       # required for large context; speeds up attention
        verbose=True,
        # chat_format=None: auto-detect from the GGUF embedded chat template.
        # This is the most correct approach for Gemma 3 and avoids hard-coding
        # a format name that may not exist in the installed llama-cpp-python version.
        chat_format=None,
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


# Load model into VRAM at container startup.
# Because the model is baked into the image this is fast (~seconds to load
# into GPU, no download). The worker only calls runpod.serverless.start()
# after the model is loaded, so RunPod never routes a job to an unready worker.
load_model()

runpod.serverless.start({"handler": handler})
