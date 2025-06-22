import os
from pathlib import Path
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def ensure_weights(repo_id: str, local_dir: Path):
    """
    Download the repo from Hugging Face into local_dir if no .bin weights are present.
    Uses resume_download=True so it only grabs missing files.
    """
    # Check for any .bin files in the target folder
    if not local_dir.exists() or not any(local_dir.glob("*.bin")):
        print(f"⏬ Downloading weights for {repo_id} into {local_dir} …")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            cache_dir=str(local_dir),
            resume_download=True,
        )
    else:
        print(f"✅ Weights already present in {local_dir}, skipping download.")

def main():
    repo_id = "Qwen/Qwen3-0.6B"
    path = Path.home() / "huggingface" / "Qwen3-0.6B"
    #path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/")
    #mpath = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    #path = os.path.expanduser("~/.cache/huggingface/hub/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/")

    ensure_weights(repo_id, path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True
    )
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
