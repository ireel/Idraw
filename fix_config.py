import os
# Set mirror endpoint before importing huggingface_hub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

cache_dir = "./models/sd15_config"
os.makedirs(cache_dir, exist_ok=True)

print(f"Downloading config files to {cache_dir}...")
try:
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=cache_dir,
        allow_patterns=[
            "model_index.json", 
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "unet/config.json",
            "vae/config.json",
            "feature_extractor/preprocessor_config.json"
        ],
        local_dir_use_symlinks=False
    )
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
