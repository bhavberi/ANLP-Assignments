from huggingface_hub import snapshot_download

model_id = "allenai/OLMo-1B-hf"
snapshot_download(
    repo_id=model_id, local_dir="OLMo-1B-hf", local_dir_use_symlinks=False
)