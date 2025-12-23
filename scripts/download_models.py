#!/usr/bin/env python3
"""
Download all Hugging Face models used in this project to local cache.
This script downloads models to the configured cache directory (HF_HOME).
"""

import os
import sys
from huggingface_hub import login, snapshot_download
from tqdm import tqdm

# Hugging Face token
HF_TOKEN = os.environ.get('HF_TOKEN')

# All models used in the project
MODELS_NON_REASONING = [
    "google/gemma-2-2b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-Math-1.5B-instruct",
    "Qwen/Qwen2.5-Math-7B-instruct",
]

MODELS_REASONING = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "microsoft/Phi-4-mini-reasoning",
]

ALL_MODELS = MODELS_NON_REASONING + MODELS_REASONING

def download_model(model_name, token):
    """Download a model from Hugging Face to local cache."""
    print(f"\n{'='*80}")
    print(f"Downloading: {model_name}")
    print(f"{'='*80}")
    
    try:
        cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        print(f"Cache directory: {cache_dir}")
        
        # Download the model
        local_path = snapshot_download(
            repo_id=model_name,
            token=token,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        
        print(f"✓ Successfully downloaded {model_name}")
        print(f"  Local path: {local_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {str(e)}", file=sys.stderr)
        return False

def main():
    print("="*80)
    print("Hugging Face Model Downloader")
    print("="*80)
    print(f"Total models to download: {len(ALL_MODELS)}")
    print(f"Cache directory: {os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")
    print()
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    try:
        login(token=HF_TOKEN)
        print("✓ Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"✗ Error logging in: {str(e)}", file=sys.stderr)
        return 1
    
    # Download all models
    successful = 0
    failed = 0
    failed_models = []
    
    for i, model_name in enumerate(ALL_MODELS, 1):
        print(f"\n[{i}/{len(ALL_MODELS)}] Processing: {model_name}")
        
        if download_model(model_name, HF_TOKEN):
            successful += 1
        else:
            failed += 1
            failed_models.append(model_name)
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Total models: {len(ALL_MODELS)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    if failed_models:
        print("\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
        return 1
    
    print("\n✓ All models downloaded successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

