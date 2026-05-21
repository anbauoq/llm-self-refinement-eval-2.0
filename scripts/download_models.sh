#!/bin/bash
# Download all Hugging Face models to local cache
# This script ensures the correct cache directories are used

set -euo pipefail

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate self_play

# Set cache directories to Lustre filesystem
export PIP_CACHE_DIR=/lustre/fsw/llmservice_nemo_reasoning/earakelyan/.cache/pip
export HF_HOME=/lustre/fsw/llmservice_nemo_reasoning/earakelyan/.cache/huggingface
export TRANSFORMERS_CACHE=/lustre/fsw/llmservice_nemo_reasoning/earakelyan/.cache/huggingface

echo "Cache directories configured:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

echo "Running model download script..."
echo "Project directory: $PROJECT_DIR"
echo ""

# Run the Python script
python3 scripts/download_models.py "$@"


