#!/bin/bash
# scripts/run_initial_inference.sh
# Runs ONLY Stage 1 (initial inference) with bfloat16/fp16 auto, FlashAttention, batching (+ optional torch.compile).

set -euo pipefail

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate self_play

MODELS_NON_REASONING=(
  "Qwen/Qwen2.5-Math-1.5B"
  "Qwen/Qwen2.5-Math-7B"
  "microsoft/Phi-4-mini-instruct"
  "google/gemma-2-2b-it"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

MODELS_REASONING=(
  "microsoft/Phi-4-mini-reasoning"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)

DATASETS=( "ar_lsat" "asdiv" "aqua" "gsm8k" "sports" )
TOKENS=(1024 2048)

INPUT_DIR="data"
OUTPUT_DIR="results/all_results_initial"

RUNNER="src/run_initial_inference.py"
BATCH_SIZE=16

# Always enable FlashAttention. (bfloat16/fp16 auto happens inside the loader.)
USE_FLASH="--use_flash_attention"

# Optional: enable torch.compile
COMPILE="--compile_model"


run_one () {
  local model="$1"
  local dataset="$2"
  local t="$3"

  echo "Running INITIAL ONLY (optimized): $model on $dataset with max_tokens=$t"
  python "$RUNNER" \
    --model_path "$model" \
    --dataset "$dataset" \
    --input_path "$INPUT_DIR/${dataset}.jsonl" \
    --output_dir "$OUTPUT_DIR/$(basename "$model")/$dataset/max${t}" \
    --max_tokens "$t" \
    --batch_size "$BATCH_SIZE" \
    $USE_FLASH \
    $COMPILE
}

# Non-reasoning
for model in "${MODELS_NON_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      run_one "$model" "$dataset" "$t"
    done
  done
done

# Reasoning
for model in "${MODELS_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      run_one "$model" "$dataset" "$t"
    done
  done
done
