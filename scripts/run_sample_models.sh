#!/bin/bash
# set -euo pipefail

# # Initialize conda
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate self_play

# Use optimized inference by default (can override with USE_OPTIMIZED=false)
USE_OPTIMIZED="${USE_OPTIMIZED:-true}"
echo "Inference mode: $([ "$USE_OPTIMIZED" = "true" ] && echo "OPTIMIZED" || echo "STANDARD")"

MODELS_NON_REASONING=(
  "Qwen/Qwen2.5-Math-1.5B"
)

MODELS_REASONING=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

DATASETS=( "ar_lsat" "aqua" "sports" )

TOKENS=(256 512)
INPUT_DIR="data"
OUTPUT_DIR_REASONING="results/sample_results/reasoning"
OUTPUT_DIR_NONREASONING="results/sample_results/nonreasoning"
MAX_SAMPLES=30

# Non-reasoning
for model in "${MODELS_NON_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      echo "Running $model on $dataset with max_tokens=$t"
      
      if [ "$USE_OPTIMIZED" = "true" ]; then
        BATCH_SIZE=$([ "$t" -lt 512 ] && echo 16 || echo 8)
        python src/run_optimized.py \
          --model_path "$model" \
          --dataset "$dataset" \
          --input_path "$INPUT_DIR/${dataset}.jsonl" \
          --output_dir "$OUTPUT_DIR_NONREASONING/$(basename "$model")/$dataset/max${t}" \
          --max_samples "$MAX_SAMPLES" \
          --max_tokens "$t" \
          --batch_size "$BATCH_SIZE"
      else
        python src/run.py \
          --model_path "$model" \
          --dataset "$dataset" \
          --input_path "$INPUT_DIR/${dataset}.jsonl" \
          --output_dir "$OUTPUT_DIR_NONREASONING/$(basename "$model")/$dataset/max${t}" \
          --max_samples "$MAX_SAMPLES" \
          --max_tokens "$t"
      fi
    done
  done
done

# Reasoning
for model in "${MODELS_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      echo "Running $model on $dataset with max_tokens=$t"
      
      if [ "$USE_OPTIMIZED" = "true" ]; then
        BATCH_SIZE=$([ "$t" -lt 512 ] && echo 16 || echo 8)
        python src/run_optimized.py \
          --model_path "$model" \
          --dataset "$dataset" \
          --input_path "$INPUT_DIR/${dataset}.jsonl" \
          --output_dir "$OUTPUT_DIR_REASONING/$(basename "$model")/$dataset/max${t}" \
          --max_samples "$MAX_SAMPLES" \
          --max_tokens "$t" \
          --batch_size "$BATCH_SIZE"
      else
        python src/run.py \
          --model_path "$model" \
          --dataset "$dataset" \
          --input_path "$INPUT_DIR/${dataset}.jsonl" \
          --output_dir "$OUTPUT_DIR_REASONING/$(basename "$model")/$dataset/max${t}" \
          --max_samples "$MAX_SAMPLES" \
          --max_tokens "$t"
      fi
    done
  done
done