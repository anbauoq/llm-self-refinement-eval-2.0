#!/bin/bash

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate self_play

MODELS_NON_REASONING=(
  "Qwen/Qwen2.5-Math-1.5B-instruct"
  "Qwen/Qwen2.5-Math-7B-instruct"
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
OUTPUT_DIR="results/all_results"

RUNNER="run.py"   # change to "src/run.py" if that's where it is
BATCH_SIZE=16     # delete this + --batch_size if run.py doesn't support it

# Non-reasoning
for model in "${MODELS_NON_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      echo "Running $model on $dataset with max_tokens=$t"
      python "$RUNNER" \
        --model_path "$model" \
        --dataset "$dataset" \
        --input_path "$INPUT_DIR/${dataset}.jsonl" \
        --output_dir "$OUTPUT_DIR/$(basename "$model")/$dataset/max${t}" \
        --max_tokens "$t" \
        --batch_size "$BATCH_SIZE"
    done
  done
done

# Reasoning
for model in "${MODELS_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      echo "Running $model on $dataset with max_tokens=$t"
      python "$RUNNER" \
        --model_path "$model" \
        --dataset "$dataset" \
        --input_path "$INPUT_DIR/${dataset}.jsonl" \
        --output_dir "$OUTPUT_DIR/$(basename "$model")/$dataset/max${t}" \
        --max_tokens "$t" \
        --batch_size "$BATCH_SIZE"
    done
  done
done