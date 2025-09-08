#!/bin/bash

MODELS_NON_REASONING=(
  "google/gemma-2-2b-it"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "microsoft/Phi-4-mini-instruct"
  "Qwen/Qwen2.5-Math-1.5B"
  "Qwen/Qwen2.5-Math-7B"
)

MODELS_REASONING=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "microsoft/Phi-4-mini-reasoning"
)

DATASETS=( "ar_lsat" "asdiv" "aqua" "gsm8k" "sports" )

TOKENS=(1024 2048)
INPUT_DIR="data"
OUTPUT_DIR="results/all_results"

# Non-reasoning
for model in "${MODELS_NON_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      echo "Running $model on $dataset with max_tokens=$t"
      python src/run.py \
        --model_path "$model" \
        --dataset "$dataset" \
        --input_path "$INPUT_DIR/${dataset}.jsonl" \
        --output_dir "$OUTPUT_DIR/$(basename "$model")/$dataset/max${t}" \
        --max_tokens "$t"
    done
  done
done

# Reasoning
for model in "${MODELS_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for t in "${TOKENS[@]}"; do
      echo "Running $model on $dataset with max_tokens=$t"
      python src/run.py \
        --model_path "$model" \
        --dataset "$dataset" \
        --input_path "$INPUT_DIR/${dataset}.jsonl" \
        --output_dir "$OUTPUT_DIR/$(basename "$model")/$dataset/max${t}" \
        --max_tokens "$t"
    done
  done
done
