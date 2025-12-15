#!/bin/bash
# scripts/run_sample_initial_inference.sh
# Runs ONLY Stage 1 (initial inference) but limits to N questions per (model, dataset, tokens) run.
# Skips runs where the expected output file already exists and is non-empty.

set -euo pipefail

MODELS_NON_REASONING=(
  "Qwen/Qwen2.5-Math-1.5B"
  "microsoft/Phi-4-mini-instruct"
  "google/gemma-2-2b-it"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

MODELS_REASONING=(
  "microsoft/Phi-4-mini-reasoning"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

DATASETS=( "ar_lsat" "aqua" "gsm8k" "sports" )
TOKENS=(1024)

INPUT_DIR="data"
OUTPUT_DIR="results/sample_results_initial"

RUNNER="src/run_initial_inference.py"
BATCH_SIZE=2

# Limit to N questions
MAX_SAMPLES=4

# Always enable FlashAttention. (bfloat16/fp16 auto happens inside the loader.)
USE_FLASH="--no_flash_attention"

# Optional: enable torch.compile
COMPILE=""

# Expected output filename created by run_initial_inference.py inside --output_dir
# Change this if your script writes a different name.
OUTPUT_FILE="initial_inference.jsonl"

run_one () {
  local model="$1"
  local dataset="$2"
  local t="$3"

  local outdir="$OUTPUT_DIR/$(basename "$model")/$dataset/max${t}/n${MAX_SAMPLES}"
  local outfile="$outdir/$OUTPUT_FILE"

  # Skip if output already exists and is non-empty
  if [[ -s "$outfile" ]]; then
    echo "SKIP (exists): $model on $dataset max_tokens=$t n=$MAX_SAMPLES -> $outfile"
    return 0
  fi

  echo "Running INITIAL ONLY (optimized): $model on $dataset with max_tokens=$t (limit=$MAX_SAMPLES)"
  mkdir -p "$outdir"

  python "$RUNNER" \
    --model_path "$model" \
    --dataset "$dataset" \
    --input_path "$INPUT_DIR/${dataset}.jsonl" \
    --output_dir "$outdir" \
    --max_samples "$MAX_SAMPLES" \
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
