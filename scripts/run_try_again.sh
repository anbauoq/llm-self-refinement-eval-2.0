#!/bin/bash
set -euo pipefail
# Runs the "try again" baseline for all model-dataset pairs at max_tokens=256.
# Output goes to results/try_again_results/ (separate from results/).
# Reuses existing initial_inference.jsonl from results/ to avoid re-running stage 1.

if [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate self_play
fi

PYTHON=$(command -v python || command -v python3)
if [ -z "$PYTHON" ]; then
  echo "Error: no python or python3 found in PATH" >&2
  exit 1
fi

MODELS_NON_REASONING=(
  # "Qwen/Qwen2.5-Math-1.5B-instruct"
  # "Qwen/Qwen2.5-Math-7B-instruct"
  # "microsoft/Phi-4-mini-instruct"
  "google/gemma-2-2b-it"
  # "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

MODELS_REASONING=(
  # "microsoft/Phi-4-mini-reasoning"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)

DATASETS=( "asdiv" "aqua" "gsm8k" "sports" "ar_lsat") # add ar_lsat heto

INPUT_DIR="data"
EXISTING_RESULTS_DIR="results"        # where initial_inference.jsonl already lives
OUTPUT_DIR="results/try_again_results"  # where try_again_inference.jsonl will be written
RUNNER="src/run_try_again.py"
BATCH_SIZE=4
MAX_TOKENS=256

FAILED_RUNS=()

for model in "${MODELS_NON_REASONING[@]}" "${MODELS_REASONING[@]}"; do
  short_name="$(basename "$model")"

  for dataset in "${DATASETS[@]}"; do
    echo "Running try-again: ${short_name} | ${dataset} | max_tokens=${MAX_TOKENS}"

    if "$PYTHON" "$RUNNER" \
      --model_path "$model" \
      --dataset "$dataset" \
      --input_path "${INPUT_DIR}/${dataset}.jsonl" \
      --output_dir "${OUTPUT_DIR}/${short_name}/${dataset}/max${MAX_TOKENS}" \
      --initial_results_dir "${EXISTING_RESULTS_DIR}/${short_name}/${dataset}/max${MAX_TOKENS}" \
      --max_tokens "$MAX_TOKENS" \
      --batch_size "$BATCH_SIZE"; then

      echo "Finished: ${short_name} | ${dataset}"
    else
      status=$?
      echo "FAILED: ${short_name} | ${dataset} | exit code ${status}" >&2
      FAILED_RUNS+=("${short_name} | ${dataset} | exit code ${status}")
      continue
    fi
  done
done

echo
echo "All runs attempted."

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo "Some runs failed:" >&2
  printf '  - %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
else
  echo "No failures."
fi


