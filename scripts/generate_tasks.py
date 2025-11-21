#!/usr/bin/env python3
"""
Generate task list and filter out completed tasks.
"""
import os
from pathlib import Path

# Model lists
MODELS_NON_REASONING = [
    "google/gemma-2-2b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-Math-7B",
]

MODELS_REASONING = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "microsoft/Phi-4-mini-reasoning",
]

DATASETS = ["ar_lsat", "asdiv", "aqua", "gsm8k", "sports"]
TOKENS = [1024, 2048]
INPUT_DIR = "data"
OUTPUT_DIR = "results/all_results"

def get_output_marker(model, dataset, max_tokens):
    """Get the marker file path that indicates task completion."""
    model_name = os.path.basename(model)
    output_path = f"{OUTPUT_DIR}/{model_name}/{dataset}/max{max_tokens}"
    # Check for a completion marker file
    return f"{output_path}/.completed"

def is_task_completed(model, dataset, max_tokens):
    """Check if a task has been completed."""
    marker_file = get_output_marker(model, dataset, max_tokens)
    return os.path.exists(marker_file)

def generate_all_tasks():
    """Generate list of all tasks."""
    tasks = []
    
    # Non-reasoning models
    for model in MODELS_NON_REASONING:
        for dataset in DATASETS:
            for tokens in TOKENS:
                tasks.append((model, dataset, tokens))
    
    # Reasoning models
    for model in MODELS_REASONING:
        for dataset in DATASETS:
            for tokens in TOKENS:
                tasks.append((model, dataset, tokens))
    
    return tasks

def main():
    # Generate all tasks
    all_tasks = generate_all_tasks()
    
    # Write all tasks to file
    with open("tasks.txt", "w") as f:
        for model, dataset, tokens in all_tasks:
            model_name = os.path.basename(model)
            output_path = f"{OUTPUT_DIR}/{model_name}/{dataset}/max{tokens}"
            f.write(f"{model}|{dataset}|{tokens}|{output_path}\n")
    
    # Filter out completed tasks
    remaining_tasks = []
    for model, dataset, tokens in all_tasks:
        if not is_task_completed(model, dataset, tokens):
            remaining_tasks.append((model, dataset, tokens))
    
    # Write remaining tasks to file
    with open("tasks_remaining.txt", "w") as f:
        for model, dataset, tokens in remaining_tasks:
            model_name = os.path.basename(model)
            output_path = f"{OUTPUT_DIR}/{model_name}/{dataset}/max{tokens}"
            f.write(f"{model}|{dataset}|{tokens}|{output_path}\n")
    
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Completed tasks: {len(all_tasks) - len(remaining_tasks)}")
    print(f"Remaining tasks: {len(remaining_tasks)}")

if __name__ == "__main__":
    main()

