#!/usr/bin/env python3
# run.py - Fast inference with bfloat16, FlashAttention, and batching
import torch
import importlib
import logging
from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import solve_questions, generate_hints
from utils import load_data, save_data
from runner_utils import load_dataset_module, load_model_and_tokenizer

# Set up a logger for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(args: ArgumentParser):
    """Execute the main self-refinement pipeline with optimizations."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    initial_results_path = output_dir / "initial_inference.jsonl"
    hints_path = output_dir / "hints.jsonl"
    post_hint_path = output_dir / "post_hint_inference.jsonl"

    # --- Skip if final results already exist ---
    if post_hint_path.exists():
        logging.warning(f"Skipping pipeline for {args.model_path} on {args.dataset}: Final results already exist at {post_hint_path}")
        return

    # --- Load dependencies ---
    logging.info(f"Loading dataset module: {args.dataset}")
    dataset_module = load_dataset_module(args.dataset)
    
    logging.info(f"Loading model and tokenizer from: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        args.device_map,
        use_flash_attention=args.use_flash_attention,
        compile_model=args.compile_model
    )

    logging.info(f"Loading data from: {args.input_path}")
    raw_data = load_data(args.input_path)
    if args.max_samples:
        raw_data = raw_data[:args.max_samples]
        logging.info(f"Limiting to {args.max_samples} samples.")

    # --- Stage 1: Initial Inference ---
    if not initial_results_path.exists():
        logging.info(f"Stage 1: Running initial inference with batch_size={args.batch_size}...")
        initial_results = solve_questions(
            raw_data, model, tokenizer, dataset_module, 
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            model_name = args.model_path
        )
        save_data(initial_results, initial_results_path)
    else:
        logging.info("Stage 1: Loading existing initial inference results.")
        initial_results = load_data(initial_results_path)

    wrong_only = [
        ex for ex in initial_results
        if ex.get("is_correct") is False and ex.get("predicted_answer") is not None
    ]
    logging.info(f"Found {len(wrong_only)} incorrect answers for hint generation.")

    if not wrong_only:
        logging.info("No incorrect answers found. Pipeline finished.")
        return

    # --- Stage 2: Hint Generation ---
    if not hints_path.exists():
        logging.info(f"Stage 2: Generating hints with batch_size={args.batch_size}...")

        hint_results = generate_hints(
            wrong_only,
            model,
            tokenizer,
            dataset_name=args.dataset,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size
        )


        save_data(hint_results, hints_path)
    else:
        logging.info("Stage 2: Loading existing hints.")
        hint_results = load_data(hints_path)

    # --- Stage 3: Post-hint Re-inference ---
    logging.info(f"Stage 3: Running post-hint inference with batch_size={args.batch_size}...")
    post_results = solve_questions(
        hint_results, model, tokenizer, dataset_module, 
        inject_hint=True, 
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        model_name = args.model_path
    )
    save_data(post_results, post_hint_path)

    logging.info(f"All done. Results saved to {output_dir}")


def main():
    parser = ArgumentParser(description="Run self-refinement evaluation pipeline with optimizations.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint or HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (matches datasets/*.py)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store result JSONLs")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of examples")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "single"],
                        help='Use "auto" (Accelerate sharding) or "single" (one device: cuda/mps/cpu).')
    parser.add_argument("--max_tokens", type=int, default=256,
                        help='Max new tokens per generation.')
    parser.add_argument("--batch_size", type=int, default=8,
                        help='Batch size for inference (default: 8). Increase for faster inference if GPU memory allows.')
    parser.add_argument("--use_flash_attention", action="store_true", default=False,
                        help='Use FlashAttention 2 for 2-4x speedup (default: False)')
    parser.add_argument("--no_flash_attention", dest="use_flash_attention", action="store_false",
                        help='Disable FlashAttention')
    parser.add_argument("--compile_model", action="store_true",
                        help='Use torch.compile for additional ~30%% speedup (first run will be slower)')
    
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
