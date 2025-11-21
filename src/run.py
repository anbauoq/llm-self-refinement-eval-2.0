#!/usr/bin/env python3
# run.py
import torch
import importlib
import logging
from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import solve_questions, generate_hints
from utils import load_data, save_data

# Set up a logger for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset_module(dataset_name: str):
    """Dynamically load the dataset module and handle potential errors."""
    try:
        return importlib.import_module(f"datasets.{dataset_name}")
    except ImportError:
        logging.error(f"Failed to import dataset module for '{dataset_name}'. Make sure 'datasets/{dataset_name}.py' exists.")
        raise

def load_model_and_tokenizer(model_path: str, device_map: str):
    """Load the model and tokenizer with robust device handling."""
    # Don't trust remote code for Microsoft Phi models (security concern)
    trust_remote = "microsoft/Phi" not in model_path
    if not trust_remote:
        logging.warning(f"Disabling trust_remote_code for Microsoft Phi model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote)
    
    common_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "dtype": torch.float16,
        "trust_remote_code": trust_remote,
    }

    if device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs, device_map="auto")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        logging.info(f"Using single device: {device}")
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs).to(device)

    model.eval()
    return model, tokenizer

def run_pipeline(args: ArgumentParser):
    """Execute the main self-refinement pipeline."""
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
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device_map)

    logging.info(f"Loading data from: {args.input_path}")
    raw_data = load_data(args.input_path)
    if args.max_samples:
        raw_data = raw_data[:args.max_samples]
        logging.info(f"Limiting to {args.max_samples} samples.")

    # --- Stage 1: Initial Inference ---
    if not initial_results_path.exists():
        logging.info("Stage 1: Running initial inference...")
        initial_results = solve_questions(raw_data, model, tokenizer, dataset_module, max_tokens=args.max_tokens)
        save_data(initial_results, initial_results_path)
    else:
        logging.info("Stage 1: Loading existing initial inference results.")
        initial_results = load_data(initial_results_path)

    wrong_only = [ex for ex in initial_results if not ex.get("is_correct", False)]
    logging.info(f"Found {len(wrong_only)} incorrect answers for hint generation.")

    if not wrong_only:
        logging.info("No incorrect answers found. Pipeline finished.")
        return

    # --- Stage 2: Hint Generation ---
    if not hints_path.exists():
        logging.info("Stage 2: Generating hints for incorrect answers...")
        hint_results = generate_hints(wrong_only, model, tokenizer, max_tokens=args.max_tokens)
        save_data(hint_results, hints_path)
    else:
        logging.info("Stage 2: Loading existing hints.")
        hint_results = load_data(hints_path)

    # --- Stage 3: Post-hint Re-inference ---
    logging.info("Stage 3: Running post-hint inference...")
    post_results = solve_questions(hint_results, model, tokenizer, dataset_module, inject_hint=True, max_tokens=args.max_tokens)
    save_data(post_results, post_hint_path)

    logging.info(f"All done. Results saved to {output_dir}")


def main():
    parser = ArgumentParser(description="Run self-refinement evaluation pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint or HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (matches datasets/*.py)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store result JSONLs")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of examples")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "single"],
                        help='Use "auto" (Accelerate sharding) or "single" (one device: cuda/mps/cpu).')
    parser.add_argument("--max_tokens", type=int, default=256,
                        help='Max new tokens per generation.')
    
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
