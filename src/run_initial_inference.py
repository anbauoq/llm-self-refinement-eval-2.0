#!/usr/bin/env python3
# run_initial_inference.py - Run ONLY Stage 1 (initial inference)

import logging
from argparse import ArgumentParser
from pathlib import Path

from inference import solve_questions
from utils import load_data, save_data
from runner_common import load_dataset_module, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_initial(args: ArgumentParser):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_results_path = output_dir / "initial_inference.jsonl"

    if initial_results_path.exists() and not args.overwrite:
        logger.warning(
            f"Initial results already exist at {initial_results_path}. "
            f"Use --overwrite to rerun."
        )
        return

    logger.info(f"Loading dataset module: {args.dataset}")
    dataset_module = load_dataset_module(args.dataset)

    logger.info(f"Loading model and tokenizer from: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device_map,
        use_flash_attention=args.use_flash_attention,
        compile_model=args.compile_model,
    )

    logger.info(f"Loading data from: {args.input_path}")
    raw_data = load_data(args.input_path)
    if args.max_samples:
        raw_data = raw_data[: args.max_samples]
        logger.info(f"Limiting to {args.max_samples} samples.")

    logger.info(f"Stage 1: Running initial inference with batch_size={args.batch_size}...")
    initial_results = solve_questions(
        raw_data,
        model,
        tokenizer,
        dataset_module,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        model_name=args.model_path,
    )

    save_data(initial_results, initial_results_path)
    logger.info(f"Done. Saved initial inference to {initial_results_path}")


def main():
    parser = ArgumentParser(description="Run ONLY initial inference (Stage 1) with optimizations.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint or HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (matches datasets/*.py)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store result JSONL")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of examples")

    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        choices=["auto", "single"],
        help='Use "auto" (Accelerate sharding) or "single" (one device: cuda/mps/cpu).',
    )
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens per generation.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")

    parser.add_argument("--use_flash_attention", action="store_true", default=False, help="Use FlashAttention 2")
    parser.add_argument("--no_flash_attention", dest="use_flash_attention", action="store_false", help="Disable FlashAttention")
    parser.add_argument("--compile_model", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing initial_inference.jsonl")

    args = parser.parse_args()
    run_initial(args)


if __name__ == "__main__":
    main()
