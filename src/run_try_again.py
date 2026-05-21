#!/usr/bin/env python3
# run_try_again.py - "Try again" baseline: re-prompts incorrect answers with a short nudge (no hint)

import logging
from argparse import ArgumentParser
from pathlib import Path
from inference import solve_questions, try_again_questions
from io_jsonl import load_data, save_data
from prompts import answers_reformatting
from loading import load_dataset_module, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_try_again_pipeline(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try_again_path = output_dir / "try_again_inference.jsonl"

    if try_again_path.exists():
        logger.warning(f"Skipping: try-again results already exist at {try_again_path}")
        return

    logger.info(f"Loading dataset module: {args.dataset}")
    dataset_module = load_dataset_module(args.dataset)

    # Resolve where to look for existing initial inference results
    if args.initial_results_dir:
        initial_results_path = Path(args.initial_results_dir) / "initial_inference.jsonl"
    else:
        initial_results_path = output_dir / "initial_inference.jsonl"

    logger.info(f"Loading model and tokenizer: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device_map,
        use_flash_attention=args.use_flash_attention,
        compile_model=args.compile_model,
    )

    if initial_results_path.exists():
        logger.info(f"Loading existing initial inference results from {initial_results_path}")
        initial_results = load_data(initial_results_path)
    else:
        logger.info(f"No initial inference found at {initial_results_path} — running stage 1 first...")
        raw_data = load_data(args.input_path)
        if args.max_samples:
            raw_data = raw_data[:args.max_samples]
        initial_results = solve_questions(
            raw_data, model, tokenizer, dataset_module,
            model_name=args.model_path,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        save_data(initial_results, output_dir / "initial_inference.jsonl")

    wrong_only = [
        ex for ex in initial_results
        if ex.get("is_correct") is False and ex.get("predicted_answer") is not None
    ]
    logger.info(f"Found {len(wrong_only)} incorrect answers to re-prompt.")

    if not wrong_only:
        logger.info("No incorrect answers found. Nothing to do.")
        return

    logged_try_again_message = args.try_again_message

    if args.model_path in (
        "Qwen/Qwen2.5-Math-1.5B-instruct",
        "Qwen/Qwen2.5-Math-7B-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    ):
        logged_try_again_message = answers_reformatting(logged_try_again_message)

    logger.info(f'Running try-again inference with message: "{logged_try_again_message}"')
    try_again_results = try_again_questions(
        wrong_only, model, tokenizer, dataset_module,
        model_name=args.model_path,
        try_again_message=args.try_again_message,
        max_tokens=args.max_tokens,
        max_attempts=1, 
        batch_size=args.batch_size,
    )
    save_data(try_again_results, try_again_path)
    logger.info(f"Done. Results saved to {try_again_path}")


def main():
    parser = ArgumentParser(description="Try-again baseline: re-prompt incorrect answers with a short nudge.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--initial_results_dir", type=str, default=None,
                        help="Directory containing an existing initial_inference.jsonl to reuse. "
                             "If omitted, looks in --output_dir (or runs stage 1 if not found).")
    parser.add_argument("--try_again_message", type=str, default="Your previous answer was incorrect. Return ONLY the final answer inside <ans></ans>.",
                        help='Short prompt to re-prompt the model (default: "Try again.")')
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "single"])
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--no_flash_attention", dest="use_flash_attention", action="store_false")
    parser.add_argument("--compile_model", action="store_true")

    args = parser.parse_args()
    run_try_again_pipeline(args)


if __name__ == "__main__":
    main()
