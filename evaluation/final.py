#!/usr/bin/env python3
"""
Build one CSV row per (model, dataset, max_tokens) from results/all_results.

Expected folder structure:
results/all_results/<MODEL>/<DATASET>/<MAXTOKENS>/
  - initial_inference.jsonl
  - hints.jsonl
  - post_hint_inference.jsonl

Outputs metrics.csv with:
- n_questions
- initial_accuracy        = initial_correct / total
- post_hint_accuracy      = (initial_correct + n_corrected) / total
- n_incorrect_initial     = total - initial_correct
- n_corrected             = initially incorrect AND post-hint correct
- correction_rate         = n_corrected / n_incorrect_initial
- delta_accuracy          = post_hint_accuracy - initial_accuracy
- initial_tokens_mean     = mean tokens in initial full_output
- hint_tokens_mean        = mean tokens in hints full_output
- post_hint_tokens_mean   = mean tokens in post-hint full_output
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from transformers import AutoTokenizer


# Folder-name -> HF tokenizer id.
TOKENIZER_NAME_BY_MODEL: Dict[str, str] = {
    "DeepSeek-R1-0528-Qwen3-8B": "Qwen/Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
    "Phi-4-mini-reasoning": "microsoft/Phi-4-mini-instruct",
    "Qwen2.5-Math-1.5B-instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen2.5-Math-7B-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
}


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def mean_token_count(texts: List[str], tokenizer) -> float:
    if not texts:
        return 0.0
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    lengths = [len(ids) for ids in enc["input_ids"]]
    return sum(lengths) / len(lengths)


@dataclass
class FileStats:
    total: int
    n_correct: int
    is_correct_by_id: Dict[str, bool]
    full_outputs: List[str]


def load_inference_stats(jsonl_path: str) -> FileStats:
    is_correct_by_id: Dict[str, bool] = {}
    full_outputs: List[str] = []
    total = 0
    n_correct = 0

    for obj in iter_jsonl(jsonl_path):
        total += 1
        qid = str(obj["id"])  # IDs can be ints or strings; treat as string always
        ok = bool(obj["is_correct"])
        is_correct_by_id[qid] = ok
        if ok:
            n_correct += 1
        full_outputs.append(obj["full_output"])

    return FileStats(
        total=total,
        n_correct=n_correct,
        is_correct_by_id=is_correct_by_id,
        full_outputs=full_outputs,
    )


def find_leaf_runs(root_dir: str) -> List[Tuple[str, str, str, str]]:
    runs: List[Tuple[str, str, str, str]] = []
    if not os.path.isdir(root_dir):
        return runs

    for model in sorted(os.listdir(root_dir)):
        model_path = os.path.join(root_dir, model)
        if not os.path.isdir(model_path):
            continue

        for dataset in sorted(os.listdir(model_path)):
            dataset_path = os.path.join(model_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for max_tokens in sorted(os.listdir(dataset_path)):
                leaf_path = os.path.join(dataset_path, max_tokens)
                if not os.path.isdir(leaf_path):
                    continue

                init_p = os.path.join(leaf_path, "initial_inference.jsonl")
                hint_p = os.path.join(leaf_path, "hints.jsonl")
                post_p = os.path.join(leaf_path, "post_hint_inference.jsonl")

                if os.path.isfile(init_p) and os.path.isfile(hint_p) and os.path.isfile(post_p):
                    runs.append((model, dataset, max_tokens, leaf_path))

    return runs


def build_csv(results_root: str, out_csv: str) -> None:
    runs = find_leaf_runs(results_root)
    if not runs:
        raise RuntimeError(f"No runs found under: {results_root}")

    tokenizers: Dict[str, object] = {}
    rows: List[dict] = []

    for model, dataset, max_tokens, leaf_path in runs:
        if model not in TOKENIZER_NAME_BY_MODEL:
            raise KeyError(
                f"Missing tokenizer mapping for model folder '{model}'. "
                f"Add it to TOKENIZER_NAME_BY_MODEL."
            )

        if model not in tokenizers:
            tok_name = TOKENIZER_NAME_BY_MODEL[model]
            tokenizers[model] = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

        tokenizer = tokenizers[model]

        init_path = os.path.join(leaf_path, "initial_inference.jsonl")
        hint_path = os.path.join(leaf_path, "hints.jsonl")
        post_path = os.path.join(leaf_path, "post_hint_inference.jsonl")

        init = load_inference_stats(init_path)
        post = load_inference_stats(post_path)  # still used to compute n_corrected correctly

        hint_full_outputs = [obj["full_output"] for obj in iter_jsonl(hint_path)]

        n_questions = init.total
        n_correct_init = init.n_correct
        n_incorrect_initial = n_questions - n_correct_init

        # Count corrected: incorrect initially AND correct post-hint
        n_corrected = 0
        for qid, init_ok in init.is_correct_by_id.items():
            post_ok = post.is_correct_by_id.get(qid, False)
            if (not init_ok) and post_ok:
                n_corrected += 1

        correction_rate = (n_corrected / n_incorrect_initial) if n_incorrect_initial else 0.0

        # Accuracy definitions
        initial_accuracy = (n_correct_init / n_questions) if n_questions else 0.0

        # post_hint_accuracy = (initial_correct + corrected) / total
        post_hint_accuracy = ((n_correct_init + n_corrected) / n_questions) if n_questions else 0.0

        delta_accuracy = post_hint_accuracy - initial_accuracy

        initial_tokens_mean = mean_token_count(init.full_outputs, tokenizer)
        hint_tokens_mean = mean_token_count(hint_full_outputs, tokenizer)
        post_hint_tokens_mean = mean_token_count(post.full_outputs, tokenizer)

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "max_tokens": max_tokens,
                "n_questions": n_questions,
                "initial_accuracy": initial_accuracy,
                "post_hint_accuracy": post_hint_accuracy,
                "n_incorrect_initial": n_incorrect_initial,
                "n_corrected": n_corrected,
                "correction_rate": correction_rate,
                "delta_accuracy": delta_accuracy,
                "initial_tokens_mean": initial_tokens_mean,
                "hint_tokens_mean": hint_tokens_mean,
                "post_hint_tokens_mean": post_hint_tokens_mean,
            }
        )

    df = pd.DataFrame(rows).sort_values(["model", "dataset", "max_tokens"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (rows={len(df)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        default="results/all_results",
        help="Path to results/all_results",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="evaluation/metrics.csv",
        help="Output CSV path (default: evaluation/metrics.csv)",
    )
    args = parser.parse_args()
    build_csv(args.results_root, args.out_csv)


if __name__ == "__main__":
    main()
