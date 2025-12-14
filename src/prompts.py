#!/usr/bin/env python3
# prompts.py
from __future__ import annotations

import re
from pathlib import Path


def gemma_prompt_formatting(prompt: str) -> str:
    return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"


def phi4_prompt_formatting(prompt: str) -> str:
    return f"<|user|>{prompt}<|end|><|assistant|>"


def llama_prompt_formatting(prompt: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def answers_reformatting(text: str) -> str:

    # Rewrite instruction phrase variants
    text = re.sub(
        r'between\s*<\s*ans\s*>\s*and\s*<\s*/\s*ans\s*>',
        r'inside the \\boxed{}',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'between\s*<\s*ans\s*>\s*<\s*/\s*ans\s*>',
        r'inside the \\boxed{}',
        text,
        flags=re.IGNORECASE,
    )

    # Convert tags to \boxed{...}
    pattern = re.compile(
        r"<\s*ans\s*>\s*(.*?)\s*<\s*/\s*ans\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = text.rstrip() + "<think>\n"

    return pattern.sub(lambda m: rf"\boxed{{{m.group(1).strip()}}}", text)


def format_initial_prompt(
    question: str,
    model: str,
    dataset_name: str = "gsm8k",
) -> str:
    """
    Load and format the dataset-specific answer prompt.
    """

    prompt_file_map = {
        "asdiv": "arithmetic_prompt.txt",
        "gsm8k": "arithmetic_prompt.txt",
    }
    prompt_filename = prompt_file_map.get(dataset_name, f"{dataset_name}_prompt.txt")

    prompt_path = Path("prompts") / prompt_filename

    template = prompt_path.read_text(encoding="utf-8")
    body = template.format(question=question.strip())

    if model in ("Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-7B"):
        return body

    if model in ("microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning"):
        return phi4_prompt_formatting(body)

    if model == "google/gemma-2-2b-it":
        return gemma_prompt_formatting(body)

    if model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        return llama_prompt_formatting(body)

    if model in (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    ):
        return answers_reformatting(body)

    return body


def format_post_hint_prompt(
    question: str,
    model: str,
    hint: str = "",
    dataset_name: str = "gsm8k",
) -> str:
    """
    Load and format the dataset-specific answer prompt with a hint.
    """

    prompt_file_map = {
        "asdiv": "arithmetic_prompt.txt",
        "gsm8k": "arithmetic_prompt.txt",
    }
    prompt_filename = prompt_file_map.get(dataset_name, f"{dataset_name}_prompt.txt")

    prompt_path = Path("prompts") / prompt_filename

    template = prompt_path.read_text(encoding="utf-8")
    body = template.format(question=question.strip())

    # always build the (possibly) hinted prompt string first
    combined = f"{hint.strip()}\n\n{body}"

    # --- model-specific formatting ---
    if model in ("Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-7B"):
        return combined

    if model in ("microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning"):
        return phi4_prompt_formatting(combined)

    if model == "google/gemma-2-2b-it":
        return gemma_prompt_formatting(combined)

    if model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        return llama_prompt_formatting(combined)

    if model in (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    ):
        return answers_reformatting(combined)

    return combined


def format_hint_prompt(
    question: str,
    predicted_answer: str,
    chain_of_thought: str,
    correct_answer: str,
    dataset_name: str,
) -> str:
    """
    Load and format the *dataset-specific* hint-generation prompt.
    asdiv and gsm8k share the same arithmetic hint prompt file.
    """
    dataset_key = (dataset_name or "").strip().lower()

    hint_prompt_map = {
        "asdiv": "prompts/hint_prompt_arithmetic.txt",
        "gsm8k": "prompts/hint_prompt_arithmetic.txt",

        "aqua": "prompts/hint_prompt_aqua.txt",
        "ar_lsat": "prompts/hint_prompt_ar_lsat.txt",
        "sports": "prompts/hint_prompt_sports.txt",
    }

    filename = hint_prompt_map.get(dataset_key)

    template = Path(filename).read_text(encoding="utf-8")
    return template.format(
        question=question.strip(),
        predicted_answer=str(predicted_answer).strip(),
        chain_of_thought=str(chain_of_thought).strip(),
        correct_answer=str(correct_answer).strip(),
    )