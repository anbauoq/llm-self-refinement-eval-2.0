#!/usr/bin/env python3
# inference.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import torch
from tqdm import tqdm

from utils import (
    format_prompt,
    format_hint_prompt,
    extract_cot,
    is_valid_hint,
    contains_bad_phrases,
    exact_match,
    _parse_max_new_tokens,
)

# --- Constants ---
DEFAULT_SOLVE_MAX_TOKENS = 256
DEFAULT_HINT_MAX_TOKENS = 128
RETRY_SEED_BASE = 21


# Helpers (centralized)

def _resolve_pad_eos(tokenizer) -> tuple[Optional[int], Optional[int]]:
    """Pick safe pad/eos once and reuse."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    return pad_id, eos_id


def _build_generation_kwargs(
    tokenizer,
    max_tokens: Optional[int],
    default_tokens: int,
    *,
    is_retry: bool,
    attempt_num: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """
    Central builder for model.generate kwargs.
    - Uses _parse_max_new_tokens (no 'full' mode).
    - First attempt greedy; retries sample with temp/top_p.
    - Deterministic seed on retries.
    """
    pad_id, eos_id = _resolve_pad_eos(tokenizer)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": _parse_max_new_tokens(max_tokens, default=default_tokens),
        "pad_token_id": pad_id,
        "use_cache": True,
        "do_sample": False,
    }
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id

    if is_retry:
        torch.manual_seed(RETRY_SEED_BASE + attempt_num)
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    return gen_kwargs


# Core functions

def solve_questions(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    dataset_module,
    inject_hint: bool = False,
    max_attempts: int = 2,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run model inference (initial or post-hint) and return structured results with
    token-accurate decoding and robust retries.
    """
    results: List[Dict[str, Any]] = []
    dataset_name = dataset_module.__name__.split(".")[-1]

    with torch.inference_mode():
        for item in tqdm(data, desc="Solving questions"):
            # Dataset item handling (match 'first file' logic):
            # - If NO 'ground_truth' in item => treat as raw and call process_item
            # - If 'ground_truth' exists => item is already structured (e.g., from hint stage)
            if "ground_truth" not in item:
                processed = dataset_module.process_item(item)
            else:
                processed = {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["ground_truth"],
                }

            base_prompt = format_prompt(
                processed["question"],
                inject_hint=inject_hint,
                hint=item.get("hint_sentence", ""),
                dataset_name=dataset_name,
            )

            trimmed_decoded = ""
            pred_answer = ""
            cot = ""
            is_correct = False

            for attempt in range(max_attempts):
                is_retry = attempt > 0
                prompt = base_prompt
                if is_retry:
                    prompt += "\nPlease provide a step-by-step breakdown and end with your final answer."

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[1]

                gen_kwargs = _build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    default_tokens=DEFAULT_SOLVE_MAX_TOKENS,
                    is_retry=is_retry,
                    attempt_num=attempt,
                    temperature=0.7,
                    top_p=0.95,
                )

                output_ids = model.generate(**inputs, **gen_kwargs)[0]
                new_ids = output_ids[input_len:]
                trimmed_decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

                pred_answer = dataset_module.extract_answer(trimmed_decoded) or ""
                if pred_answer:
                    cot = extract_cot(trimmed_decoded)
                    is_correct = exact_match(processed["answer"], pred_answer)
                    break  # success

            results.append(
                {
                    "id": processed["id"],
                    "question": processed["question"],
                    "chain_of_thought": cot,
                    "full_output": trimmed_decoded,
                    "ground_truth": processed["answer"],
                    "predicted_answer": pred_answer,
                    "is_correct": is_correct,
                }
            )

    return results


def generate_hints(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    num_attempts: int = 3,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate helpful hints using the model and ground-truth answers, avoiding answer leakage.
    Uses token-accurate decoding, deterministic retry sampling, and safe termination settings.
    """
    hints: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for item in tqdm(data, desc="Generating hints"):
            prompt = format_hint_prompt(
                item["question"],
                item.get("predicted_answer", ""),
                item.get("chain_of_thought", ""),
                item["ground_truth"],
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            hint_sentence = ""
            for attempt in range(num_attempts):
                gen_kwargs = _build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    default_tokens=DEFAULT_HINT_MAX_TOKENS,
                    is_retry=True,
                    attempt_num=attempt,
                    temperature=temperature,
                    top_p=0.95,
                )

                out_ids = model.generate(**inputs, **gen_kwargs)[0]
                decoded = tokenizer.decode(out_ids[input_len:], skip_special_tokens=True).strip()

                if is_valid_hint(decoded, item["ground_truth"]) and not contains_bad_phrases(
                    decoded, item["ground_truth"]
                ):
                    hint_sentence = decoded
                    break

                # keep last attempt's text even if not ideal
                hint_sentence = decoded

            item_with_hint = item.copy()
            item_with_hint["hint_sentence"] = hint_sentence
            hints.append(item_with_hint)

    return hints
