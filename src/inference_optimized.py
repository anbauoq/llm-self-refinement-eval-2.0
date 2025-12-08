#!/usr/bin/env python3
# inference_optimized.py - Faster batched inference with FlashAttention
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import torch
from tqdm import tqdm
import logging

from utils import (
    format_prompt,
    format_hint_prompt,
    extract_cot,
    is_valid_hint,
    exact_match,
    _parse_max_new_tokens,
    extract_hint_text
)

# --- Constants ---
DEFAULT_SOLVE_MAX_TOKENS = 2048
DEFAULT_HINT_MAX_TOKENS = 1024
RETRY_SEED_BASE = 21
DEFAULT_BATCH_SIZE = 8

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Central builder for model.generate kwargs with batching support."""
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


def _batch_data(data: List, batch_size: int) -> List[List]:
    """Split data into batches."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def _strip_prompt_from_outputs(
    output_ids: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """
    Strip the prompt tokens from the generated sequence.

    Works for decoder-only models with padding on either side, and is safe-ish
    for encoder-decoder models (falls back to returning the whole output if
    it's shorter than prompt_length).
    """
    if output_ids.size(0) > prompt_length:
        return output_ids[prompt_length:]
    return output_ids



def solve_questions(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    dataset_module,
    inject_hint: bool = False,
    max_attempts: int = 2,
    max_tokens: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Batched inference for faster processing, with per-item retries.
    """
    data_list = list(data) #just making a list of dictionaries, our dataset, each dict is one question with its corresponding features
    results: List[Dict[str, Any]] = [] # just initializing object where results will be stored?
    dataset_name = dataset_module.__name__.split(".")[-1] # since dataset_module is given by utils.asdiv, extracting dataset name in this way

    # Process in batches
    batches = _batch_data(data_list, batch_size) # dividing our data into batches, list of lists, each of which contains batch_size dictionaries

    with torch.inference_mode(): 
        retry_suffix_text = (
        "\nIMPORTANT FORMAT REMINDER:\n"
        "- You must show your full reasoning between <cot_start> and <cot_end>.\n"
        "- Then, on a NEW line, you MUST output a SINGLE final answer in the exact format "
        "required by the instructions above (for this question type).\n"
        "<ans>X</ans>"
        "You are NOT allowed to output <ans>None</ans> or anything other than a single letter."
    )
        retry_suffix_ids: List[int] = tokenizer(
            retry_suffix_text,
            add_special_tokens=False,
        )["input_ids"]

        for batch in tqdm(batches, desc=f"Solving questions (batch_size={batch_size})"): # outer loop, considering one batch at a time
            
            # Prepare batch: process items + base prompts (without retry suffix) 
            processed_batch: List[Dict[str, Any]] = [] # initializing a list that is gonna contain each question processed - dictionaries with features we need
            prompts_batch: List[str] = [] # initializing a list where generated prompts are going to be stored

            for item in batch: # considering one question (dictionary) at a time, want to build the data structure that contains everything for further inference!
                # Process item
                if "ground_truth" not in item: # when doing our first step and the data is not procecessed yet, i.e. no ground_truth etc etc
                    
                    processed = dataset_module.process_item(item)
                    
                else: # when doing step 3 everything is already processed
                    processed = {
                        "id": item["id"],
                        "question": item["question"],
                        "answer": item["ground_truth"]
                    }

                # just creating prompt
                base_prompt = format_prompt(
                    processed["question"],
                    inject_hint=inject_hint,
                    hint=item.get("hint_sentence", ""),
                    dataset_name=dataset_name,
                )

                processed_batch.append(processed) # must have a list with all questions from the given batch
                prompts_batch.append(base_prompt) # here corresponding prompts must be added


            
            # pre-tokenize base prompts once per batch ---
            base_encoded = tokenizer( 
                prompts_batch,
                add_special_tokens=True,
                return_tensors=None,
                padding=False
            )

            
            base_input_ids: List[List[int]] = base_encoded["input_ids"] # tokenizations for all questions in the current batch

            


            batch_size_actual = len(batch) # just for the case when last batch is smaller
            
            # One slot per item in the batch; None means "still unresolved"
            
            batch_results: List[Optional[Dict[str, Any]]] = [None] * batch_size_actual # keeping status weather pred_answer succesfully extracted or no, initially oll are not so None
            
            pending_indices = list(range(batch_size_actual)) # initially all indices, all questions are pending (unanswered)

            for attempt in range(max_attempts):

                # where im going to come at the lasttttttt
                if not pending_indices:
                    break  # all items in this batch have been resolved

    
                is_retry = attempt > 0

                # Build tokenized inputs only for unresolved items (no re-tokenizing full text)
                current_input_ids: List[List[int]] = []
                current_indices: List[int] = []
                for idx in pending_indices:
                    ids = list(base_input_ids[idx])  # copy to avoid mutating base
                    if is_retry:
                        ids = ids + retry_suffix_ids
                    current_input_ids.append(ids)
                    current_indices.append(idx)

                # Pad unresolved inputs to tensor form
                padded = tokenizer.pad(
                    {"input_ids": current_input_ids},
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in padded.items()}

                # All rows have the same sequence length after padding
                prompt_length = inputs["input_ids"].shape[1]

                gen_kwargs = _build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    default_tokens=DEFAULT_SOLVE_MAX_TOKENS,
                    is_retry=is_retry,
                    attempt_num=attempt,
                    temperature=0.7,
                    top_p=0.95,
                )

                # Generate for unresolved subset
                output_ids = model.generate(**inputs, **gen_kwargs)

                # Decode and update only unresolved items
                for local_idx, output in enumerate(output_ids):
                    global_idx = current_indices[local_idx]
                    processed = processed_batch[global_idx]

                    new_ids = _strip_prompt_from_outputs(output, prompt_length)
                    trimmed_decoded = tokenizer.decode(
                        new_ids, skip_special_tokens=True
                    ).strip()

                    pred_answer = dataset_module.extract_answer(trimmed_decoded) or ""
                    if (not pred_answer) or (pred_answer == "no_final_answer"):
                        continue

                    cot = extract_cot(trimmed_decoded)
                    is_correct = exact_match(processed["answer"], pred_answer)

                    batch_results[global_idx] = {
                        "id": processed["id"],
                        "question": processed["question"],
                        "chain_of_thought": cot,
                        "full_output": trimmed_decoded,
                        "ground_truth": processed["answer"],
                        "predicted_answer": pred_answer,
                        "is_correct": is_correct
                    }

                # Keep only those still unresolved for the next attempt
                pending_indices = [i for i in pending_indices if batch_results[i] is None]

            # Fill in failures for items that never produced a valid answer
            for idx, res in enumerate(batch_results):
                if res is None:
                    processed = processed_batch[idx]
                    batch_results[idx] = {
                        "id": processed["id"],
                        "question": processed["question"],
                        "full_output": False,
                        "chain_of_thought": False,
                        "predicted_answer": False,
                        "ground_truth": processed["answer"],
                        "is_correct": False
                    }

            # Extend global results in batch order
            results.extend(batch_results)

    return results


def generate_hints(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    dataset_name: str,
    num_attempts: int = 3,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Batched hint generation with per-item retries and proper fallback to last decoded hint.
    """
    data_list = list(data)
    hints: List[Dict[str, Any]] = []

    batches = _batch_data(data_list, batch_size)

    with torch.inference_mode():
        for batch in tqdm(batches, desc=f"Generating hints (batch_size={batch_size})"):
            # Build base prompts for the entire batch (one per item)
            prompts_batch: List[str] = []
            for item in batch:
                prompt = format_hint_prompt(
                    item["question"],
                    item.get("predicted_answer", ""),
                    item.get("chain_of_thought", ""),
                    item["ground_truth"],
                    dataset_name=dataset_name
                )
                prompts_batch.append(prompt)

            batch_size_actual = len(batch)
            batch_hints: List[Optional[Dict[str, Any]]] = [None] * batch_size_actual
            pending_indices = list(range(batch_size_actual))
            last_decoded: Dict[int, str] = {}

            for attempt in range(num_attempts):
                if not pending_indices:
                    break

                # Tokenize prompts only for unresolved items
                current_prompts = [prompts_batch[i] for i in pending_indices]

                inputs = tokenizer(
                    current_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(model.device)

                prompt_length = inputs["input_ids"].shape[1]

                # For hints we want sampling every time -> is_retry=True
                gen_kwargs = _build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    default_tokens=DEFAULT_HINT_MAX_TOKENS,
                    is_retry=True,
                    attempt_num=attempt,
                    temperature=temperature,
                    top_p=0.95
                )

                out_ids = model.generate(**inputs, **gen_kwargs)

                # Decode and validate hints for unresolved items
                for local_idx, output in enumerate(out_ids):
                    global_idx = pending_indices[local_idx]
                    item = batch[global_idx]

                    new_ids = _strip_prompt_from_outputs(output, prompt_length)
                    decoded = tokenizer.decode(
                        new_ids, skip_special_tokens=True
                    ).strip()

                    # Remember last decoded attempt for fallback (raw text)
                    last_decoded[global_idx] = decoded

                    # Extract just the inner <hint>...</hint> if present
                    hint_text = extract_hint_text(decoded)
                    if hint_text:
                        item_with_hint = item.copy()
                        item_with_hint["hint_sentence"] = hint_text
                        batch_hints[global_idx] = item_with_hint

                # Filter out those that already have a valid hint
                pending_indices = [i for i in pending_indices if batch_hints[i] is None]

            # Add items with last attempt's hint if validation failed
            for idx, res in enumerate(batch_hints):
                if res is None:
                    item_with_hint = batch[idx].copy()
                    raw = last_decoded.get(idx, "")
                    item_with_hint["hint_sentence"] = extract_hint_text(raw) if raw else ""
                    batch_hints[idx] = item_with_hint

            hints.extend(batch_hints)

    return hints