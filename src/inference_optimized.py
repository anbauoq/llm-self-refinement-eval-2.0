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
    contains_bad_phrases,
    exact_match,
    _parse_max_new_tokens,
)

# --- Constants ---
DEFAULT_SOLVE_MAX_TOKENS = 256
DEFAULT_HINT_MAX_TOKENS = 128
RETRY_SEED_BASE = 21
DEFAULT_BATCH_SIZE = 8  # Process 8 samples at once

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
    Batched inference for faster processing.
    """
    data_list = list(data)
    results: List[Dict[str, Any]] = []
    dataset_name = dataset_module.__name__.split(".")[-1]

    # Process in batches
    batches = _batch_data(data_list, batch_size)
    
    with torch.inference_mode():
        for batch in tqdm(batches, desc=f"Solving questions (batch_size={batch_size})"):
            # Prepare batch
            processed_batch = []
            prompts_batch = []
            
            for item in batch:
                # Process item
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
                
                processed_batch.append(processed)
                prompts_batch.append(base_prompt)
            
            # Process entire batch at once
            batch_results = []
            for attempt in range(max_attempts):
                is_retry = attempt > 0
                current_prompts = [p + ("\nPlease provide a step-by-step breakdown and end with your final answer." if is_retry else "") 
                                 for p in prompts_batch]
                
                # Tokenize batch with padding
                inputs = tokenizer(
                    current_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(model.device)
                
                input_lens = (inputs["attention_mask"].sum(dim=1)).tolist()
                
                gen_kwargs = _build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    default_tokens=DEFAULT_SOLVE_MAX_TOKENS,
                    is_retry=is_retry,
                    attempt_num=attempt,
                    temperature=0.7,
                    top_p=0.95,
                )
                
                # Generate for entire batch
                output_ids = model.generate(**inputs, **gen_kwargs)
                
                # Decode each output
                for idx, (output, input_len, processed) in enumerate(zip(output_ids, input_lens, processed_batch)):
                    new_ids = output[input_len:]
                    trimmed_decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    
                    pred_answer = dataset_module.extract_answer(trimmed_decoded) or ""
                    if pred_answer:
                        cot = extract_cot(trimmed_decoded)
                        is_correct = exact_match(processed["answer"], pred_answer)
                        
                        batch_results.append({
                            "id": processed["id"],
                            "question": processed["question"],
                            "chain_of_thought": cot,
                            "full_output": trimmed_decoded,
                            "ground_truth": processed["answer"],
                            "predicted_answer": pred_answer,
                            "is_correct": is_correct,
                        })
                        break  # Success for this item
                
                if len(batch_results) == len(batch):
                    break  # All items in batch succeeded
            
            # Add any remaining items that failed all attempts
            while len(batch_results) < len(batch):
                idx = len(batch_results)
                processed = processed_batch[idx]
                batch_results.append({
                    "id": processed["id"],
                    "question": processed["question"],
                    "chain_of_thought": "",
                    "full_output": "",
                    "ground_truth": processed["answer"],
                    "predicted_answer": "",
                    "is_correct": False,
                })
            
            results.extend(batch_results)
    
    return results


def generate_hints(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    num_attempts: int = 3,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Batched hint generation for faster processing.
    """
    data_list = list(data)
    hints: List[Dict[str, Any]] = []
    
    batches = _batch_data(data_list, batch_size)
    
    with torch.inference_mode():
        for batch in tqdm(batches, desc=f"Generating hints (batch_size={batch_size})"):
            prompts_batch = []
            for item in batch:
                prompt = format_hint_prompt(
                    item["question"],
                    item.get("predicted_answer", ""),
                    item.get("chain_of_thought", ""),
                    item["ground_truth"],
                )
                prompts_batch.append(prompt)
            
            batch_hints = []
            for attempt in range(num_attempts):
                # Tokenize batch
                inputs = tokenizer(
                    prompts_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(model.device)
                
                input_lens = (inputs["attention_mask"].sum(dim=1)).tolist()
                
                gen_kwargs = _build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    default_tokens=DEFAULT_HINT_MAX_TOKENS,
                    is_retry=True,
                    attempt_num=attempt,
                    temperature=temperature,
                    top_p=0.95,
                )
                
                out_ids = model.generate(**inputs, **gen_kwargs)
                
                # Decode and validate hints
                for idx, (output, input_len, item) in enumerate(zip(out_ids, input_lens, batch)):
                    decoded = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                    
                    if is_valid_hint(decoded, item["ground_truth"]) and not contains_bad_phrases(decoded, item["ground_truth"]):
                        item_with_hint = item.copy()
                        item_with_hint["hint_sentence"] = decoded
                        batch_hints.append(item_with_hint)
                        break
                
                if len(batch_hints) == len(batch):
                    break  # All hints validated
            
            # Add items with last attempt's hint if validation failed
            while len(batch_hints) < len(batch):
                idx = len(batch_hints)
                item_with_hint = batch[idx].copy()
                item_with_hint["hint_sentence"] = ""  # or use last decoded
                batch_hints.append(item_with_hint)
            
            hints.extend(batch_hints)
    
    return hints

