#!/usr/bin/env python3
# inference.py
from __future__ import annotations
import torch
import logging
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional
from prompts import format_initial_prompt, format_post_hint_prompt, format_hint_prompt
from hints import extract_hint_text, is_valid_hint, strip_answer_from_hint
from utils import ( 
    extract_cot,
    exact_match,
    parse_max_new_tokens,
    resolve_pad_eos,
    batch_data,
    strip_prompt_from_outputs
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def solve_questions(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    dataset_module,
    model_name,
    inject_hint: bool = False,
    max_attempts: int = 3,
    max_tokens: Optional[int] = None,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    data_list = list(data) #just making a list of dictionaries, our dataset, each dict is one question with its corresponding features
    results: List[Dict[str, Any]] = [] # just initializing object where results will be stored?
    dataset_name = dataset_module.__name__.split(".")[-1]

    # Process in batches
    batches = batch_data(data_list, batch_size) # dividing our data into batches, list of lists, each of which contains batch_size dictionaries

    with torch.inference_mode(): 
        
        retry_suffix_text = """ 
        You must ONLY answer this question by writing your full reasoning between <think> and </think>,
        and at the end stating the final answer. Do NOT output anything else. 
        """
  
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

                if inject_hint:
                    base_prompt = format_post_hint_prompt(
                        question = processed["question"],
                        model = model_name,
                        hint=item.get("hint_sentence", ""),
                        dataset_name=dataset_name,)
                    
                else:
                    base_prompt = format_initial_prompt(
                        question = processed["question"],
                        model = model_name,
                        dataset_name=dataset_name)

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
            
            
            batch_results: List[Optional[Dict[str, Any]]] = [None] * batch_size_actual # keeping status weather pred_answer succesfully extracted or no, initially oll are not so None

            last_raw_outputs: List[Optional[str]] = [None] * batch_size_actual # storing all outputs so in case of failure the full output is available
            
            pending_indices = list(range(batch_size_actual)) # initially all indices, all questions are pending (unanswered)

            last_attempt_is_retry: List[bool] = [False] * batch_size_actual

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

                # all rows have the same sequence length after padding
                prompt_length = inputs["input_ids"].shape[1]

                pad_id, eos_id = resolve_pad_eos(tokenizer)
                max_new = parse_max_new_tokens(max_tokens, default=2048)

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new,
                    "min_new_tokens": min(64, max_new),
                    "pad_token_id": pad_id,
                    "use_cache": True,
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95
                }
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id

                if is_retry:
                    torch.manual_seed(21 + attempt)

                output_ids = model.generate(**inputs, **gen_kwargs)

                # decode and update only unresolved items
                for local_idx, output in enumerate(output_ids):
                    global_idx = current_indices[local_idx]
                    processed = processed_batch[global_idx]
                
                    new_ids = strip_prompt_from_outputs(output, prompt_length)
                    trimmed_decoded = tokenizer.decode(
                        new_ids, skip_special_tokens=True
                    ).strip()
                                    
                    last_raw_outputs[global_idx] = trimmed_decoded
                                    
                    cot = extract_cot(trimmed_decoded)
                    pred_answer = dataset_module.extract_answer(trimmed_decoded) or ""
                                    
                    if (not pred_answer) or (pred_answer == "no_final_answer"):
                        continue
                
                    is_correct = exact_match(processed["answer"], pred_answer)

                    if is_retry:
                        last_attempt_is_retry[global_idx] = True

                    batch_results[global_idx] = {
                        "id": processed["id"],
                        "question": processed["question"],
                        "chain_of_thought": cot,
                        "full_output": trimmed_decoded,
                        "ground_truth": processed["answer"],
                        "predicted_answer": pred_answer,
                        "is_correct": is_correct,
                        "from_retry": last_attempt_is_retry[global_idx]
                    }


                # Keep only those still unresolved for the next attempt
                pending_indices = [i for i in pending_indices if batch_results[i] is None]

            # Fill in failures for items that never produced a valid answer
            for idx, res in enumerate(batch_results):
                if res is None:
                    processed = processed_batch[idx]
                    raw_out = last_raw_outputs[idx]
                    if raw_out is not None:
                        cot_fallback = extract_cot(raw_out)
                        batch_results[idx] = {
                            "id": processed["id"],
                            "question": processed["question"],
                            "full_output": raw_out,
                            "chain_of_thought": cot_fallback,
                            "predicted_answer": None,
                            "ground_truth": processed["answer"],
                            "is_correct": None,
                            "from_retry": last_attempt_is_retry[idx],
                        }
                    else:
                        batch_results[idx] = {
                            "id": processed["id"],
                            "question": processed["question"],
                            "full_output": "",
                            "chain_of_thought": None,
                            "predicted_answer": None,
                            "ground_truth": processed["answer"],
                            "is_correct": None,
                            "from_retry": last_attempt_is_retry[idx]
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
    max_tokens: Optional[int] = None,
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    Batched hint generation with per-item retries and proper fallback to last decoded hint.
    """
    data_list = list(data)
    hints: List[Dict[str, Any]] = []

    batches = batch_data(data_list, batch_size)

    with torch.inference_mode():
        for batch in tqdm(batches, desc=f"Generating hints (batch_size={batch_size})"):
            # Build base prompts for the entire batch (one per item)
            prompts_batch: List[str] = []
            for item in batch:
                prompt = format_hint_prompt(
                    item["question"],
                    item.get("predicted_answer"),
                    item.get("chain_of_thought"),
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

                pad_id, eos_id = resolve_pad_eos(tokenizer)
                max_new = parse_max_new_tokens(max_tokens, default=1024)

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new,
                    "min_new_tokens": min(64, max_new),
                    "pad_token_id": pad_id,
                    "use_cache": True,
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95
                }
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id

                torch.manual_seed(21 + attempt)

                out_ids = model.generate(**inputs, **gen_kwargs)

                # Decode and validate hints for unresolved items
                for local_idx, output in enumerate(out_ids):
                    global_idx = pending_indices[local_idx]
                    item = batch[global_idx]

                    new_ids = strip_prompt_from_outputs(output, prompt_length)
                    decoded = tokenizer.decode(
                        new_ids, skip_special_tokens=True
                    ).strip()

                    # Remember last decoded attempt for fallback (raw text)
                    last_decoded[global_idx] = decoded

                    # Extract hint sentences if present
                    hint_text = extract_hint_text(decoded)

                    # Accept only non-leaking hints here
                    if hint_text and is_valid_hint(hint_text, item["ground_truth"], dataset_name):
                        item_with_hint = item.copy()
                        item_with_hint["hint_sentence"] = hint_text
                        batch_hints[global_idx] = item_with_hint

                # Filter out those that already have a valid hint
                pending_indices = [i for i in pending_indices if batch_hints[i] is None]

            # Add items with last attempt's hint if validation failed
            for idx, res in enumerate(batch_hints):
                if res is None:
                    item_with_hint = batch[idx].copy()
                    raw = last_decoded.get(idx)
                    hint_text = extract_hint_text(raw)

                    if hint_text:
                        # If all attempts leaked, strip the answer out and reuse the rest
                        if not is_valid_hint(hint_text, item_with_hint["ground_truth"], dataset_name):
                            hint_text = strip_answer_from_hint(
                                hint_text,
                                item_with_hint["ground_truth"],
                            )

                    item_with_hint["hint_sentence"] = hint_text
                    batch_hints[idx] = item_with_hint

            hints.extend(batch_hints)

    return hints
