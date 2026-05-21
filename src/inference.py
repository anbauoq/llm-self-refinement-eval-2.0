#!/usr/bin/env python3
# inference.py
from __future__ import annotations
import torch
import logging
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional
from prompts import format_initial_prompt, format_post_hint_prompt, format_hint_prompt, answers_reformatting
from parsing import extract_cot, exact_match
from hints import extract_hint_text, is_valid_hint, strip_answer_from_hint
from generation import ( 
    resolve_pad_eos,
    batch_data,
    strip_prompt_from_outputs,
    encode_chat
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
    max_tokens: int = 2048,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    data_list = list(data)
    results: List[Dict[str, Any]] = []
    dataset_name = dataset_module.__name__.split(".")[-1]

    # Process in batches
    batches = batch_data(data_list, batch_size)

    with torch.inference_mode(): 
        
        followup_user_msg = (
            "Now return ONLY single final answer to the question inside <ans> </ans>."
        )

        if model_name in (
            "Qwen/Qwen2.5-Math-1.5B-instruct",
            "Qwen/Qwen2.5-Math-7B-instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        ):
            followup_user_msg = answers_reformatting(followup_user_msg)

        for batch in tqdm(batches, desc=f"Solving questions (batch_size={batch_size})"):
            
            # Prepare batch: process items + base prompts
            processed_batch: List[Dict[str, Any]] = []
            prompts_batch: List[str] = []

            for item in batch:
                
                # Process item
                if "ground_truth" not in item:
                    
                    processed = dataset_module.process_item(item)
                    
                else:
                    processed = {
                        "id": item["id"],
                        "question": item["question"],
                        "answer": item["ground_truth"],
                        "options": item.get("options", [])
                    }

                # Prepare prompt
                if inject_hint:
                    base_prompt = format_post_hint_prompt(
                        question = processed["question"],
                        model = model_name,
                        hint=item.get("hint_sentence"),
                        dataset_name=dataset_name,)
                    
                else:
                    base_prompt = format_initial_prompt(
                        question = processed["question"],
                        model = model_name,
                        dataset_name=dataset_name)

                processed_batch.append(processed)
                prompts_batch.append(base_prompt)


            

            # Pre-encode base prompts once per item.
            base_input_ids: List[List[int]] = []
            for p in prompts_batch:
                ids = encode_chat(
                    tokenizer,
                    messages=[{"role": "user", "content": p}],
                    add_generation_prompt=True,
                )
                base_input_ids.append(list(ids))

        

            batch_size_actual = len(batch)
            
            
            batch_results: List[Optional[Dict[str, Any]]] = [None] * batch_size_actual

            last_raw_outputs: List[Optional[str]] = [None] * batch_size_actual

            attempt_outputs: List[List[str]] = [[] for _ in range(batch_size_actual)]

            
            pending_indices = list(range(batch_size_actual))

            was_retried: List[bool] = [False] * batch_size_actual
            
            retry_logged: set[int] = set()



            for attempt in range(max_attempts):

                if not pending_indices:
                    break

    
                is_retry = attempt > 0
                if is_retry:
                    for idx in pending_indices:
                        was_retried[idx] = True
                        
                        if idx not in retry_logged:
                            qid = processed_batch[idx].get("id", idx)
                            prev = (last_raw_outputs[idx] or "")
                            logger.info(
                                f"[RETRY] dataset={dataset_name} id={qid} attempt={attempt+1}/{max_attempts} "
                                f"reason=no_valid_answer_extracted prev_chars={len(prev)} "
                                f"action=chat_continuation_followup"
                            )
                            retry_logged.add(idx)


                current_input_ids: List[List[int]] = []
                current_indices: List[int] = []

                for idx in pending_indices:
                    if not is_retry:
                        ids = list(base_input_ids[idx])
                    else:
                        prev = (last_raw_outputs[idx] or "").strip()

                        # Multi-turn chat continuation:
                        # user -> assistant(prev output) -> user(followup) -> assistant(to generate)
                        ids = encode_chat(
                            tokenizer,
                            messages=[
                                {"role": "user", "content": prompts_batch[idx]},
                                {"role": "assistant", "content": prev},
                                {"role": "user", "content": followup_user_msg},
                            ],
                            add_generation_prompt=True,
                        )
                        ids = list(ids)

                    current_input_ids.append(ids)
                    current_indices.append(idx)

                pad_id, eos_id = resolve_pad_eos(tokenizer)
                
                padded = tokenizer.pad(
                    {"input_ids": current_input_ids},
                    padding=True,
                    return_tensors="pt",
                )


                # Safety fallback: if some tokenizer still didn't return it, create it from padding
                if "attention_mask" not in padded:
                    padded["attention_mask"] = (padded["input_ids"] != pad_id).long()

                inputs = {k: v.to(model.device) for k, v in padded.items()}

                # All rows have the same sequence length after padding
                prompt_length = inputs["input_ids"].shape[1]


                # Set model-specific temperature
                temp = 0.6
                if model_name in (
                    "microsoft/Phi-4-mini-instruct",
                    "microsoft/Phi-4-mini-reasoning",
                ):
                    temp = 0.8

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": min(24, max_tokens),
                    "pad_token_id": pad_id,
                    "use_cache": True,
                    "do_sample": True,
                    "temperature": temp,
                    "top_p": 0.95,
                }

                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id


                output_ids = model.generate(**inputs, **gen_kwargs)

                # Decode and update only unresolved items
                for local_idx, output in enumerate(output_ids):
                    global_idx = current_indices[local_idx]
                    processed = processed_batch[global_idx]
                
                    new_ids = strip_prompt_from_outputs(output, prompt_length)
                    trimmed_decoded = tokenizer.decode(
                        new_ids, skip_special_tokens=True
                    ).strip()
                                    
                    last_raw_outputs[global_idx] = trimmed_decoded

                    attempt_outputs[global_idx].append(trimmed_decoded)

                                    
                    cot = extract_cot(trimmed_decoded)
                    options = processed.get("options", [])


                    if dataset_name == "aqua":
                        pred_answer = dataset_module.extract_answer(trimmed_decoded, options=options) or ""
                    else:
                        pred_answer = dataset_module.extract_answer(trimmed_decoded) or ""

                                    
                    if (not pred_answer) or (pred_answer == "no_final_answer"):
                        continue
                
                    is_correct = exact_match(processed["answer"], pred_answer)


                    if is_retry:
                        
                        qid = processed.get("id", global_idx)
                        logger.info(
                            f"[RETRY_SUCCESS] dataset={dataset_name} id={qid} attempt={attempt+1}/{max_attempts} "
                            f"pred={pred_answer} correct={is_correct}"
                        )


                    merged_full_output = (
                        "\n\n".join(attempt_outputs[global_idx])
                        if was_retried[global_idx] and len(attempt_outputs[global_idx]) > 1
                        else trimmed_decoded
                    )


                    batch_results[global_idx] = {
                        "id": processed["id"],
                        "question": processed["question"],
                        "chain_of_thought": cot,
                        "full_output": merged_full_output,
                        "ground_truth": processed["answer"],
                        "predicted_answer": pred_answer,
                        "is_correct": is_correct,
                        "was_retried": was_retried[global_idx]

                    }


                # Keep only those still unresolved for the next attempt
                pending_indices = [i for i in pending_indices if batch_results[i] is None]

            # Fill in failures for items that never produced a valid answer
            for idx, res in enumerate(batch_results):
                if res is None:
                    qid = processed_batch[idx].get("id", idx)
                    logger.warning(
                        f"[FAILED] dataset={dataset_name} id={qid} attempts={max_attempts} "
                        f"was_retried={was_retried[idx]} last_output_chars={len(last_raw_outputs[idx] or '')}"
                    )


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
                            "was_retried": was_retried[idx],
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
                            "was_retried": was_retried[idx]
                        }



            # Extend global results in batch order
            results.extend(batch_results)

    return results


def try_again_questions(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    dataset_module,
    model_name: str,
    try_again_message: str = "Try again.",
    max_attempts: int = 3,
    max_tokens: int = 2048,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Re-prompts incorrectly answered questions with a short nudge (no hint).
    Multi-turn: original prompt → model's wrong output → try_again_message.
    """
    data_list = list(data)
    results: List[Dict[str, Any]] = []
    dataset_name = dataset_module.__name__.split(".")[-1]

    boxed_models = (
        "Qwen/Qwen2.5-Math-1.5B-instruct",
        "Qwen/Qwen2.5-Math-7B-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    )

    if model_name in boxed_models:
        try_again_message = answers_reformatting(try_again_message)

    batches = batch_data(data_list, batch_size)

    with torch.inference_mode():
        followup_user_msg = (
            "Now return ONLY single final answer to the question inside <ans> </ans>."
        )

        if model_name in boxed_models:
            followup_user_msg = answers_reformatting(followup_user_msg)

        temp = 0.8 if model_name in (
            "microsoft/Phi-4-mini-instruct",
            "microsoft/Phi-4-mini-reasoning",
        ) else 0.6

        for batch in tqdm(batches, desc=f"Try-again inference (batch_size={batch_size})"):
            batch_size_actual = len(batch)
            batch_results: List[Optional[Dict[str, Any]]] = [None] * batch_size_actual
            last_raw_outputs: List[Optional[str]] = [None] * batch_size_actual
            attempt_outputs: List[List[str]] = [[] for _ in range(batch_size_actual)]
            pending_indices = list(range(batch_size_actual))
            was_retried: List[bool] = [False] * batch_size_actual
            retry_logged: set[int] = set()

            base_prompts: List[str] = []
            processed_batch: List[Dict[str, Any]] = []

            for item in batch:
                processed = {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["ground_truth"],
                    "options": item.get("options", []),
                }
                processed_batch.append(processed)
                base_prompts.append(
                    format_initial_prompt(item["question"], model_name, dataset_name)
                )

            for attempt in range(max_attempts):
                if not pending_indices:
                    break

                is_retry = attempt > 0

                if is_retry:
                    for idx in pending_indices:
                        was_retried[idx] = True
                        if idx not in retry_logged:
                            qid = processed_batch[idx].get("id", idx)
                            logger.info(
                                f"[RETRY] dataset={dataset_name} id={qid} "
                                f"attempt={attempt+1}/{max_attempts} "
                                f"reason=no_valid_answer_extracted "
                                f"action=chat_continuation_followup"
                            )
                            retry_logged.add(idx)

                current_input_ids: List[List[int]] = []
                current_indices: List[int] = []

                for idx in pending_indices:
                    orig_item = batch[idx]

                    if attempt == 0:
                        ids = encode_chat(
                            tokenizer,
                            messages=[
                                {"role": "user", "content": base_prompts[idx]},
                                {"role": "assistant", "content": orig_item.get("full_output", "")},
                                {"role": "user", "content": try_again_message},
                            ],
                            add_generation_prompt=True,
                        )
                    else:
                        prev = (last_raw_outputs[idx] or "").strip()
                        ids = encode_chat(
                            tokenizer,
                            messages=[
                                {"role": "user", "content": base_prompts[idx]},
                                {"role": "assistant", "content": orig_item.get("full_output", "")},
                                {"role": "user", "content": try_again_message},
                                {"role": "assistant", "content": prev},
                                {"role": "user", "content": followup_user_msg},
                            ],
                            add_generation_prompt=True,
                        )

                    current_input_ids.append(list(ids))
                    current_indices.append(idx)

                pad_id, eos_id = resolve_pad_eos(tokenizer)
                padded = tokenizer.pad(
                    {"input_ids": current_input_ids},
                    padding=True,
                    return_tensors="pt",
                )

                if "attention_mask" not in padded:
                    padded["attention_mask"] = (padded["input_ids"] != pad_id).long()

                inputs = {k: v.to(model.device) for k, v in padded.items()}
                prompt_length = inputs["input_ids"].shape[1]

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": min(24, max_tokens),
                    "pad_token_id": pad_id,
                    "use_cache": True,
                    "do_sample": True,
                    "temperature": temp,
                    "top_p": 0.95,
                }

                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id

                output_ids = model.generate(**inputs, **gen_kwargs)

                for local_idx, output in enumerate(output_ids):
                    global_idx = current_indices[local_idx]
                    processed = processed_batch[global_idx]
                    orig_item = batch[global_idx]

                    new_ids = strip_prompt_from_outputs(output, prompt_length)
                    trimmed_decoded = tokenizer.decode(
                        new_ids, skip_special_tokens=True
                    ).strip()

                    last_raw_outputs[global_idx] = trimmed_decoded
                    attempt_outputs[global_idx].append(trimmed_decoded)

                    options = processed.get("options", [])

                    if dataset_name == "aqua":
                        pred_answer = dataset_module.extract_answer(
                            trimmed_decoded,
                            options=options,
                        ) or ""
                    else:
                        pred_answer = dataset_module.extract_answer(trimmed_decoded) or ""

                    if (not pred_answer) or (pred_answer == "no_final_answer"):
                        continue

                    corrected_after_retry = exact_match(processed["answer"], pred_answer)

                    if is_retry:
                        qid = processed.get("id", global_idx)
                        logger.info(
                            f"[RETRY_SUCCESS] dataset={dataset_name} id={qid} "
                            f"attempt={attempt+1}/{max_attempts} "
                            f"pred={pred_answer} correct={corrected_after_retry}"
                        )

                    retry_output = (
                        "\n\n".join(attempt_outputs[global_idx])
                        if was_retried[global_idx] and len(attempt_outputs[global_idx]) > 1
                        else trimmed_decoded
                    )

                    batch_results[global_idx] = {
                        "question": processed["question"],
                        "initial_output": orig_item.get("full_output", ""),
                        "retry_output": retry_output,
                        "correct_answer": processed["answer"],
                        "initial_answer": orig_item.get("predicted_answer"),
                        "retry_answer": pred_answer,
                        "corrected_after_retry": corrected_after_retry,
                    }

                pending_indices = [
                    i for i in pending_indices
                    if batch_results[i] is None
                ]

            for idx, res in enumerate(batch_results):
                if res is None:
                    qid = processed_batch[idx].get("id", idx)
                    logger.warning(
                        f"[FAILED] dataset={dataset_name} id={qid} "
                        f"attempts={max_attempts} "
                        f"was_retried={was_retried[idx]} "
                        f"last_output_chars={len(last_raw_outputs[idx] or '')}"
                    )

                    processed = processed_batch[idx]
                    orig_item = batch[idx]
                    raw_out = last_raw_outputs[idx] or ""

                    batch_results[idx] = {
                        "question": processed["question"],
                        "initial_output": orig_item.get("full_output", ""),
                        "retry_output": raw_out,
                        "correct_answer": processed["answer"],
                        "initial_answer": orig_item.get("predicted_answer"),
                        "retry_answer": "no answer",
                        "corrected_after_retry": False,
                    }

            results.extend(batch_results)

    return results


def generate_hints(
    data: Iterable[Dict[str, Any]],
    model,
    tokenizer,
    dataset_name: str,
    num_attempts: int = 3,
    max_tokens: int = 512,
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

            was_retried: List[bool] = [False] * batch_size_actual
            retry_logged: set[int] = set()

            for attempt in range(num_attempts):
                if not pending_indices:
                    break

                is_retry = attempt > 0
                if is_retry:
                    for idx in pending_indices:
                        was_retried[idx] = True
                        if idx not in retry_logged:
                            qid = batch[idx].get("id", idx)  
                            prev = (last_decoded.get(idx, "") or "") 
                            logger.info( 
                                f"[RETRY] dataset={dataset_name} id={qid} attempt={attempt+1}/{num_attempts} "
                                f"reason=no_valid_hint_extracted prev_chars={len(prev)} "
                                f"action=regenerate_hint"
                            )
                            retry_logged.add(idx)

                current_input_ids: List[List[int]] = []
                for i in pending_indices:
                    ids = encode_chat(
                        tokenizer,
                        messages=[{"role": "user", "content": prompts_batch[i]}],
                        add_generation_prompt=True,
                    )
                    current_input_ids.append(list(ids))

                
                pad_id, eos_id = resolve_pad_eos(tokenizer)
                
                padded = tokenizer.pad(
                    {"input_ids": current_input_ids},
                    padding=True,
                    return_tensors="pt",
                )
                
                if "attention_mask" not in padded:
                    padded["attention_mask"] = (padded["input_ids"] != pad_id).long()  # USE pad_id
                
                inputs = {k: v.to(model.device) for k, v in padded.items()}
                prompt_length = inputs["input_ids"].shape[1]


                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": min(24, max_tokens),
                    "pad_token_id": pad_id,
                    "use_cache": True,
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95
                }
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id


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

                        if is_retry:
                            qid = item.get("id", global_idx)
                            logger.info(
                                f"[RETRY_SUCCESS] dataset={dataset_name} id={qid} attempt={attempt+1}/{num_attempts} "
                                f"hint_chars={len(hint_text)}"
                            )

                # Filter out those that already have a valid hint
                pending_indices = [i for i in pending_indices if batch_hints[i] is None]

            # Add items with last attempt's hint if validation failed
            for idx, res in enumerate(batch_hints):
                if res is None:
                    qid = batch[idx].get("id", idx)
                    logger.warning(
                        f"[FAILED] dataset={dataset_name} id={qid} attempts={num_attempts} "
                        f"was_retried={was_retried[idx]} last_output_chars={len(last_decoded.get(idx, '') or '')}"
                    )

                    item_with_hint = batch[idx].copy()
                    raw = last_decoded.get(idx, "") or ""
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
