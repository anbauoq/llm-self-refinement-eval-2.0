#!/usr/bin/env python3
# inference.py
from __future__ import annotations
import torch
import logging
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional
from prompts import format_initial_prompt, format_post_hint_prompt, format_hint_prompt, answers_reformatting
from hints import extract_hint_text, is_valid_hint, strip_answer_from_hint
from utils import ( 
    extract_cot,
    exact_match,
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
    data_list = list(data) # making a list of dictionaries, our dataset, each dict is one question with its corresponding features
    results: List[Dict[str, Any]] = []
    dataset_name = dataset_module.__name__.split(".")[-1]

    # Process in batches
    batches = batch_data(data_list, batch_size) # dividing our data into batches, list of lists, each of which contains batch_size dictionaries

    with torch.inference_mode(): 
        
        followup_user_msg = (
            "Explicitly write your reasoning on how to solve this problem inside <think> </think> and state the final answer isnide <ans>...</ans>. You MUST end your response with the final answer!"
        )

        if model_name in (
            "Qwen/Qwen2.5-Math-1.5B",
            "Qwen/Qwen2.5-Math-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        ):
            followup_user_msg = answers_reformatting(followup_user_msg)

        for batch in tqdm(batches, desc=f"Solving questions (batch_size={batch_size})"): # outer loop, considering one batch at a time
            
            # Prepare batch: process items + base prompts
            processed_batch: List[Dict[str, Any]] = [] # initializing a list that is gonna contain each question processed - dictionaries with features we need
            prompts_batch: List[str] = [] # initializing a list where generated prompts are going to be stored

            for item in batch: # considering one question (dictionary) at a time
                
                # Process item
                processed = dataset_module.process_item(item)

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

                processed_batch.append(processed) # must have a list with all questions from the given batch
                prompts_batch.append(base_prompt) # here corresponding prompts must be added


            


            # Pre-encode base prompts once per item.
            base_input_ids: List[List[int]] = []
            for p in prompts_batch:
                ids = encode_chat(
                    tokenizer,
                    messages=[{"role": "user", "content": p}],
                    add_generation_prompt=True,
                )
                base_input_ids.append(list(ids))

        

            batch_size_actual = len(batch) # just for the case when last batch is smaller
            
            
            batch_results: List[Optional[Dict[str, Any]]] = [None] * batch_size_actual # keeping status weather pred_answer succesfully extracted or no, initially oll are not so None

            last_raw_outputs: List[Optional[str]] = [None] * batch_size_actual # storing all outputs so in case of failure the full output is available
            
            pending_indices = list(range(batch_size_actual)) # initially all indices, all questions are pending (unanswered)

            last_attempt_is_retry: List[bool] = [False] * batch_size_actual
            retry_logged: set[int] = set()  # global_idx values we've already logged as retried


            for attempt in range(max_attempts):

                # where im going to come at the lasttttttt
                if not pending_indices:
                    break  # all items in this batch have been resolved

    
                is_retry = attempt > 0
                if is_retry:
                    for idx in pending_indices:
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
                        # keep it from exploding prompt length
                        if len(prev) > 4000:
                            prev = prev[-4000:]

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


                padded = tokenizer.pad(
                    {"input_ids": current_input_ids},
                    padding=True,
                    return_tensors="pt",
                )


                # Safety fallback: if some tokenizer still didn't return it, create it from padding
                if "attention_mask" not in padded:
                    pad_id = tokenizer.pad_token_id
                    padded["attention_mask"] = (padded["input_ids"] != pad_id).long()

                inputs = {k: v.to(model.device) for k, v in padded.items()}

                # all rows have the same sequence length after padding
                prompt_length = inputs["input_ids"].shape[1]

                pad_id, eos_id = resolve_pad_eos(tokenizer)


                # set model-specific temperature
                temp = 0.6
                if model_name in (
                    "microsoft/Phi-4-mini-instruct",
                    "microsoft/Phi-4-mini-reasoning",
                ):
                    temp = 0.8

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    #"min_new_tokens": min(64, max_tokens),
                    "pad_token_id": pad_id,
                    "use_cache": True,
                    "do_sample": True,
                    "temperature": temp,
                    "top_p": 0.95,
                }

                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id


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
                    options = processed.get("options", [])


                    if dataset_name == "aqua":
                        pred_answer = dataset_module.extract_answer(trimmed_decoded, options=options) or ""
                    else:
                        pred_answer = dataset_module.extract_answer(trimmed_decoded) or ""

                                    
                    if (not pred_answer) or (pred_answer == "no_final_answer"):
                        continue
                
                    is_correct = exact_match(processed["answer"], pred_answer)

                    if is_retry:
                        last_attempt_is_retry[global_idx] = True

                    if is_retry:
                        qid = processed.get("id", global_idx)
                        logger.info(
                            f"[RETRY_SUCCESS] dataset={dataset_name} id={qid} attempt={attempt+1}/{max_attempts} "
                            f"pred={pred_answer} correct={is_correct}"
                        )


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
                    qid = processed_batch[idx].get("id", idx)
                    logger.warning(
                        f"[FAILED] dataset={dataset_name} id={qid} attempts={max_attempts} "
                        f"from_retry={last_attempt_is_retry[idx]} last_output_chars={len(last_raw_outputs[idx] or '')}"
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
    max_tokens: int = 2048,
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

                current_input_ids: List[List[int]] = []
                for i in pending_indices:
                    ids = encode_chat(
                        tokenizer,
                        messages=[{"role": "user", "content": prompts_batch[i]}],
                        add_generation_prompt=True,
                    )
                    current_input_ids.append(list(ids))

                
                padded = tokenizer.pad(
                    {"input_ids": current_input_ids},
                    padding=True,
                    return_tensors="pt",
                )


                if "attention_mask" not in padded:
                    pad_id = tokenizer.pad_token_id
                    padded["attention_mask"] = (padded["input_ids"] != pad_id).long()

                inputs = {k: v.to(model.device) for k, v in padded.items()}
                prompt_length = inputs["input_ids"].shape[1]


                pad_id, eos_id = resolve_pad_eos(tokenizer)

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    #"min_new_tokens": min(64, max_tokens),
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
