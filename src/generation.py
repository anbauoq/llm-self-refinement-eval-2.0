#!/usr/bin/env python3
# utils.py
from __future__ import annotations

import torch
from typing import Dict, List, Optional


def resolve_pad_eos(tokenizer) -> tuple[Optional[int], Optional[int]]:
    """Pick safe pad/eos once and reuse."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    return pad_id, eos_id


def batch_data(data: List, batch_size: int) -> List[List]:
    """Split data into batches."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def strip_prompt_from_outputs(
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

def encode_chat(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> List[int]:
    """
    Convert chat messages -> input_ids.
    Uses tokenizer.apply_chat_template if available; otherwise falls back to
    a simple plaintext conversation format.
    """
    can_chat = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    )

    if can_chat:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback: plain text "User:/Assistant:" chat.
    text_parts = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        text_parts.append(f"{role}: {content}")
    if add_generation_prompt:
        text_parts.append("assistant:")
    text = "\n".join(text_parts)

    return tokenizer(text, add_special_tokens=True)["input_ids"]
