#!/usr/bin/env python3
# utils.py
from __future__ import annotations

import json
import re
import torch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def load_data(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.
    Skips empty lines and surfaces the line number on JSON errors.
    """
    p = Path(path)
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {p}: {e}") from e
    return out


def save_data(data: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """
    Save an iterable of dict-like objects to a JSONL file.
    """
    p = Path(path)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_cot(output: str) -> str:
    """
    Extract the last reasoning block.

    Priority:
    1) <think> ... </think>  OR  <thinking> ... </thinking>  (whichever appears last)
    2) If only a closing tag exists (no matching opening), take text before the last closing
       - handles </think> and </thinking>
    3) <cot_start> ... <cot_end>
    If none are present, return full output stripped.
    """
    if not output:
        return ""

    text = output
    flags = re.DOTALL | re.IGNORECASE

    # 1) Full blocks: choose whichever (think/thinking) appears last in the text
    blocks = []
    for tag in ("think", "thinking"):
        for m in re.finditer(rf"<{tag}\b[^>]*>(.*?)</{tag}\b[^>]*>", text, flags=flags):
            blocks.append((m.start(), m.group(1).strip()))
    if blocks:
        blocks.sort(key=lambda x: x[0])
        return blocks[-1][1]

    # 2) closing tag without opening tag: </think>
    if not re.search(r"<think\b", text, flags=re.IGNORECASE):
        close_tags = list(re.finditer(r"</think\b[^>]*>", text, flags=re.IGNORECASE))
        if close_tags:
            end = close_tags[-1].start()
            return text[:end].strip()

    # 2b) closing tag without opening tag: </thinking>
    if not re.search(r"<thinking\b", text, flags=re.IGNORECASE):
        close_tags = list(re.finditer(r"</thinking\b[^>]*>", text, flags=re.IGNORECASE))
        if close_tags:
            end = close_tags[-1].start()
            return text[:end].strip()

    # 3) <cot_start> ... <cot_end>
    cot_matches = re.findall(r"<cot_start\b[^>]*>(.*?)<cot_end\b[^>]*>", text, flags=flags)
    if cot_matches:
        return cot_matches[-1].strip()

    # 4) No tags – treat full output as reasoning
    return text.strip()


# Evaluation helpers

def exact_match(true_answer: str, predicted_answer: str) -> bool:
    """
    Case-insensitive equality, robust to accidental newlines in the model output.
    If predicted_answer has multiple lines, match against any line.
    """
    ta = (true_answer or "").strip().lower()
    if not ta:
        return False

    for line in (predicted_answer or "").splitlines():
        if ta == line.strip().lower():
            return True
    return False

# Parsing
def parse_max_new_tokens(max_tokens: Any, default: int) -> int:
    """Accept int or int-like str; otherwise use default."""
    if isinstance(max_tokens, int):
        return max(1, max_tokens)
    if isinstance(max_tokens, str):
        try:
            return max(1, int(max_tokens))
        except ValueError:
            return default
    return default

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
