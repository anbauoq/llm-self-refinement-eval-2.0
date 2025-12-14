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
    1. <cot_start> ... <cot_end>
    2. <think> ... </think>
    If neither is present, return the full output stripped.
    """
    if not output:
        return ""
        
    text = output

    # 1) Try <cot_start> ... <cot_end>
    cot_matches = re.findall(
        r"<cot_start>(.*?)<cot_end>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if cot_matches:
        return cot_matches[-1].strip()

    # 2) Fall back to <think> ... </think>
    think_matches = re.findall(
        r"<think>(.*?)</think>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if think_matches:
        return think_matches[-1].strip()

    # 3) No tags – treat full output as reasoning
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
