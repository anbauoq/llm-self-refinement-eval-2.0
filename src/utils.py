#!/usr/bin/env python3
# utils.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


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


# Prompt formatting

def format_prompt(
    question: str,
    inject_hint: bool = False,
    hint: str = "",
    dataset_name: str = "gsm8k",
) -> str:
    """
    Load and format the dataset-specific answer prompt.
    Optionally prepend a hint.
    """

    prompt_file_map = {
        "asdiv": "arithmetic_prompt.txt",
        "gsm8k": "arithmetic_prompt.txt",
    }
    prompt_filename = prompt_file_map.get(dataset_name, f"{dataset_name}_prompt.txt")

    prompt_path = Path("prompts") / prompt_filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")
    body = template.format(question=question.strip())
    return f"{hint.strip()}\n\n{body}" if inject_hint and hint else body


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



def contains_bad_phrases(hint: str, answer: str, dataset_name: str) -> bool:
    """
    True if the hint very likely reveals the answer directly.

    Dataset-aware behavior:
    - asdiv, gsm8k (arithmetic):
        numeric/word answers – treat any standalone occurrence as a leak
        (plus explicit "answer is X" patterns).
    - aqua, ar_lsat (MC, letter answers):
        only treat as leak when clearly tied to 'answer', 'option', etc.
        (don't kill every random 'a'/'b'/...).
    - sports (True/False as 0/1):
        only treat as leak when 0/1 appears in explicit 'answer/label is' phrases
        (scores like 1–0 should NOT be flagged).
    """
    if not hint:
        return False

    h = (hint or "").lower()
    a = (answer or "").strip().lower()

    # Generic "this is the answer" phrases (bad everywhere, even without seeing `a`)
    generic_triggers = (
        "the correct answer",
        "this is the correct answer",
        "this is the right answer",
        "is the correct answer",
        "is the right answer",
    )
    if any(t in h for t in generic_triggers):
        return True

    if not a:
        return False

    if dataset_name in {"asdiv", "gsm8k"}:
        is_numeric_like = a.replace(".", "", 1).isdigit() or "/" in a
        # For arithmetic, be aggressive on numeric / longer answers
        if len(a) > 2 or is_numeric_like:
            # Any standalone occurrence is suspicious
            pattern = r"\b" + re.escape(a) + r"\b"
            if re.search(pattern, h):
                return True

            # Also catch explicit "answer is X" patterns
            patterns = [
                rf"final answer\s*[:\-]?\s*{re.escape(a)}",
                rf"the answer is\s*[:\-]?\s*{re.escape(a)}",
                rf"correct answer\s*[:\-]?\s*{re.escape(a)}",
            ]
            for pat in patterns:
                if re.search(pat, h):
                    return True

        return False

    if dataset_name in {"aqua", "ar_lsat"}:
        # We assume `a` is the option letter (A/B/C/D/...)
        # Only flag when tied explicitly to "answer/option/choice/letter".
        short_patterns = [
            rf"final answer[^.\n]*\b{re.escape(a)}\b",
            rf"the answer is[^.\n]*\b{re.escape(a)}\b",
            rf"correct answer[^.\n]*\b{re.escape(a)}\b",
            rf"answer\s*(is|=)\s*\b{re.escape(a)}\b",
            rf"option\s+\b{re.escape(a)}\b",
            rf"choice\s+\b{re.escape(a)}\b",
            rf"letter\s+\b{re.escape(a)}\b",
            rf"\(\s*{re.escape(a)}\s*\)",  # e.g. "(A)"
        ]
        for pat in short_patterns:
            if re.search(pat, h):
                return True
        return False

    if dataset_name == "sports":
        # Do NOT treat any standalone '0'/'1' as leak – scores, times, etc.
        # Only flag explicit "answer/label is 0/1" style patterns.
        patterns = [
            rf"final answer\s*[:\-]?\s*{re.escape(a)}",
            rf"the answer is\s*[:\-]?\s*{re.escape(a)}",
            rf"correct answer\s*[:\-]?\s*{re.escape(a)}",
            rf"answer\s*(is|=)\s*{re.escape(a)}",
        ]
        for pat in patterns:
            if re.search(pat, h):
                return True
        return False

    return False



def is_valid_hint(hint: str, correct_answer: str, dataset_name: str) -> bool:
    return not contains_bad_phrases(hint, correct_answer, dataset_name)

    
def strip_answer_from_hint(hint: str, answer: str) -> str:
    """
    Remove direct mentions of the correct answer from the hint.

    - For numeric / multi-char answers: remove them as standalone tokens.
    - For single-letter answers (A/B/C/D/True/False): same idea, using word boundaries.
    """
    if not hint or not answer:
        return hint

    h = hint
    a = answer.strip()

    # Remove common "answer + value" patterns first
    # e.g., "the answer is 18", "final answer: C"
    patterns = [
        rf"(final answer\s*[:\-]?\s*){re.escape(a)}",
        rf"(the answer is\s*[:\-]?\s*){re.escape(a)}",
        rf"(correct answer is\s*[:\-]?\s*){re.escape(a)}",
    ]
    for pat in patterns:
        h = re.sub(pat, r"\1", h, flags=re.IGNORECASE)

    # Then remove the answer token itself when it appears as a separate word/number
    token_pattern = r"\b" + re.escape(a) + r"\b"
    h = re.sub(token_pattern, "", h, flags=re.IGNORECASE)

    # Collapse extra whitespace
    h = " ".join(h.split())
    return h

def extract_hint_text(output: str) -> str:
    """
    Extract inner text from <hint>...</hint> or <hints>...</hints> blocks.

    - If such a block exists, return the inner text of the *last* one.
    - Accepts both <hint> and <hints> (case-insensitive).
    - Self-closing tags like <hints/> cannot contain inner text, so they are
      effectively ignored; if only those are present, we fall back to the full
      output stripped.
    - If no such block exists, return the full decoded output stripped.
    """
    if not output:
        return ""

    # Match either <hint>...</hint> or <hints>...</hints>, any casing, multiline.
    # `hint[s]?` allows `hint` or `hints` on both open and close tags.
    matches = re.findall(
        r"<hint[s]?\b[^>]*>(.*?)</hint[s]?>",
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not matches:
        # No paired <hint>/<hints> tags – fall back to using the whole output
        # as the hint (even if there's a self-closing <hints/> somewhere).
        return output.strip()

    # Take the last block in case the model produced multiple
    inner = matches[-1].strip()
    return inner


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
def _parse_max_new_tokens(max_tokens: Any, default: int) -> int:
    """Accept int or int-like str; otherwise use default."""
    if isinstance(max_tokens, int):
        return max(1, max_tokens)
    if isinstance(max_tokens, str):
        try:
            return max(1, int(max_tokens))
        except ValueError:
            return default
    return default
