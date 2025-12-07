#!/usr/bin/env python3
# utils.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


# I/O

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



from pathlib import Path

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
    if filename is None:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r} for hint prompts. "
            f"Known: {sorted(hint_prompt_map.keys())}"
        )

    template = Path(filename).read_text(encoding="utf-8")
    return template.format(
        question=question.strip(),
        predicted_answer=str(predicted_answer).strip(),
        chain_of_thought=str(chain_of_thought).strip(),
        correct_answer=str(correct_answer).strip(),
    )



# Parsing / validation

_COT_RE = re.compile(r"<cot_start>(.*?)<cot_end>", flags=re.DOTALL)


def extract_cot(output: str) -> str:
    """
    Extract the last CoT block delimited by <cot_start> ... <cot_end>.
    Returns "" if none found.
    """
    matches = _COT_RE.findall(output or "")
    return matches[-1].strip() if matches else output


def contains_bad_phrases(hint: str, answer: str) -> bool:
    """
    True if the hint likely reveals the answer directly.
    """
    h = (hint or "").lower()
    a = (answer or "").strip().lower()
    blacklist = (
        "final answer",
        "the answer is",
        "answer:",
        a,
    )
    return any(token in h for token in blacklist if token)


def is_valid_hint(hint: str, answer: str) -> bool:
    """Valid iff it does NOT contain any blacklisted phrase or the answer itself."""
    return not contains_bad_phrases(hint, answer)


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

# --- Parsing ---
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
