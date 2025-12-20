#!/usr/bin/env python3
# hints.py

from __future__ import annotations

import re

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
    a = answer

    generic_triggers = (
        "the correct answer",
        "this is the correct answer",
        "this is the right answer",
        "is the correct answer",
        "is the right answer",
        "final answer is",
        "<ans>",
        "boxed{"
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
            rf"option\s+\b{re.escape(a)}\b",
            rf"choice\s+\b{re.escape(a)}\b",
            rf"letter\s+\b{re.escape(a)}\b",
            rf"\(\s*{re.escape(a)}\s*\)",
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
    Replace direct mentions of the correct answer in the hint with '<MSK>'.

    - Masks common "answer is X" patterns by replacing X with <MSK>
    """
    if not hint or not answer:
        return hint

    h = hint
    a = answer.strip()
    msk = "<MSK>"

    # 1) Mask common "answer + value" patterns first
    # e.g., "the answer is 18" -> "the answer is <MSK>"
    patterns = [
        rf"(final answer\s*[:\-]?\s*){re.escape(a)}",
        rf"(the answer is\s*[:\-]?\s*){re.escape(a)}",
        rf"(correct answer is\s*[:\-]?\s*){re.escape(a)}",
        rf"(answer\s*[:\-]?\s*){re.escape(a)}",
    ]
    for pat in patterns:
        h = re.sub(pat, rf"\1{msk}", h, flags=re.IGNORECASE)

    # 2) Mask the answer token itself when it appears as a separate word/number
    token_pattern = r"\b" + re.escape(a) + r"\b"
    h = re.sub(token_pattern, msk, h, flags=re.IGNORECASE)

    return h

def extract_hint_text(output: str) -> str:

    # Proper paired tags
    matches = re.findall(
        r"<hint[s]?\b[^>]*>(.*?)</hint[s]?>",
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()

    # Handle malformed case: missing opening tag but has a closing tag
    m = re.search(r"</hint[s]?\s*>", output, flags=re.IGNORECASE)
    if m:
        return output[: m.start()].strip()

    # Fallback
    return output.strip()