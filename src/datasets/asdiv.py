import re
from typing import Dict, Any

def _normalize_number_token(s: str) -> str:
    """
    Normalize numeric string for comparison:
      - strip spaces
      - remove $, %
      - remove commas
      - treat patterns like '1.300' as 1300 (dot as thousands sep),
        but leave genuine decimals like '1.3' alone.
    """
    s = s.strip().replace(" ", "")
    s = s.replace("$", "").replace("%", "").replace(",", "")

    # thousands pattern: digits '.' exactly 3 digits (e.g., 1.300, 12.000)
    m = re.fullmatch(r"(-?\d+)\.(\d{3})", s)
    if m:
        return m.group(1) + m.group(2)

    return s


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    raw_answer = str(item["answer"]).strip()
    norm_answer = _normalize_number_token(raw_answer)

    return {
        "id": item.get("id"),
        "question": item["question"],
        "answer": norm_answer
    }


import re

def extract_answer(text: str) -> str:
    """
    Generic numeric extractor for GSM/ASDiv-style prompts.

    Priority:
      1) Last <ans>...</ans> OR <answer>...</answer> block (tolerant to whitespace + nesting)
      2) Last \\boxed{...} (e.g., \\boxed{42}, \\boxed{\\text{42}}, \\boxed{$1,300})
      3) Anchored phrases like 'final answer is 42'
      4) Else: 'no_final_answer'
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()

    def _extract_last_number(blob: str) -> str | None:
        if not blob:
            return None
        # keep this numeric regex consistent with your original
        numbers = re.findall(r"[-+]?\$?\d+(?:[.,]\d+)?%?", blob)
        if not numbers:
            return None
        return _normalize_number_token(numbers[-1])

    # 1) <ans>...</ans> or <answer>...</answer> (take LAST)
    tag_blocks = re.findall(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_blocks:
        _, block = tag_blocks[-1]
        block = block.strip()

        # strip nested tags inside the block
        clean = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()

        num = _extract_last_number(clean)
        return num if num is not None else "no_final_answer"

    # 2) \\boxed{...} (take LAST)
    boxed = re.findall(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL)
    if boxed:
        inside = boxed[-1].strip()
        # remove common latex wrappers like \text{...}, \mathbf{...}, etc.
        inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()

        num = _extract_last_number(inside)
        if num is not None:
            return num

    # 3) backup: 'final answer is 42', 'answer: $1,300%', etc.
    anchored = re.findall(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*([-+]?\$?\d+(?:[.,]\d+)?%?)",
        t,
        flags=re.IGNORECASE,
    )
    if anchored:
        return _normalize_number_token(anchored[-1])

    return "no_final_answer"