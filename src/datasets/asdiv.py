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


def extract_answer(text: str) -> str:
    """
    Generic numeric extractor for GSM/ASDiv-style prompts.

    Priority:
      1) Last <ans>...</ans> block
      2) Anchored phrases like 'final answer is 42'
      3) Else: 'no_final_answer'
    """
    if not text or not text.strip():
        return "no_final_answer"

    raw = text
    t = raw.strip()

    # 1) <ans>...</ans>, take LAST one
    ans_blocks = re.findall(r"<ans>(.*?)</ans>", t, flags=re.IGNORECASE | re.DOTALL)
    if ans_blocks:
        candidate = ans_blocks[-1].strip()
        m = re.search(r"[-+]?\$?\d+(?:[.,]\d+)?%?", candidate)
        if m:
            return _normalize_number_token(m.group(0))
        return "no_final_answer"

    # 2) backup: 'final answer is 42', 'answer: $1,300%', etc.
    anchored = re.findall(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*([-+]?\$?\d+(?:[.,]\d+)?%?)",
        t,
        flags=re.IGNORECASE,
    )
    if anchored:
        return _normalize_number_token(anchored[-1])

    # 3) do NOT guess from random numbers in the whole text
    return "no_final_answer"