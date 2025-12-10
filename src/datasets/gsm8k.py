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
    raw_answer = str(item["answer"])

    # Extract the final answer after "####"
    m = re.search(r"####\s*(.+?)\s*$", raw_answer)
    if m:
        final_answer = m.group(1).strip()
    else:
        # Fallback: last token of last non-empty line
        lines = [l for l in raw_answer.splitlines() if l.strip()]
        if not lines:
            final_answer = ""
        else:
            last_line = lines[-1].strip()
            final_answer = last_line.split()[-1]

    # Use the shared normalization
    final_answer = _normalize_number_token(final_answer)

    return {
        "id": item.get("id"),
        "question": item["question"],
        "answer": final_answer
    }



def extract_answer(text: str) -> str:
    """
    Generic numeric extractor for GSM/ASDiv-style prompts.

    Priority:
      1) Last <ans>...</ans> block (tolerant to whitespace + nesting)
      2) Anchored phrases like 'final answer is 42'
      3) Else: 'no_final_answer'
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()

    # 1) <ans>...</ans>, take LAST one, tolerate `< ans >` spacing + nested tags
    ans_blocks = re.findall(
        r"<\s*ans\s*>(.*?)</\s*ans\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if ans_blocks:
        candidate_block = ans_blocks[-1].strip()

        # Strip any nested <ans> tags inside the block
        candidate_clean = re.sub(
            r"</?\s*ans\s*>",
            " ",
            candidate_block,
            flags=re.IGNORECASE,
        ).strip()

        # Pull the last numeric-looking token from inside the block
        numbers = re.findall(
            r"[-+]?\$?\d+(?:[.,]\d+)?%?",
            candidate_clean,
        )
        if numbers:
            return _normalize_number_token(numbers[-1])

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