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

        clean = re.sub(
            r"</?\s*(?:ans|answer)\s*>",
            " ",
            block,
            flags=re.IGNORECASE,
        ).strip()

        num = _extract_last_number(clean)
        return num if num is not None else "no_final_answer"

    # 2) \\boxed{...} (take LAST)
    boxed = re.findall(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL)
    if boxed:
        inside = boxed[-1].strip()
        inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()

        num = _extract_last_number(inside)
        if num is not None:
            return num

    # 3) Anchored phrases
    anchored = re.findall(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*([-+]?\$?\d+(?:[.,]\d+)?%?)",
        t,
        flags=re.IGNORECASE,
    )
    if anchored:
        return _normalize_number_token(anchored[-1])

    return "no_final_answer"
