import re
from typing import Dict, Any, Optional, List, Tuple

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

    Rule: return the numeric candidate that appears LAST in the text,
    across:
      - <ans>...</ans> / <answer>...</answer>
      - \\boxed{...}
      - anchored phrases like "final answer is 42"
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()
    candidates: List[Tuple[int, str]] = []  # (position, normalized_number)

    def _extract_last_number(blob: str) -> Optional[str]:
        if not blob:
            return None
        numbers = re.findall(r"[-+]?\$?\d+(?:[.,]\d+)?%?", blob)
        if not numbers:
            return None
        return _normalize_number_token(numbers[-1])

    # 1) <ans>...</ans> or <answer>...</answer> (collect ALL)
    for m in re.finditer(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = (m.group(2) or "").strip()
        clean = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()

        num = _extract_last_number(clean)
        if num is not None:
            candidates.append((m.start(), num))

    # 2) \\boxed{...} (collect ALL)
    for m in re.finditer(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL):
        inside = (m.group(1) or "").strip()

        # unwrap common latex wrappers like \text{...}, \mathbf{...}, etc. (repeat to handle nesting)
        while True:
            new_inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()
            if new_inside == inside:
                break
            inside = new_inside

        num = _extract_last_number(inside)
        if num is not None:
            candidates.append((m.start(), num))

    # 3) Anchored phrases (collect ALL)
    for m in re.finditer(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*([-+]?\$?\d+(?:[.,]\d+)?%?)",
        t,
        flags=re.IGNORECASE,
    ):
        candidates.append((m.start(1), _normalize_number_token(m.group(1))))

    if not candidates:
        return "no_final_answer"

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]