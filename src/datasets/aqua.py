import re
from typing import Optional, Tuple, List, Dict, Any

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "id": item.get("id"),
        "question": item['question'],
        "answer": item["answer"].strip().upper()
    }

def extract_answer(text: str) -> str:
    """
    Extract final MCQ letter (A–E) from model output.

    Rule: return the answer candidate that appears LAST in the text,
    across these formats:
      - <ans>...</ans> or <answer>...</answer>
      - \\boxed{...}
      - anchored phrases like "final answer is C"
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()
    candidates: List[Tuple[int, str]] = []  # (position_in_text, letter)

    def _pick_last_letter(blob: str) -> Optional[str]:
        if not blob:
            return None
        s = blob.strip()

        # 1) exact single letter
        m = re.fullmatch(r"[A-E]", s, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()

        # 2) last standalone letter token
        toks = re.findall(r"\b([A-E])\b", s, flags=re.IGNORECASE)
        if toks:
            return toks[-1].upper()

        # 3) last bracketed-ish letter like "(C)" "[C]" "{C}" "<C>" "*C*" '"C"'
        br = re.findall(r"[\(\[\{<\*\"']\s*([A-E])\s*[\)\]\}>\"\*']",
                        s, flags=re.IGNORECASE)
        if br:
            return br[-1].upper()

        return None

    # A) <ans>/<answer> blocks (record each, take last overall later)
    for m in re.finditer(r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
                         t, flags=re.IGNORECASE | re.DOTALL):
        block = (m.group(2) or "").strip()

        # strip nested <ans>/<answer> tags inside
        block = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()

        letter = _pick_last_letter(block)
        if letter:
            candidates.append((m.start(), letter))

    # B) \boxed{...}
    for m in re.finditer(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL):
        inside = (m.group(1) or "").strip()

        # unwrap common LaTeX wrappers like \text{C}, \mathbf{C}, etc. (repeat to handle nesting)
        while True:
            new_inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()
            if new_inside == inside:
                break
            inside = new_inside

        letter = _pick_last_letter(inside)
        if letter:
            candidates.append((m.start(), letter))

    # C) anchored phrases
    for m in re.finditer(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)\s*(?:is|:|=)?\s*[\*\(\[\{\"']*([A-E])",
        t,
        flags=re.IGNORECASE,
    ):
        candidates.append((m.start(1), m.group(1).upper()))

    if not candidates:
        return "no_final_answer"

    # return the candidate that appears last in the output
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]