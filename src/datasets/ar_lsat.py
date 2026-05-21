import re
from typing import Dict, Any, Optional, List, Tuple

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single AR_LSAT item into internal format.

    - `question`: context + blank line + question + options (one per line)
    - `answer`: kept exactly as in the source item (no normalization)
    - `task_type`: 'mc'
    """
    choices_block = "\n".join(item["options"])

    full_question = (
        item["context"] + "\n\n" +
        item["question"] + "\n" +
        choices_block
    )

    return {
        "id": item.get("id"),
        "question": full_question,
        "answer": item["answer"],
    }

def extract_answer(text: str) -> str:
    """
    Extract final MCQ letter (A–E) from model output.

    Rule: return the answer candidate that appears LAST in the text,
    across:
      - <ans>...</ans> / <answer>...</answer>
      - \\boxed{...}
      - anchored phrases like "final answer is C"
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()
    candidates: List[Tuple[int, str]] = []  # (position, letter)

    def _pick_letter(blob: str) -> Optional[str]:
        if not blob:
            return None
        s = blob.strip()

        m = re.fullmatch(r"[A-E]", s, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()

        # pick LAST standalone letter token (safer than first)
        toks = re.findall(r"\b([A-E])\b", s, flags=re.IGNORECASE)
        if toks:
            return toks[-1].upper()

        # pick LAST bracketed-ish letter
        br = re.findall(r"[\(\[\{<\*\"']\s*([A-E])\s*[\)\]\}>\"\*']",
                        s, flags=re.IGNORECASE)
        if br:
            return br[-1].upper()

        return None

    # 1) <ans>...</ans> or <answer>...</answer> (collect ALL)
    for m in re.finditer(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = (m.group(2) or "").strip()

        # strip nested <ans>/<answer> tags inside
        clean = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()

        letter = _pick_letter(clean)
        if letter:
            candidates.append((m.start(), letter))

    # 2) \\boxed{...} (collect ALL)
    for m in re.finditer(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL):
        inside = (m.group(1) or "").strip()

        # unwrap common latex wrappers like \text{C}, \mathbf{C}, etc. (repeat to handle nesting)
        while True:
            new_inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()
            if new_inside == inside:
                break
            inside = new_inside

        letter = _pick_letter(inside)
        if letter:
            candidates.append((m.start(), letter))

    # 3) Anchored phrases (collect ALL)
    for m in re.finditer(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*[\*\(\[\{\"']*([A-E])",
        t,
        flags=re.IGNORECASE,
    ):
        candidates.append((m.start(1), m.group(1).upper()))

    if not candidates:
        return "no_final_answer"

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]
