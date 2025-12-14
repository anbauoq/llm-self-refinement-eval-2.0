import re
from typing import Dict, Any

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

    Priority:
      1) Last <ans>...</ans> OR <answer>...</answer> block (tolerant to whitespace + nesting)
      2) Last \\boxed{...} (e.g., \\boxed{C}, \\boxed{\\text{C}}, \\boxed{(C)})
      3) Anchored phrases like 'final answer is C'
      4) Otherwise: 'no_final_answer'
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()

    def _pick_letter(blob: str):
        if not blob:
            return None
        s = blob.strip()

        m = re.fullmatch(r"[A-E]", s, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()

        m = re.search(r"\b([A-E])\b", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        m = re.search(
            r"[\(\[\{<\*\"']\s*([A-E])\s*[\)\]\}>\"\*']",
            s,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

        return None

    # 1) <ans>...</ans> or <answer>...</answer> (take LAST)
    tag_blocks = re.findall(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_blocks:
        _, block = tag_blocks[-1]
        block = block.strip()

        # strip nested <ans>/<answer> tags inside
        clean = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()

        letter = _pick_letter(clean)
        if letter:
            return letter

    # 2) \\boxed{...} (take LAST)
    boxed = re.findall(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL)
    if boxed:
        inside = boxed[-1].strip()
        # remove common latex wrappers like \text{C}, \mathbf{C}, etc.
        inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()

        letter = _pick_letter(inside)
        if letter:
            return letter

    # 3) Anchored phrases
    anchored = re.findall(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*[\*\(\[\{\"']*([A-E])",
        t,
        flags=re.IGNORECASE,
    )
    if anchored:
        return anchored[-1].upper()

    return "no_final_answer"
