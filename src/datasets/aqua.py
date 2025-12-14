import re
from typing import Dict, Any

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "id": item.get("id"),
        "question": item['question'],
        "answer": item["answer"].strip().upper()
    }

def extract_answer(text: str) -> str:
    """
    Extract final MCQ letter (A–E) from model output for AQuA-style prompts.

    Priority:
      1) Last <ans>...</ans> OR <answer>...</answer> block (primary), tolerant to:
         - whitespace in tags: < ans >, < answer >
         - nested tags: <ans><ans>E</ans></ans>
      2) Last \\boxed{...} (e.g., \\boxed{C}, \\boxed{\\text{C}}, \\boxed{(C)})
      3) Anchored phrases like 'final answer is C'
      4) Otherwise: 'no_final_answer'
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()

    # Helper: from a blob, try to pick a single A–E
    def _pick_letter(blob: str) -> str | None:
        if not blob:
            return None
        s = blob.strip()

        # exact single letter
        m = re.fullmatch(r"[A-E]", s, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()

        # contains a standalone letter token
        m = re.search(r"\b([A-E])\b", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # contains something like "(C)" or "[C]" or "*C*"
        m = re.search(r"[\(\[\{<\*\"']\s*([A-E])\s*[\)\]\}>\"\*']", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        return None

    # 1) Prefer explicit tags: <ans> or <answer> (take the LAST block found)
    tag_blocks = re.findall(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_blocks:
        _tag, candidate_block = tag_blocks[-1]
        candidate_block = candidate_block.strip()

        # strip any nested <ans>/<answer> tags inside
        candidate_clean = re.sub(
            r"</?\s*(?:ans|answer)\s*>",
            " ",
            candidate_block,
            flags=re.IGNORECASE,
        ).strip()

        letter = _pick_letter(candidate_clean)
        if letter:
            return letter

    # 2) LaTeX \\boxed{...} (take the LAST one)
    boxed = re.findall(
        r"\\boxed\s*\{([^}]*)\}",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if boxed:
        inside = boxed[-1].strip()

        # remove common LaTeX wrappers like \text{C}, \mathbf{C}, etc.
        inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()
        letter = _pick_letter(inside)
        if letter:
            return letter

    # 3) Strong anchored patterns (fallback)
    anchored = re.findall(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*[\*\(\[\{\"']*([A-E])",
        t,
        flags=re.IGNORECASE,
    )
    if anchored:
        return anchored[-1].upper()

    return "no_final_answer"
