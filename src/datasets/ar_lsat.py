import re
from typing import Dict, Any, List

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
        "answer": item["answer"]
    }


def extract_answer(text: str) -> str:
    """
    Extract final MCQ letter (A–E) from model output for AQuA-style prompts.

    Priority:
      1) Last <ans>...</ans> block (primary, expected path), tolerant to:
         - `< ans >` vs `<ans>`
         - nested `<ans><ans>E</ans></ans>`
      2) Anchored phrases like 'final answer is C'
      3) Otherwise: 'no_final_answer'
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()

    # 1) Prefer explicit <ans>...</ans> tag, take the LAST one
    #    Be tolerant to whitespace in the tag name: < ans > ... </ ans >
    ans_blocks = re.findall(
        r"<\s*ans\s*>(.*?)</\s*ans\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if ans_blocks:
        candidate_block = ans_blocks[-1].strip()

        # Strip any nested <ans> tags inside the block
        candidate_block_clean = re.sub(
            r"</?\s*ans\s*>",
            " ",
            candidate_block,
            flags=re.IGNORECASE,
        ).strip()

        # First, require that the whole thing is just a single letter A–E
        m_exact = re.fullmatch(r"[A-E]", candidate_block_clean, flags=re.IGNORECASE)
        if m_exact:
            return m_exact.group(0).upper()

        # If it's not *just* a letter but contains one, grab that
        m_inside = re.search(r"\b([A-E])\b", candidate_block_clean, flags=re.IGNORECASE)
        if m_inside:
            return m_inside.group(1).upper()

    # 2) Strong anchored patterns (fallback if no usable <ans> block)
    anchored = re.findall(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer)"
        r"\s*(?:is|:|=)?\s*[\*\(\[\{\"']*([A-E])",
        t,
        flags=re.IGNORECASE,
    )
    if anchored:
        return anchored[-1].upper()

    # 3) No safe signal - treat as no final answer
    return "no_final_answer"