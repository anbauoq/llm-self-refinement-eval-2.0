import re
from typing import Dict, Any

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item["question"],
        "answer": str(item["answer"]).strip()
    }

def map_to_binary(answer: str) -> str:
    answer = answer.strip().lower()
    if answer in ["yes", "1"]:
        return "1"
    elif answer in ["no", "0"]:
        return "0"
    return "no_final_answer"

import re

def extract_answer(output: str) -> str:
    """
    Extracts the final 0/1 answer from the model output for the SPORTS prompt.

    Priority:
      1) Last <ans>...</ans> OR <answer>...</answer> block (whitespace-tolerant), tolerant to:
         - `< ans >` vs `<ans>`, `< answer >` vs `<answer>`
         - nested tags like <ans><ans>1</ans></ans>
         - 'yes'/'no' inside the tag
      2) Last \\boxed{...} (e.g., \\boxed{1}, \\boxed{yes})
      3) Fallback: 'Answer: 0/1/yes/no' patterns (optionally after <cot_end>)
    """
    if not output or not output.strip():
        return "no_final_answer"

    def _pick_binary(blob: str) -> str | None:
        if not blob:
            return None
        s = blob.strip()

        m_digit = re.search(r"\b([01])\b", s)
        if m_digit:
            return m_digit.group(1)

        m_yn = re.search(r"\b(yes|no)\b", s, flags=re.IGNORECASE)
        if m_yn:
            return map_to_binary(m_yn.group(1))

        return None

    # 1) PRIMARY: <ans>/<answer> tags (take LAST)
    tag_blocks = re.findall(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_blocks:
        _, block = tag_blocks[-1]
        block = block.strip()

        # Strip nested <ans>/<answer> tags inside
        clean = re.sub(
            r"</?\s*(?:ans|answer)\s*>",
            " ",
            block,
            flags=re.IGNORECASE,
        ).strip()

        val = _pick_binary(clean)
        if val is not None:
            return val

    # 2) \\boxed{...} (take LAST)
    boxed = re.findall(r"\\boxed\s*\{([^}]*)\}", output, flags=re.IGNORECASE | re.DOTALL)
    if boxed:
        inside = boxed[-1].strip()
        # remove common latex wrappers like \text{yes}, \mathbf{1}, etc.
        inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()

        val = _pick_binary(inside)
        if val is not None:
            return val

    # 3) FALLBACK: legacy 'Answer: x' patterns

    # Split off anything after <cot_end> (also tolerate `< cot_end >`)
    parts = re.split(r"<\s*cot_end\s*>", output, flags=re.IGNORECASE)
    after_cot = parts[-1].lower()

    # normalize whitespace
    after_cot = re.sub(r"\s+", " ", after_cot)
    output_lower = re.sub(r"\s+", " ", output.lower())

    match = re.search(r"answer:\s*(yes|no|1|0)\b", after_cot)
    if match:
        return map_to_binary(match.group(1))

    match_fallback = re.search(r"answer:\s*(yes|no|1|0)\b", output_lower)
    if match_fallback:
        return map_to_binary(match_fallback.group(1))

    return "no_final_answer"