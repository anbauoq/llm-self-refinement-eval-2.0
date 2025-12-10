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

def extract_answer(output: str) -> str:
    """
    Extracts the final 0/1 answer from the model output for the SPORTS prompt.

    Priority:
      1) Last <ans>...</ans> block (expected path), tolerant to:
         - `< ans >` vs `<ans>`
         - nested `<ans><ans>1</ans></ans>`
         - 'yes'/'no' inside <ans>
      2) Fallback: 'Answer: 0/1/yes/no' patterns
    """
    if not output or not output.strip():
        return "no_final_answer"

    # --- 1) PRIMARY: <ans>...</ans> tags (whitespace-tolerant) ---
    ans_blocks = re.findall(
        r"<\s*ans\s*>(.*?)</\s*ans\s*>",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if ans_blocks:
        # Take the LAST <ans>...</ans>
        candidate_block = ans_blocks[-1].strip()

        # Strip any nested <ans> tags inside the block
        candidate_clean = re.sub(
            r"</?\s*ans\s*>",
            " ",
            candidate_block,
            flags=re.IGNORECASE,
        ).strip()

        # First try to find an explicit 0/1
        m_digit = re.search(r"\b([01])\b", candidate_clean)
        if m_digit:
            return m_digit.group(1)

        # Then try yes/no inside the block
        m_yn = re.search(r"\b(yes|no)\b", candidate_clean, flags=re.IGNORECASE)
        if m_yn:
            return map_to_binary(m_yn.group(1))

    # --- 2) FALLBACK: legacy 'Answer: x' patterns, in case model ignores tags ---

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