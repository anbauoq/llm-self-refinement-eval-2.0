import re
from typing import Dict, Any, Optional, List, Tuple

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
    Extract final 0/1 answer from model output for SPORTS.

    Rule: return the binary candidate that appears LAST in the text,
    across:
      - <ans>...</ans> / <answer>...</answer>
      - \\boxed{...}
      - 'Answer: yes/no/0/1' patterns (optionally after <cot_end>)
    """
    if not output or not output.strip():
        return "no_final_answer"

    candidates: List[Tuple[int, str]] = []  # (position, "0"/"1")

    def _pick_binary(blob: str) -> Optional[str]:
        if not blob:
            return None
        s = blob.strip()

        # prefer last digit token if present
        digits = re.findall(r"\b([01])\b", s)
        if digits:
            return digits[-1]

        # else last yes/no
        yn = re.findall(r"\b(yes|no)\b", s, flags=re.IGNORECASE)
        if yn:
            return map_to_binary(yn[-1])

        return None

    # 1) <ans>/<answer> tags (collect ALL)
    for m in re.finditer(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = (m.group(2) or "").strip()
        clean = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()

        val = _pick_binary(clean)
        if val is not None:
            candidates.append((m.start(), val))

    # 2) \\boxed{...} (collect ALL)
    for m in re.finditer(r"\\boxed\s*\{([^}]*)\}", output, flags=re.IGNORECASE | re.DOTALL):
        inside = (m.group(1) or "").strip()

        # unwrap common latex wrappers like \text{yes}, \mathbf{1}, etc. (repeat for nesting)
        while True:
            new_inside = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", inside).strip()
            if new_inside == inside:
                break
            inside = new_inside

        val = _pick_binary(inside)
        if val is not None:
            candidates.append((m.start(), val))

    # 3) 'Answer: ...' patterns (collect ALL)

    # after <cot_end> if present
    parts = re.split(r"<\s*cot_end\s*>", output, flags=re.IGNORECASE)
    after_cot = parts[-1]
    for m in re.finditer(r"answer:\s*(yes|no|1|0)\b", after_cot, flags=re.IGNORECASE):
        candidates.append((m.start(1), map_to_binary(m.group(1))))

    # anywhere in output
    for m in re.finditer(r"answer:\s*(yes|no|1|0)\b", output, flags=re.IGNORECASE):
        candidates.append((m.start(1), map_to_binary(m.group(1))))

    if not candidates:
        return "no_final_answer"

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]