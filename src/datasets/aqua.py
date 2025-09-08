import re
from typing import Dict, Any

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # Join choices into properly formatted multiple-choice format
    choices = "\n".join(item["options"])
    full_question = f"{item['question'].strip()}\nChoices:\n{choices}"
    
    return {
        "id": item.get("id"),
        "question": full_question,
        "answer": item["answer"].strip().upper()
    }

def extract_answer(text: str) -> str:
    """
    Return the final MCQ letter (A–E) from a model's output.
    Priorities:
      1) Unwrap special "######Here is the answer ... <|im_start|>...<|im_end|>" blocks
      2) Strong anchors like 'answer:', 'final answer is', 'correct answer ='
      3) Fallback: last standalone A–E token (after removing choice lines)
    Fails closed with "".
    """
    if not text:
        return "no_final_answer"

    raw = text

    # 1) Unwrap special block if present
    t = raw.strip()
    if "######Here is the answer" in t:
        after = t.split("######Here is the answer", 1)[1]
        m = re.search(r"<\|im_start\|>(.*?)<\|im_end\|>", after, flags=re.DOTALL | re.IGNORECASE)
        t = m.group(1) if m else after

    # normalize whitespace for anchored scans
    t = re.sub(r"\s+", " ", t).strip()

    # 2) Strong, deterministic anchors (prefer the LAST one)
    anchored = re.findall(
        r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer|choice|option)\s*(?:is|:|=)?\s*[\*\(\[\{\"']*([A-E])[\)\]\}\"']*",
        t,
        flags=re.IGNORECASE,
    )
    if anchored:
        return anchored[-1].upper()

    # 3) Fallback: generic tokens — but first remove option-list lines to avoid false positives
    #    Strip lines like "A) foo", "B. bar", "C: baz" (case-insensitive), optionally after a "Choices:" header
    t2 = raw
    t2 = re.sub(r"(?:^|\n)\s*choices?\s*:\s*", "\n", t2, flags=re.IGNORECASE)
    t2 = re.sub(r"(?:^|\n)\s*[A-E][\)\].:][^\n]*", "\n", t2, flags=re.IGNORECASE)

    generic = re.findall(r"(?:^|\s)([A-E])(?=[\)\].,:;\s]|$)", t2, flags=re.IGNORECASE)
    if generic:
        return generic[-1].upper()

    return "no_final_answer"

