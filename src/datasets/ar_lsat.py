import re
from typing import Dict, Any, List

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    choices = "\n".join(item["options"]) 
    full_question = f"{item['question']}\nChoices:\n{choices}"
    return {
        "id": item.get("id"),
        "question": f"{item['context']}\n\n{full_question}",
        "answer": item["answer"].strip().upper()
    }

def extract_answer(text: str) -> str:
    if not text:
        return "no_final_answer"

    t = text.strip()

    # Unwrap special "Here is the answer … <|im_start|> … <|im_end|>" block if present
    if "######Here is the answer" in t:
        after = t.split("######Here is the answer", 1)[1]
        m = re.search(r"<\|im_start\|>(.*?)<\|im_end\|>", after, flags=re.DOTALL | re.IGNORECASE)
        t = m.group(1) if m else after

    # Prefer content after an explicit CoT terminator if you use one
    if "<cot_end>" in t:
        t = t.split("<cot_end>")[-1]

    # Normalize whitespace for anchored scans
    t_norm = re.sub(r"\s+", " ", t).strip()

    # 1) Strong anchors (take LAST): Answer/Final/Correct/Choice/Option
    anchor_re = r"(?:final\s*)?(?:answer|ans\.?|correct\s*answer|choice|option)\s*(?:is|:|=)?\s*[\*\(\[\{\"']*([A-E])[\)\]\}\"']*"
    anchored = re.findall(anchor_re, t_norm, flags=re.IGNORECASE)

    if anchored:
        return anchored[-1].upper()

    # 2) Fallback: remove option-list lines so we don't pick letters from choices
    t_no_choices = re.sub(r"(?im)^\s*choices?\s*:\s*$", "", t)
    t_no_choices = re.sub(r"(?im)^\s*[A-E][\)\].:]\s.*$", "", t_no_choices)

    # Generic standalone A–E tokens; take LAST
    generic = re.findall(r"(?:^|\s)([A-E])(?=[\)\].,:;\s]|$)", t_no_choices, flags=re.IGNORECASE)
    if generic:
        return generic[-1].upper()

    return "no_final_answer"
