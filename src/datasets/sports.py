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
    # Normalize formatting characters
    output = output.replace('\xa0', ' ')  # non-breaking space
    parts = output.split("<cot_end>")
    after_cot = parts[-1].lower()

    # Strip extra stuff
    after_cot = re.sub(r"\s+", " ", after_cot)  # collapse all whitespace
    output = re.sub(r"\s+", " ", output.lower())

    match = re.search(r"answer:\s*(yes|no|1|0)\b", after_cot)
    if match:
        return map_to_binary(match.group(1))

    match_fallback = re.search(r"answer:\s*(yes|no|1|0)\b", output)
    if match_fallback:
        return map_to_binary(match_fallback.group(1))

    return "no_final_answer"