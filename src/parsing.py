from __future__ import annotations

import re

def extract_cot(output: str) -> str:
    """
    Extract the last reasoning block.

    Priority:
    1) <think> ... </think>  OR  <thinking> ... </thinking>  (whichever appears last)
    2) If only a closing tag exists (no matching opening), take text before the last closing
       - handles </think> and </thinking>
    3) <cot_start> ... <cot_end>
    If none are present, return full output stripped.
    """
    if not output:
        return ""

    text = output
    flags = re.DOTALL | re.IGNORECASE

    # 1) Full blocks: choose whichever (think/thinking) appears last in the text
    blocks = []
    for tag in ("think", "thinking"):
        for m in re.finditer(rf"<{tag}\b[^>]*>(.*?)</{tag}\b[^>]*>", text, flags=flags):
            blocks.append((m.start(), m.group(1).strip()))
    if blocks:
        blocks.sort(key=lambda x: x[0])
        return blocks[-1][1]

    # 2) closing tag without opening tag: </think>
    if not re.search(r"<think\b", text, flags=re.IGNORECASE):
        close_tags = list(re.finditer(r"</think\b[^>]*>", text, flags=re.IGNORECASE))
        if close_tags:
            end = close_tags[-1].start()
            return text[:end].strip()

    # 2b) closing tag without opening tag: </thinking>
    if not re.search(r"<thinking\b", text, flags=re.IGNORECASE):
        close_tags = list(re.finditer(r"</thinking\b[^>]*>", text, flags=re.IGNORECASE))
        if close_tags:
            end = close_tags[-1].start()
            return text[:end].strip()

    # 3) <cot_start> ... <cot_end>
    cot_matches = re.findall(r"<cot_start\b[^>]*>(.*?)<cot_end\b[^>]*>", text, flags=flags)
    if cot_matches:
        return cot_matches[-1].strip()

    # 4) No tags – treat full output as reasoning
    return text.strip()

def exact_match(true_answer: str, predicted_answer: str) -> bool:
    """
    Case-insensitive equality, robust to accidental newlines in the model output.
    If predicted_answer has multiple lines, match against any line.
    """
    ta = (true_answer or "").strip().lower()
    if not ta:
        return False

    for line in (predicted_answer or "").splitlines():
        if ta == line.strip().lower():
            return True
    return False