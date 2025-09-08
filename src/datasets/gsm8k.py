import re
from typing import Dict, Any

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item["question"],
        "answer": str(item["answer"]).strip()
    }

_NUM_RE = r"[-+]?\s*\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%?"

def _canon_number_variants(s: str):
    """
    From a raw numeric token, produce ordered unique variants as strings:
      - canonical (no commas/$/%/spaces)
      - integer cast if decimal (e.g., '12.0' -> '12')
      - 'thousands-dot' variant (remove dot entirely) if applicable ('1.234' -> '1234')
    """
    v = s.strip()
    # strip wrappers/symbols
    v = v.replace(" ", "")
    v = v.replace("$", "").replace("%", "")
    v = v.replace(",", "")

    out = []
    def add(x):
        if x not in out:
            out.append(x)

    add(v)

    # if decimal, add int(float(...)) (handles '12.0' -> '12')
    if "." in v:
        try:
            add(str(int(float(v))))
        except ValueError:
            pass
        # treat dot as thousands separator (e.g., '1.234' -> '1234')
        dotless = v.replace(".", "")
        if dotless and dotless != v:
            add(dotless)

    return out

def extract_answer(text: str) -> str:
    """
    Extract a numeric answer from model output.
    Priority:
      1) Unwrap special '######Here is the answer' block with <|im_start|>...<|im_end|>
      2) Anchored numeric after 'Answer'/'Final answer' markers (return LAST)
      3) Fallback to LAST numeric in the text
    Returns newline-separated variants to maximize exact_match success.
    """
    if not text:
        return "no_final_answer"

    raw = text

    # 1) Unwrap special block (if present)
    t = raw.strip()
    if "######Here is the answer" in t:
        after = t.split("######Here is the answer", 1)[1]
        m = re.search(r"<\|im_start\|>(.*?)<\|im_end\|>", after, flags=re.DOTALL | re.IGNORECASE)
        t = m.group(1) if m else after
    # normalize whitespace for anchor scan
    t_norm = re.sub(r"\s+", " ", t).strip()

    # 2) Anchored scan (prefer LAST)
    anchored = re.findall(
        rf"(?:final\s*)?(?:answer|ans\.?|result|correct\s*answer)\s*(?:is|:|=)?\s*({_NUM_RE})",
        t_norm,
        flags=re.IGNORECASE,
    )
    if anchored:
        variants = _canon_number_variants(anchored[-1])
        return "\n".join(variants)

    # 3) Fallback: last numeric anywhere
    nums = re.findall(_NUM_RE, t, flags=re.IGNORECASE)
    if not nums:
        return "no_final_answer"
    variants = _canon_number_variants(nums[-1])
    return "\n".join(variants)