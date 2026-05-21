import re
from typing import Optional, Tuple, List, Dict, Any

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    stem = item["question"].split("\n# Answer option:")[0].strip()

    opts = item["options"]  # already clean list like ["A)...", ...]
    q_formatted = stem + "\n\nOptions:\n" + "\n".join(opts)

    return {
        "id": item["id"],
        "question": q_formatted,
        "answer": item["answer"].strip().upper(),
        "options": opts,
    }

def extract_answer(text: str, options: Optional[List[str]] = None) -> str:
    """
    Extract final MCQ letter (A–E) from model output.

    Rule: ALWAYS return the answer candidate that appears LAST in the text.
    Candidates can be:
      - a letter A–E from <ans>/<answer>, \\boxed{...}, or anchored phrases
      - a numeric inside \\boxed{...} mapped to A–E using `options` (AQuA)
    """
    if not text or not text.strip():
        return "no_final_answer"

    t = text.strip()

    letter_candidates: List[Tuple[int, str]] = []  # (pos, letter)
    num_candidates: List[Tuple[int, float]] = []   # (pos, numeric)

    def _pick_last_letter(blob: str) -> Optional[str]:
        if not blob:
            return None
        s = blob.strip()

        m = re.fullmatch(r"[A-E]", s, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()

        toks = re.findall(r"\b([A-E])\b", s, flags=re.IGNORECASE)
        if toks:
            return toks[-1].upper()

        br = re.findall(
            r"[\(\[\{<\*\"']\s*([A-E])\s*[\)\]\}>\"\*']",
            s,
            flags=re.IGNORECASE,
        )
        if br:
            return br[-1].upper()

        return None

    def _unwrap_latex_wrappers(s: str) -> str:
        out = (s or "").strip()
        while True:
            new_out = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", out).strip()
            if new_out == out:
                break
            out = new_out
        return out

    def _extract_num(s: str) -> Optional[str]:
        if not s:
            return None
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
        return m.group(0) if m else None

    def _map_num_to_letter(target: float) -> Optional[str]:
        if not options:
            return None
        for opt in options:
            mo = re.match(r"\s*([A-E])\s*\)\s*(.*)$", (opt or "").strip(), flags=re.IGNORECASE)
            if not mo:
                continue
            letter = mo.group(1).upper()
            rhs = mo.group(2).strip()
            rhs_num = _extract_num(rhs)
            if not rhs_num:
                continue
            try:
                val = float(rhs_num)
            except Exception:
                continue
            if abs(val - target) <= 1e-6:
                return letter
        return None

    # A) <ans>/<answer> blocks
    for m in re.finditer(
        r"<\s*(ans|answer)\s*>(.*?)</\s*\1\s*>",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = (m.group(2) or "").strip()
        block = re.sub(r"</?\s*(?:ans|answer)\s*>", " ", block, flags=re.IGNORECASE).strip()
        letter = _pick_last_letter(block)
        if letter:
            letter_candidates.append((m.start(), letter))

    # B) \boxed{...} (collect BOTH letter and numeric boxed, with positions)
    for m in re.finditer(r"\\boxed\s*\{([^}]*)\}", t, flags=re.IGNORECASE | re.DOTALL):
        inside = _unwrap_latex_wrappers((m.group(1) or "").strip())

        letter = _pick_last_letter(inside)
        if letter:
            letter_candidates.append((m.start(), letter))
            continue

        if options:
            num_str = _extract_num(inside)
            if num_str:
                try:
                    num_candidates.append((m.start(), float(num_str)))
                except Exception:
                    pass

    # C) anchored phrases (STRICT)
    for m in re.finditer(
        r"""
        \b(?:the\s*)?
        final\s+
        answer
        \s*(?:is|:|=)\s*
        (?:\*\*|__)?\s*
        [\(\[\{<"']?\s*
        ([A-E])
        \s*[\)\]\}>\"']?
        \b
        """,
        t,
    ):
        letter_candidates.append((m.start(1), m.group(1).upper()))

    # ---- decide by the LAST thing in the text ----
    last_letter_pos = max((p for p, _ in letter_candidates), default=-1)
    last_num_pos = max((p for p, _ in num_candidates), default=-1)

    # If the last boxed thing is numeric and occurs AFTER any letter candidate, map it.
    if last_num_pos > last_letter_pos:
        target = next(v for p, v in num_candidates if p == last_num_pos)
        mapped = _map_num_to_letter(target)
        if mapped:
            return mapped
        # if mapping fails, we don't "pretend"—fall through to letters if any

    # Otherwise return last letter candidate
    if letter_candidates:
        letter_candidates.sort(key=lambda x: x[0])
        return letter_candidates[-1][1]

    # Last resort: if there were numeric boxed candidates but mapping failed
    return "no_final_answer"
