import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

def load_data(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.
    Skips empty lines and surfaces the line number on JSON errors.
    """
    p = Path(path)
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {p}: {e}") from e
    return out


def save_data(data: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """
    Save an iterable of dict-like objects to a JSONL file.
    """
    p = Path(path)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

