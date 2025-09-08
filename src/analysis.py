#!/usr/bin/env python3
# analysis.py
# Compute run statistics and output:
# 1) Pretty text blocks to stdout (and save the same to a .txt file)
# 2) A CSV with columns: model, dataset, max_tokes, initial_accuracy, n_incorrect_answer,
#    n_corrected_answer, posthint_accuracy

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------- I/O ----------------

def read_jsonl(path: Path) -> List[Dict]:
    """Read JSONL; return [] if missing."""
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


# ------------- Metrics / parsing -------------

def safe_pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d > 0 else 0.0

def safe_ratio(n: int, d: int) -> float:
    return (n / d) if d > 0 else 0.0

def parse_run_meta(run_dir: Path) -> Dict[str, Optional[str]]:
    """
    Infer model/dataset/max_tokes from a directory like:
      results/.../<model>/<dataset>/max256/
    """
    parts = run_dir.parts
    try:
        max_idx = max(i for i, p in enumerate(parts) if p.startswith("max"))
    except ValueError:
        return {"model": None, "dataset": None, "max_tokes": None}

    max_seg = parts[max_idx]
    max_tokes = max_seg.replace("max", "")
    dataset = parts[max_idx - 1] if max_idx - 1 >= 0 else None
    model = parts[max_idx - 2] if max_idx - 2 >= 0 else None
    return {"model": model, "dataset": dataset, "max_tokes": max_tokes}

def find_run_dirs(root: Path) -> List[Path]:
    """Identify run dirs by presence of initial_inference.jsonl (post is optional)."""
    run_dirs: List[Path] = []
    for p in root.rglob("initial_inference.jsonl"):
        run_dirs.append(p.parent)
    # include dirs that only have post (rare); we'll skip later if initial missing
    for p in root.rglob("post_hint_inference.jsonl"):
        if p.parent not in run_dirs:
            run_dirs.append(p.parent)
    return sorted(run_dirs)


# ------------- Core aggregation -------------

def analyze(results_root: Path) -> Tuple[str, List[Dict[str, str]]]:
    """
    Returns:
      pretty_text (string with all blocks),
      rows (for CSV) with required columns.
    """
    blocks: List[str] = []
    csv_rows: List[Dict[str, str]] = []

    # stable ordering: dataset, model, max_tokes (int asc)
    def sort_key(p: Path) -> Tuple[str, str, int]:
        meta = parse_run_meta(p)
        ds = meta["dataset"] or ""
        md = meta["model"] or ""
        try:
            mt = int(meta["max_tokes"] or "0")
        except Exception:
            mt = 0
        return (ds, md, mt)

    for run_dir in sorted(find_run_dirs(results_root), key=sort_key):
        meta = parse_run_meta(run_dir)
        model = meta["model"] or ""
        dataset = meta["dataset"] or ""
        max_tokes = meta["max_tokes"] or ""

        # Need initial to compute both metrics
        initial = read_jsonl(run_dir / "initial_inference.jsonl")
        if not initial:
            continue
        post = read_jsonl(run_dir / "post_hint_inference.jsonl")

        n_total = len(initial)
        n_init_ok = sum(1 for r in initial if r.get("is_correct", False))
        n_wrong = n_total - n_init_ok
        n_post_ok = sum(1 for r in post if r.get("is_correct", False)) if post else 0

        # Metrics
        init_acc = safe_ratio(n_init_ok, n_total)                  # 0..1
        posthint_acc = safe_ratio(n_post_ok, n_wrong)              # 0..1; denom = initially wrong

        # Pretty block (exact format you asked for)
        block = (
            f"Model/Dataset: {model}/{dataset}\n"
            f"  Initial Accuracy: {n_init_ok}/{n_total} = {safe_pct(n_init_ok, n_total):.2f}%\n"
            f"  Post-Hint Accuracy on those initially wrong: {n_post_ok}/{n_wrong} = {posthint_acc*100:.2f}%\n"
        )
        blocks.append(block)

        # CSV row with the required columns (note the exact names requested)
        csv_rows.append({
            "model": model,
            "dataset": dataset,
            "max_tokes": max_tokes,                         # (spelled as requested)
            "initial_accuracy": f"{init_acc:.6f}",          # 0-1 range
            "n_incorrect_answer": str(n_wrong),
            "n_corrected_answer": str(n_post_ok),
            "posthint_accuracy": f"{posthint_acc:.6f}",     # 0-1 range
        })

    return ("\n".join(blocks).rstrip() + ("\n" if blocks else "")), csv_rows


# ------------- CLI -------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute and print/save run statistics.")
    ap.add_argument("--results_root", type=str, default="results",
                    help="Root directory containing the run outputs.")
    ap.add_argument("--out_txt", type=str, default="results/summary.txt",
                    help="Path to save the pretty text summary.")
    ap.add_argument("--out_csv", type=str, default="results/summary.csv",
                    help="Path to save the CSV summary.")
    args = ap.parse_args()

    root = Path(args.results_root)
    pretty_text, rows = analyze(root)

    if not rows:
        print(f"[analysis] No runs found under: {root}")
        return

    # 1) Print to stdout
    print(pretty_text, end="")

    # 2) Save .txt
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(pretty_text, encoding="utf-8")
    print(f"\nSaved text summary to: {out_txt}")

    # 3) Save .csv
    import csv
    fieldnames = ["model", "dataset", "max_tokes", "initial_accuracy",
                  "n_incorrect_answer", "n_corrected_answer", "posthint_accuracy"]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved CSV summary to: {out_csv}")


if __name__ == "__main__":
    main()