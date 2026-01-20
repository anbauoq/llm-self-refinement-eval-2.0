#!/usr/bin/env python3
# analysis.py

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_run_meta(run_dir: Path) -> Dict[str, Optional[str]]:
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


def find_run_dirs(root: Path, require_completed: bool) -> List[Path]:
    run_dirs: List[Path] = []
    for p in root.rglob("initial_inference.jsonl"):
        rd = p.parent
        if require_completed and not (rd / ".completed").exists():
            continue
        run_dirs.append(rd)
    return sorted(set(run_dirs))


def safe_ratio(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def count_correct(rows: List[Dict]) -> int:
    return sum(1 for r in rows if r.get("is_correct", False))


def index_by_id(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in rows:
        if "id" in r:
            out[str(r["id"])] = r
    return out


@dataclass
class RunStats:
    model: str
    dataset: str
    max_tokes: str

    n_total: int
    n_init_correct: int
    n_incorrect_answer: int
    n_corrected_answer: int

    posthint_accuracy: float          # (init correct + corrected) / total
    correction_rate: float            # corrected / initially wrong
    delta_accuracy: float             # posthint_accuracy - initial_accuracy


def analyze_one_run(run_dir: Path) -> Optional[RunStats]:
    meta = parse_run_meta(run_dir)
    model = meta["model"] or "UNKNOWN_MODEL"
    dataset = meta["dataset"] or "UNKNOWN_DATASET"
    max_tokes = meta["max_tokes"] or "0"

    initial = read_jsonl(run_dir / "initial_inference.jsonl")
    if not initial:
        return None

    post = read_jsonl(run_dir / "post_hint_inference.jsonl")

    n_total = len(initial)
    n_init_correct = count_correct(initial)
    n_wrong = n_total - n_init_correct

    if not post or n_wrong == 0:
        n_corrected = 0
        posthint_acc = safe_ratio(n_init_correct, n_total)
        initial_acc = safe_ratio(n_init_correct, n_total)
        correction_rate = 0.0
        delta = posthint_acc - initial_acc

        return RunStats(
            model, dataset, max_tokes,
            n_total, n_init_correct, n_wrong,
            n_corrected, posthint_acc, correction_rate, delta
        )

    init_by_id = index_by_id(initial)
    post_by_id = index_by_id(post)
    have_ids = (len(init_by_id) == n_total) and (len(post_by_id) == len(post))

    if have_ids:
        post_ids = set(post_by_id.keys())

        if len(post_ids) <= n_wrong:
            n_corrected = sum(
                1 for pid in post_ids
                if post_by_id[pid].get("is_correct", False)
            )

        elif len(post_ids) == n_total:
            n_corrected = 0
            for pid in post_ids:
                if (not init_by_id[pid].get("is_correct", False)) and post_by_id[pid].get("is_correct", False):
                    n_corrected += 1
        else:
            intersect = post_ids & set(init_by_id.keys())
            n_corrected = sum(
                1 for pid in intersect
                if (not init_by_id[pid].get("is_correct", False)) and post_by_id[pid].get("is_correct", False)
            )
    else:
        n_corrected = count_correct(post) if len(post) == n_wrong else count_correct(post)

    initial_acc = safe_ratio(n_init_correct, n_total)
    posthint_acc = safe_ratio(n_init_correct + n_corrected, n_total)
    correction_rate = safe_ratio(n_corrected, n_wrong)
    delta = posthint_acc - initial_acc

    return RunStats(
        model, dataset, max_tokes,
        n_total, n_init_correct, n_wrong,
        n_corrected, posthint_acc, correction_rate, delta
    )


def aggregate_rows(rows: List[RunStats]) -> List[RunStats]:
    grouped: Dict[Tuple[str, str, str], List[RunStats]] = {}
    for r in rows:
        grouped.setdefault((r.model, r.dataset, r.max_tokes), []).append(r)

    out: List[RunStats] = []
    for (m, d, t), rs in grouped.items():
        n_total = sum(x.n_total for x in rs)
        n_init_correct = sum(x.n_init_correct for x in rs)
        n_wrong = sum(x.n_incorrect_answer for x in rs)
        n_corrected = sum(x.n_corrected_answer for x in rs)

        initial_acc = safe_ratio(n_init_correct, n_total)
        posthint_acc = safe_ratio(n_init_correct + n_corrected, n_total)
        correction_rate = safe_ratio(n_corrected, n_wrong)
        delta = posthint_acc - initial_acc

        out.append(RunStats(
            m, d, t,
            n_total, n_init_correct, n_wrong,
            n_corrected, posthint_acc, correction_rate, delta
        ))

    out.sort(key=lambda r: (r.dataset, r.model, int(r.max_tokes) if r.max_tokes.isdigit() else 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results/all_results")
    ap.add_argument("--out_csv", type=str, default="evaluation/analysis.csv")
    ap.add_argument("--require_completed", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    root = Path(args.results_root)
    run_dirs = find_run_dirs(root, require_completed=args.require_completed)

    stats: List[RunStats] = []
    for rd in run_dirs:
        s = analyze_one_run(rd)
        if s:
            stats.append(s)

    if args.aggregate:
        stats = aggregate_rows(stats)

    fieldnames = [
        "model",
        "dataset",
        "max_tokes",
        "initial_accuracy",
        "n_incorrect_answer",
        "n_corrected_answer",
        "posthint_accuracy",
        "correction_rate",
        "delta_accuracy",
    ]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in stats:
            initial_acc = safe_ratio(r.n_init_correct, r.n_total)
            w.writerow({
                "model": r.model,
                "dataset": r.dataset,
                "max_tokes": r.max_tokes,
                "initial_accuracy": f"{initial_acc:.6f}",
                "n_incorrect_answer": str(r.n_incorrect_answer),
                "n_corrected_answer": str(r.n_corrected_answer),
                "posthint_accuracy": f"{r.posthint_accuracy:.6f}",
                "correction_rate": f"{r.correction_rate:.6f}",
                "delta_accuracy": f"{r.delta_accuracy:.6f}",
            })

    print(f"Saved: {out_csv}")
    print(f"Rows: {len(stats)}")


if __name__ == "__main__":
    main()