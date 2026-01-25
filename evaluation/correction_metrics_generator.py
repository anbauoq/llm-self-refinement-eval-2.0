#!/usr/bin/env python3
"""
correction_metrics_generator.py

This script combines two utilities:

1) Token counting over jsonl outputs (from token_count_analysis.py):
   - initial_inference.jsonl  -> "full_output"
   - hints(.jsonl)            -> "hint_sentence"
   - post_hint_inference.jsonl -> "full_output"

   Traverses:
     <results_root>/<MODEL>/<DATASET>/max<MAX_TOKENS>/*.jsonl

   Writes a CSV with one row per (model, dataset, max_tokens, file_type)
   and token statistics.

2) Accuracy / self-correction stats (from analysis.py):
   Computes initial accuracy, post-hint accuracy, correction rate, etc.
   Traverses the same directory structure and expects (by default) jsonl files named:
     - initial_inference.jsonl
     - post_hint_inference.jsonl
   (You can change filenames via CLI flags.)

You can run each part independently via subcommands:
  - tokens
  - accuracy
  - all  (runs both; optionally merges on model/dataset/max_tokens)

Examples
--------
# Token counts only
python results_analysis_combined.py tokens \
  --results_root results/all_results \
  --out_csv evaluation/token_counts.csv \
  --tokenizer_map_json scripts/tokenizer_map.json

# Accuracy only
python results_analysis_combined.py accuracy \
  --results_root results/all_results \
  --out_csv evaluation/posthint_accuracy.csv

# Run both + merge into one CSV
python results_analysis_combined.py all \
  --results_root results/all_results \
  --out_tokens_csv evaluation/token_counts.csv \
  --out_accuracy_csv evaluation/posthint_accuracy.csv \
  --out_merged_csv evaluation/merged_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# pandas + transformers are used in the token counting part
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# -------------------------
# Shared helpers
# -------------------------

def setup_logging(verbosity: int = 0) -> logging.Logger:
    """Create a module-level logger with sensible defaults."""
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("results_analysis")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read jsonl into a list of dicts. Returns [] if the file doesn't exist."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole run
                continue
    return rows


def categorize_model(model_name: str) -> str:
    """Categorize models into Reasoning / Non-Reasoning for analysis outputs."""
    reasoning_keywords = [
        "DeepSeek-R1",
        "Phi-4-mini-reasoning",
    ]
    for keyword in reasoning_keywords:
        if keyword in model_name:
            return "Reasoning"
    return "Non-Reasoning"


def simplify_model_name(model_name: str) -> str:
    """Create a shorter, human-readable model name for plotting."""
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


# -------------------------
# Part 1: Token counts
# -------------------------

@dataclass(frozen=True)
class RunMeta:
    model: str
    dataset: str
    max_tokens: int


def parse_run_meta_from_file(results_root: Path, fp: Path) -> Optional[RunMeta]:
    """
    Extract (model, dataset, max_tokens) from a file path like:
      <results_root>/<MODEL>/<DATASET>/max<MAXTOKENS>/<file>.jsonl
    """
    try:
        rel = fp.relative_to(results_root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 4:
        return None

    model, dataset, max_dir = parts[0], parts[1], parts[2]
    if not max_dir.startswith("max"):
        return None

    max_tokens = safe_int(max_dir.replace("max", ""), 0)
    return RunMeta(model=model, dataset=dataset, max_tokens=max_tokens)


def load_tokenizer_for_model(model: str, tokenizer_map: Dict[str, str], logger: logging.Logger):
    """
    Load a tokenizer for the given model name. Uses a map if provided, otherwise
    tries a few heuristics. Falls back to gpt2.
    """
    from transformers import AutoTokenizer  # type: ignore

    tok_name = tokenizer_map.get(model, None)

    # If not found, try common patterns.
    if tok_name is None:
        # Many model folder names are also valid HF tokenizer ids.
        tok_name = model

    try:
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        return tok
    except Exception as e:
        logger.debug("Could not load tokenizer '%s' for model '%s': %s", tok_name, model, e)
        # Hard fallback
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        return tok


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_text(row: Dict[str, Any], kind: str) -> Optional[str]:
    """
    For each file kind, extract the correct field.
    kind: "initial" | "hints" | "post_hint"
    """
    if kind in ("initial", "post_hint"):
        val = row.get("full_output", None)
    elif kind == "hints":
        val = row.get("hint_sentence", None)
    else:
        val = None

    if val is None:
        return None
    if isinstance(val, str):
        return val
    # Some pipelines store nested values; try to stringify safely.
    try:
        return str(val)
    except Exception:
        return None


def find_jsonl_files(results_root: Path) -> Iterable[Path]:
    """Yield all jsonl files under results_root."""
    yield from results_root.rglob("*.jsonl")


def summarize_token_counts(token_counts: List[int]) -> Dict[str, Any]:
    """
    Summarize a list of token counts into useful stats.
    """
    if not token_counts:
        return {
            "n_rows": 0,
            "total_tokens": 0,
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
            "p95_tokens": 0.0,
            "max_tokens_in_row": 0,
        }

    import numpy as np  # local import

    arr = np.array(token_counts, dtype=float)
    return {
        "n_rows": int(arr.size),
        "total_tokens": int(arr.sum()),
        "mean_tokens": float(arr.mean()),
        "median_tokens": float(np.median(arr)),
        "p95_tokens": float(np.percentile(arr, 95)),
        "max_tokens_in_row": int(arr.max()),
    }


def token_counts_main(
    results_root: Path,
    out_csv: Path,
    tokenizer_map_json: Optional[Path],
    verbosity: int = 0,
) -> None:
    logger = setup_logging(verbosity)

    if pd is None:
        raise RuntimeError("pandas is required for the token counting part, but couldn't be imported.")

    tokenizer_map: Dict[str, str] = {}
    if tokenizer_map_json and tokenizer_map_json.exists():
        tokenizer_map = json.loads(tokenizer_map_json.read_text(encoding="utf-8"))

    # Determine file kind from filename
    kind_by_name = {
        "initial_inference.jsonl": "initial",
        "hints.jsonl": "hints",
        "hint_sentences.jsonl": "hints",
        "post_hint_inference.jsonl": "post_hint",
        "posthint_inference.jsonl": "post_hint",
        "post_hint.jsonl": "post_hint",
    }

    rows_out: List[Dict[str, Any]] = []

    # Cache tokenizers per model
    tok_cache: Dict[str, Any] = {}

    for fp in find_jsonl_files(results_root):
        kind = kind_by_name.get(fp.name, None)
        if kind is None:
            continue

        meta = parse_run_meta_from_file(results_root, fp)
        if meta is None:
            continue

        if meta.model not in tok_cache:
            tok_cache[meta.model] = load_tokenizer_for_model(meta.model, tokenizer_map, logger)
        tok = tok_cache[meta.model]

        rows = read_jsonl(fp)
        token_counts: List[int] = []
        missing_text = 0

        for r in rows:
            text = extract_text(r, kind)
            if text is None:
                missing_text += 1
                continue
            token_counts.append(count_tokens(tok, text))

        stats = summarize_token_counts(token_counts)

        rows_out.append({
            "model": meta.model,
            "dataset": meta.dataset,
            "max_tokens": meta.max_tokens,
            "file_type": kind,
            "missing_text_rows": missing_text,
            **stats,
        })

    df = pd.DataFrame(rows_out)
    if not df.empty:
        df = df.sort_values(["dataset", "model", "max_tokens", "file_type"]).reset_index(drop=True)

    # Enrich with model metadata for downstream notebooks/plots
    df["model_category"] = df["model"].apply(categorize_model)
    df["model_short"] = df["model"].apply(simplify_model_name)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    logger.info("Saved token counts CSV: %s", out_csv)
    logger.info("Rows: %d", len(df))
    logger.info("Columns: %s", list(df.columns))


# -------------------------
# Part 2: Accuracy stats
# -------------------------

@dataclass(frozen=True)
class RunStats:
    model: str
    dataset: str
    max_tokens: int
    n_total: int
    n_init_correct: int
    n_post_correct: int
    n_incorrect_answer: int
    n_corrected_answer: int
    posthint_accuracy: float
    correction_rate: float
    delta_accuracy: float


def parse_run_meta(results_root: Path, run_dir: Path) -> Optional[Tuple[str, str, int]]:
    """
    Parse model/dataset/max_tokens from:
      <results_root>/<MODEL>/<DATASET>/max<MAXTOKENS>/
    """
    try:
        rel = run_dir.relative_to(results_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None
    model, dataset, max_dir = parts[0], parts[1], parts[2]
    if not max_dir.startswith("max"):
        return None
    max_tokens = safe_int(max_dir.replace("max", ""), 0)
    return model, dataset, max_tokens


def find_run_dirs(results_root: Path, require_completed: bool = False) -> List[Path]:
    """
    Find run directories: <model>/<dataset>/max<tokens>.
    If require_completed=True, only include those that contain a completion marker.
    """
    run_dirs: List[Path] = []
    for p in results_root.rglob("*"):
        if not p.is_dir():
            continue
        meta = parse_run_meta(results_root, p)
        if meta is None:
            continue
        if require_completed:
            # Completion marker is project-specific; keep permissive.
            # Users can enforce strictness with --require_completed false/true.
            if not any((p / f).exists() for f in ["completed.txt", "_completed", "done.txt"]):
                # If none exist, still allow if jsonl files exist
                if not any(p.glob("*.jsonl")):
                    continue
        run_dirs.append(p)
    # Deduplicate + sort for stable output
    run_dirs = sorted(set(run_dirs))
    return run_dirs


def index_by_id(rows: List[Dict[str, Any]], id_key: str = "id") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = r.get(id_key)
        if rid is None:
            continue
        out[str(rid)] = r
    return out


def count_correct(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Count correct rows and total rows based on common keys.
    We try these keys in order:
      - "is_correct" (bool)
      - "correct" (bool)
      - "accuracy" (0/1)
      - "label" / "pred" (exact match) if both exist
    """
    n_total = 0
    n_correct = 0

    for r in rows:
        n_total += 1

        if "is_correct" in r:
            n_correct += int(bool(r["is_correct"]))
            continue
        if "correct" in r:
            n_correct += int(bool(r["correct"]))
            continue
        if "accuracy" in r:
            try:
                n_correct += int(float(r["accuracy"]) > 0.5)
            except Exception:
                pass
            continue

        if "label" in r and "pred" in r:
            n_correct += int(str(r["label"]).strip() == str(r["pred"]).strip())

    return n_correct, n_total


def analyze_one_run(
    results_root: Path,
    run_dir: Path,
    initial_filename: str,
    post_filename: str,
) -> Optional[RunStats]:
    meta = parse_run_meta(results_root, run_dir)
    if meta is None:
        return None
    model, dataset, max_tokens = meta

    initial_path = run_dir / initial_filename
    post_path = run_dir / post_filename

    init_rows = read_jsonl(initial_path)
    post_rows = read_jsonl(post_path)

    if not init_rows and not post_rows:
        return None

    n_init_correct, n_total = count_correct(init_rows)
    n_post_correct, n_post_total = count_correct(post_rows)

    # Use init total as canonical; if missing, fall back to post.
    if n_total == 0:
        n_total = n_post_total

    # corrected answer: incorrect initially but correct post-hint
    # We can compute this precisely if both files share ids.
    init_by_id = index_by_id(init_rows, "id")
    post_by_id = index_by_id(post_rows, "id")

    n_incorrect_answer = 0
    n_corrected_answer = 0

    if init_by_id and post_by_id:
        shared_ids = set(init_by_id.keys()) & set(post_by_id.keys())
        for sid in shared_ids:
            i = init_by_id[sid]
            p = post_by_id[sid]

            # Determine correctness with same logic as count_correct on a single row
            def row_is_correct(r: Dict[str, Any]) -> Optional[bool]:
                if "is_correct" in r:
                    return bool(r["is_correct"])
                if "correct" in r:
                    return bool(r["correct"])
                if "accuracy" in r:
                    try:
                        return float(r["accuracy"]) > 0.5
                    except Exception:
                        return None
                if "label" in r and "pred" in r:
                    return str(r["label"]).strip() == str(r["pred"]).strip()
                return None

            ic = row_is_correct(i)
            pc = row_is_correct(p)

            if ic is False:
                n_incorrect_answer += 1
                if pc is True:
                    n_corrected_answer += 1
    else:
        # If we can't align by id, approximate corrected as (post_correct - init_correct) clipped.
        n_incorrect_answer = max(n_total - n_init_correct, 0)
        n_corrected_answer = max(min(n_post_correct - n_init_correct, n_incorrect_answer), 0)

    posthint_accuracy = safe_ratio(n_post_correct, n_total)
    correction_rate = safe_ratio(n_corrected_answer, n_incorrect_answer) if n_incorrect_answer else 0.0
    initial_accuracy = safe_ratio(n_init_correct, n_total)
    delta_accuracy = posthint_accuracy - initial_accuracy

    return RunStats(
        model=model,
        dataset=dataset,
        max_tokens=max_tokens,
        n_total=n_total,
        n_init_correct=n_init_correct,
        n_post_correct=n_post_correct,
        n_incorrect_answer=n_incorrect_answer,
        n_corrected_answer=n_corrected_answer,
        posthint_accuracy=posthint_accuracy,
        correction_rate=correction_rate,
        delta_accuracy=delta_accuracy,
    )


def aggregate_rows(rows: List[RunStats]) -> List[RunStats]:
    """
    Aggregate by (model, dataset, max_tokens).
    """
    from collections import defaultdict

    buckets: Dict[Tuple[str, str, int], List[RunStats]] = defaultdict(list)
    for r in rows:
        buckets[(r.model, r.dataset, r.max_tokens)].append(r)

    out: List[RunStats] = []
    for (model, dataset, max_tokens), rs in buckets.items():
        n_total = sum(x.n_total for x in rs)
        n_init_correct = sum(x.n_init_correct for x in rs)
        n_post_correct = sum(x.n_post_correct for x in rs)
        n_incorrect_answer = sum(x.n_incorrect_answer for x in rs)
        n_corrected_answer = sum(x.n_corrected_answer for x in rs)

        posthint_accuracy = safe_ratio(n_post_correct, n_total)
        initial_accuracy = safe_ratio(n_init_correct, n_total)
        correction_rate = safe_ratio(n_corrected_answer, n_incorrect_answer) if n_incorrect_answer else 0.0
        delta_accuracy = posthint_accuracy - initial_accuracy

        out.append(
            RunStats(
                model=model,
                dataset=dataset,
                max_tokens=max_tokens,
                n_total=n_total,
                n_init_correct=n_init_correct,
                n_post_correct=n_post_correct,
                n_incorrect_answer=n_incorrect_answer,
                n_corrected_answer=n_corrected_answer,
                posthint_accuracy=posthint_accuracy,
                correction_rate=correction_rate,
                delta_accuracy=delta_accuracy,
            )
        )

    return sorted(out, key=lambda r: (r.dataset, r.model, r.max_tokens))


def accuracy_main(
    results_root: Path,
    out_csv: Path,
    initial_filename: str,
    post_filename: str,
    aggregate: bool,
    require_completed: bool,
    verbosity: int = 0,
) -> None:
    logger = setup_logging(verbosity)

    run_dirs = find_run_dirs(results_root, require_completed=require_completed)

    stats: List[RunStats] = []
    for rd in run_dirs:
        s = analyze_one_run(results_root, rd, initial_filename=initial_filename, post_filename=post_filename)
        if s:
            stats.append(s)

    if aggregate:
        stats = aggregate_rows(stats)

    fieldnames = [
        "model",
        "model_category",
        "model_short",
        "dataset",
        "max_tokens",
        "n_total",
        "initial_accuracy",
        "n_incorrect_answer",
        "n_corrected_answer",
        "posthint_accuracy",
        "correction_rate",
        "delta_accuracy",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in stats:
            initial_acc = safe_ratio(r.n_init_correct, r.n_total)
            w.writerow({
                "model": r.model,
                "model_category": categorize_model(r.model),
                "model_short": simplify_model_name(r.model),
                "dataset": r.dataset,
                "max_tokens": r.max_tokens,
                "n_total": r.n_total,
                "initial_accuracy": f"{initial_acc:.6f}",
                "n_incorrect_answer": str(r.n_incorrect_answer),
                "n_corrected_answer": str(r.n_corrected_answer),
                "posthint_accuracy": f"{r.posthint_accuracy:.6f}",
                "correction_rate": f"{r.correction_rate:.6f}",
                "delta_accuracy": f"{r.delta_accuracy:.6f}",
            })

    logger.info("Saved accuracy CSV: %s", out_csv)
    logger.info("Rows: %d", len(stats))


# -------------------------
# Part 3: Combined runner
# -------------------------

def merge_csvs(tokens_csv: Path, accuracy_csv: Path, out_csv: Path, verbosity: int = 0) -> None:
    logger = setup_logging(verbosity)

    if pd is None:
        raise RuntimeError("pandas is required for merging, but couldn't be imported.")

    if not tokens_csv.exists():
        raise FileNotFoundError(tokens_csv)
    if not accuracy_csv.exists():
        raise FileNotFoundError(accuracy_csv)

    df_tokens = pd.read_csv(tokens_csv)
    df_acc = pd.read_csv(accuracy_csv)

    # For merging, aggregate token stats by (model, dataset, max_tokens, file_type) if needed.
    # Here we keep file_type granularity and merge accuracy onto each file_type row.
    keys = ["model", "dataset", "max_tokens"]
    df_merged = df_tokens.merge(df_acc, on=keys, how="left", suffixes=("", "_acc"))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_csv, index=False)

    logger.info("Saved merged CSV: %s", out_csv)
    logger.info("Rows: %d", len(df_merged))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="results_analysis_combined.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # tokens
    p_tok = sub.add_parser("tokens", help="Compute token counts/stats for jsonl outputs.")
    p_tok.add_argument("--results_root", type=str, required=True)
    p_tok.add_argument("--out_csv", type=str, required=True)
    p_tok.add_argument("--tokenizer_map_json", type=str, default=None)
    p_tok.add_argument("-v", "--verbose", action="count", default=0)

    # accuracy
    p_acc = sub.add_parser("accuracy", help="Compute accuracy/self-correction stats.")
    p_acc.add_argument("--results_root", type=str, required=True)
    p_acc.add_argument("--out_csv", type=str, required=True)
    p_acc.add_argument("--initial_filename", type=str, default="initial_inference.jsonl")
    p_acc.add_argument("--post_filename", type=str, default="post_hint_inference.jsonl")
    p_acc.add_argument("--aggregate", action="store_true", help="Aggregate across multiple run dirs.")
    p_acc.add_argument("--require_completed", action="store_true", help="Only include runs with completion marker or jsonl files.")
    p_acc.add_argument("-v", "--verbose", action="count", default=0)

    # all
    p_all = sub.add_parser("all", help="Run tokens + accuracy, optionally merge.")
    p_all.add_argument("--results_root", type=str, required=True)
    p_all.add_argument("--tokenizer_map_json", type=str, default=None)
    p_all.add_argument("--out_tokens_csv", type=str, required=True)
    p_all.add_argument("--out_accuracy_csv", type=str, required=True)
    p_all.add_argument("--out_merged_csv", type=str, default=None)
    p_all.add_argument("--initial_filename", type=str, default="initial_inference.jsonl")
    p_all.add_argument("--post_filename", type=str, default="post_hint_inference.jsonl")
    p_all.add_argument("--aggregate", action="store_true")
    p_all.add_argument("--require_completed", action="store_true")
    p_all.add_argument("-v", "--verbose", action="count", default=0)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "tokens":
        token_counts_main(
            results_root=Path(args.results_root),
            out_csv=Path(args.out_csv),
            tokenizer_map_json=Path(args.tokenizer_map_json) if args.tokenizer_map_json else None,
            verbosity=args.verbose,
        )
        return

    if args.cmd == "accuracy":
        accuracy_main(
            results_root=Path(args.results_root),
            out_csv=Path(args.out_csv),
            initial_filename=args.initial_filename,
            post_filename=args.post_filename,
            aggregate=bool(args.aggregate),
            require_completed=bool(args.require_completed),
            verbosity=args.verbose,
        )
        return

    if args.cmd == "all":
        results_root = Path(args.results_root)
        out_tokens = Path(args.out_tokens_csv)
        out_acc = Path(args.out_accuracy_csv)

        token_counts_main(
            results_root=results_root,
            out_csv=out_tokens,
            tokenizer_map_json=Path(args.tokenizer_map_json) if args.tokenizer_map_json else None,
            verbosity=args.verbose,
        )

        accuracy_main(
            results_root=results_root,
            out_csv=out_acc,
            initial_filename=args.initial_filename,
            post_filename=args.post_filename,
            aggregate=bool(args.aggregate),
            require_completed=bool(args.require_completed),
            verbosity=args.verbose,
        )

        if args.out_merged_csv:
            merge_csvs(out_tokens, out_acc, Path(args.out_merged_csv), verbosity=args.verbose)
        return


if __name__ == "__main__":
    main()
