#!/usr/bin/env python3
"""
token_count_analysis.py

Counts tokenizer tokens in:
  1) initial_inference.jsonl -> "full_output"
  2) hints.jsonl            -> "hint_sentence"
  3) post_hint_inference.jsonl -> "full_output"

Traverses a results folder structured like:
results/all_results/<MODEL>/<DATASET>/max<MAXTOKES>/*.jsonl

Outputs a single CSV with aggregated stats per (model, dataset, max_tokens, filetype).

Usage:
  python scripts/token_count_analysis.py \
    --results_root results/all_results \
    --out_csv evaluation/token_counts.csv \
    --tokenizer_map_json scripts/tokenizer_map.json

If you don't pass --tokenizer_map_json, it uses a reasonable default mapping
and falls back to "gpt2" tokenizer if a model tokenizer can't be loaded.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Hugging Face tokenizers
from transformers import AutoTokenizer


# ----------------------------
# Minimal logging
# ----------------------------
logger = logging.getLogger("token_count_analysis")


def setup_logging() -> None:
    # Minimal + practical: INFO by default, switch to DEBUG via env if you want later.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


# ----------------------------
# IO helpers
# ----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        logger.warning("JSONL not found: %s", path)
        return []
    rows: List[Dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole run
                malformed += 1
                continue
    if malformed:
        logger.warning("Skipped %d malformed JSON lines in %s", malformed, path)
    return rows


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


# ----------------------------
# Path parsing
# ----------------------------
@dataclass(frozen=True)
class RunMeta:
    model: str
    dataset: str
    max_tokens: str
    run_dir: Path


def parse_run_meta_from_file(jsonl_path: Path) -> Optional[RunMeta]:
    """
    Expects path like:
      .../results/all_results/<MODEL>/<DATASET>/max<MAXTOKES>/<file>.jsonl
    """
    parts = jsonl_path.parts
    # Find the segment starting with "max"
    max_idx = None
    for i, p in enumerate(parts):
        if p.startswith("max"):
            max_idx = i
    if max_idx is None:
        logger.debug("Could not parse run meta (no max* segment): %s", jsonl_path)
        return None

    max_seg = parts[max_idx]
    max_tokens = max_seg.replace("max", "")
    dataset = parts[max_idx - 1] if max_idx - 1 >= 0 else "UNKNOWN_DATASET"
    model = parts[max_idx - 2] if max_idx - 2 >= 0 else "UNKNOWN_MODEL"
    run_dir = jsonl_path.parent
    return RunMeta(model=model, dataset=dataset, max_tokens=max_tokens, run_dir=run_dir)


# ----------------------------
# Tokenizer loading
# ----------------------------
DEFAULT_TOKENIZER_MAP: Dict[str, str] = {
    # You can adjust these to your exact HF repo names if needed.
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "Qwen2.5-Math-1.5B-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen2.5-Math-7B-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
    "Phi-4-mini-reasoning": "microsoft/Phi-4-mini-reasoning",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

_TOKENIZER_CACHE: Dict[str, Any] = {}


def load_tokenizer_for_model(model_name: str, tokenizer_map: Dict[str, str]) -> Any:
    """
    Loads and caches HF tokenizer for the given experiment model folder name.
    Falls back to a generic tokenizer if missing.
    """
    if model_name in _TOKENIZER_CACHE:
        logger.debug("Tokenizer cache hit: %s", model_name)
        return _TOKENIZER_CACHE[model_name]

    hf_name = tokenizer_map.get(model_name)
    try:
        if hf_name:
            logger.info("Loading tokenizer for model '%s' via HF id '%s'", model_name, hf_name)
            tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        else:
            # fallback: try using the model_name directly (sometimes folder name == HF name)
            logger.info("Loading tokenizer for model '%s' via folder name", model_name)
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        # last resort fallback
        logger.warning("Failed to load tokenizer for '%s' (%s). Falling back to gpt2.", model_name, type(e).__name__)
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # Some tokenizers (Llama) may not have pad token; set to eos to avoid warnings
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    _TOKENIZER_CACHE[model_name] = tok
    return tok


def count_tokens(tok: Any, text: str) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    # encode returns token ids; do NOT add special tokens for pure length comparison
    return len(tok.encode(text, add_special_tokens=False))


# ----------------------------
# Schema handling
# ----------------------------
def extract_text(row: Dict[str, Any], preferred_keys: List[str]) -> Optional[str]:
    """
    Returns the first present key in preferred_keys that contains non-empty text.
    """
    for k in preferred_keys:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, str) and v.strip() == "":
                continue
            return v if isinstance(v, str) else str(v)
    return None


# ----------------------------
# Main analysis
# ----------------------------
def find_jsonl_files(results_root: Path) -> List[Path]:
    """
    Collect all relevant jsonl files under results_root.
    """
    patterns = [
        "**/initial_inference.jsonl",
        "**/post_hint_inference.jsonl",
        "**/hints.jsonl",  # your hint_sentences file
    ]
    out: List[Path] = []
    for pat in patterns:
        out.extend(results_root.rglob(pat))
    return sorted(set(out))


def summarize_token_counts(token_counts: List[int]) -> Dict[str, float]:
    if not token_counts:
        return {
            "n_rows": 0,
            "tokens_sum": 0.0,
            "tokens_mean": 0.0,
            "tokens_median": 0.0,
            "tokens_p95": 0.0,
            "tokens_min": 0.0,
            "tokens_max": 0.0,
        }
    s = pd.Series(token_counts, dtype="int64")
    return {
        "n_rows": int(s.shape[0]),
        "tokens_sum": float(s.sum()),
        "tokens_mean": float(s.mean()),
        "tokens_median": float(s.median()),
        "tokens_p95": float(s.quantile(0.95)),
        "tokens_min": float(s.min()),
        "tokens_max": float(s.max()),
    }


def main() -> None:
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results/all_results")
    ap.add_argument("--out_csv", type=str, default="evaluation/token_counts.csv")
    ap.add_argument(
        "--tokenizer_map_json",
        type=str,
        default="",
        help="Optional JSON file mapping experiment model folder -> HF tokenizer name.",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("results_root=%s", results_root)
    logger.info("out_csv=%s", out_csv)
    if args.tokenizer_map_json:
        logger.info("tokenizer_map_json=%s", args.tokenizer_map_json)

    # Load tokenizer map if provided
    tokenizer_map = dict(DEFAULT_TOKENIZER_MAP)
    if args.tokenizer_map_json:
        p = Path(args.tokenizer_map_json)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                user_map = json.load(f)
            if isinstance(user_map, dict):
                tokenizer_map.update({str(k): str(v) for k, v in user_map.items()})
                logger.info("Loaded %d tokenizer map overrides.", len(user_map))
        else:
            logger.warning("tokenizer_map_json path does not exist: %s", p)

    files = find_jsonl_files(results_root)
    logger.info("Found %d matching jsonl files.", len(files))
    if not files:
        print(f"No matching jsonl files found under: {results_root}")
        return

    rows_out: List[Dict[str, Any]] = []

    for i, fp in enumerate(files, start=1):
        meta = parse_run_meta_from_file(fp)
        if meta is None:
            logger.debug("Skipping file (meta parse failed): %s", fp)
            continue

        model = meta.model
        dataset = meta.dataset
        max_tokens = meta.max_tokens

        logger.info("[%d/%d] Processing %s (model=%s dataset=%s max=%s)", i, len(files), fp.name, model, dataset, max_tokens)

        tok = load_tokenizer_for_model(model, tokenizer_map)

        data = read_jsonl(fp)
        if fp.name == "initial_inference.jsonl":
            kind = "initial"
            text_keys = ["full_output", "output", "response", "completion"]
        elif fp.name == "post_hint_inference.jsonl":
            kind = "post_hint"
            text_keys = ["full_output", "output", "response", "completion"]
        else:  # hints.jsonl
            kind = "hints"
            text_keys = ["hint_sentence", "hint", "hint_text", "sentence"]

        token_counts: List[int] = []
        missing_text = 0

        for r in data:
            txt = extract_text(r, text_keys)
            if txt is None:
                missing_text += 1
                continue
            token_counts.append(count_tokens(tok, txt))

        stats = summarize_token_counts(token_counts)

        logger.info(
            "Done %s: n_rows=%d missing_text_rows=%d tokens_mean=%.2f tokens_p95=%.2f",
            kind,
            stats["n_rows"],
            missing_text,
            stats["tokens_mean"],
            stats["tokens_p95"],
        )

        rows_out.append({
            "model": model,
            "dataset": dataset,
            "max_tokens": safe_int(max_tokens, 0),
            "file_type": kind,
            "source_file": str(fp),
            "missing_text_rows": missing_text,
            **stats,
        })

    df = pd.DataFrame(rows_out)

    # Sort nicely
    if not df.empty:
        df = df.sort_values(["dataset", "model", "max_tokens", "file_type"]).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    logger.info("Saved CSV: %s", out_csv)
    logger.info("Rows: %d", len(df))
    logger.info("Columns: %s", list(df.columns))

    print(f"Saved: {out_csv}")
    print(f"Rows: {len(df)}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
