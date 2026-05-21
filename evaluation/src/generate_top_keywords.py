"""
Generate top-keywords CSV from hint sentences, grouped by model/dataset/post-hint correctness.
Reads post_hint_inference.jsonl + hints.jsonl under a results root, runs KeyNMF, writes eval_results/top_keywords_<encoder>.csv.
"""
# Encoder used for KeyNMF (filename slug = encoder name with hyphens replaced by underscores)
DEFAULT_ENCODER = "all-mpnet-base-v2"
from pathlib import Path
import argparse
import json
from collections import Counter, defaultdict

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def pick_col(df: pd.DataFrame, candidates):
    """Return first existing column name from candidates; else None. Warns if a non-primary fallback is used."""
    for i, c in enumerate(candidates):
        if c in df.columns:
            if i > 0:
                print(f"[WARN] pick_col: primary column '{candidates[0]}' not found; using fallback '{c}'")
            return c
    return None


def top_keywords_keynmf(hint_sentences, encoder=DEFAULT_ENCODER, top_k=5):
    docs = [str(s).strip() for s in hint_sentences if s is not None and str(s).strip() != ""]
    if len(docs) == 0:
        return []

    vectorizer = CountVectorizer(
        min_df=1,
        max_df=1.0,
        stop_words="english",
        ngram_range=(1, 2),
    )

    model = KeyNMF(
        5,
        encoder=encoder,
        vectorizer=vectorizer,
        top_n=25,
    )

    keyword_dicts = model.extract_keywords(docs, fitting=True)

    counts = Counter()
    for kw in keyword_dicts:
        for w, _score in sorted(kw.items(), key=lambda x: x[1], reverse=True)[:5]:
            counts[w] += 1

    return [w for w, _c in counts.most_common(top_k)]


def main():
    parser = argparse.ArgumentParser(description="Build top_keywords.csv from hint/post-hint JSONL results.")
    _repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_repo_root / "results",
        help="Root directory containing model/dataset/max*/post_hint_inference.jsonl and hints.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output CSV path (default: eval_results/top_keywords_<encoder>.csv)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_ENCODER,
        help=f"Sentence encoder for KeyNMF (default: {DEFAULT_ENCODER})",
    )
    args = parser.parse_args()

    encoder = args.encoder
    encoder_slug = encoder.replace("-", "_")
    script_dir = Path(__file__).resolve().parent
    eval_results_dir = script_dir.parent / "eval_results"

    out_path = args.output
    if out_path is None:
        out_path = eval_results_dir / f"top_keywords_{encoder_slug}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ROOT = args.results_dir
    if not ROOT.exists():
        raise FileNotFoundError(f"Results root not found: {ROOT}")

    # --- collect sentences per (model, dataset, is_correct) across ALL max* ---
    bucket = defaultdict(list)

    post_files = list(ROOT.rglob("post_hint_inference.jsonl"))
    print(f"Found post_hint_inference.jsonl files: {len(post_files)}")

    missing_hints = 0
    merged_rows_total = 0

    for post_path in post_files:
        folder = post_path.parent

        hints_path = None
        for candidate in ["hints.jsonl"]:
            p = folder / candidate
            if p.exists():
                hints_path = p
                break
        if hints_path is None:
            candidates = list(folder.glob("hints*.jsonl"))
            if candidates:
                hints_path = candidates[0]

        if hints_path is None or not hints_path.exists():
            missing_hints += 1
            continue

        parts = post_path.parts
        root_name = ROOT.name
        if root_name in parts:
            i = parts.index(root_name)
            model_name = parts[i + 1] if len(parts) > i + 1 else "UNKNOWN_MODEL"
            dataset_name = parts[i + 2] if len(parts) > i + 2 else "UNKNOWN_DATASET"
            max_dir = parts[i + 3] if len(parts) > i + 3 else "UNKNOWN_MAX"
        else:
            model_name, dataset_name, max_dir = parts[-4], parts[-3], parts[-2]

        post_df = read_jsonl(post_path)
        hints_df = read_jsonl(hints_path)

        id_post = pick_col(post_df, ["id", "question_id", "qid", "sample_id"])
        id_hint = pick_col(hints_df, ["id", "question_id", "qid", "sample_id"])
        corr_col = pick_col(post_df, ["is_correct", "correct", "post_correct", "hint_correct"])
        hint_col = pick_col(hints_df, ["hint_sentence", "hint", "hint_text", "sentence"])

        if id_post is None or id_hint is None or corr_col is None or hint_col is None:
            print("\n[SKIP] Missing expected columns")
            print(" post file:", post_path)
            print(" hints file:", hints_path)
            print(" post cols:", list(post_df.columns)[:30])
            print(" hint cols:", list(hints_df.columns)[:30])
            continue

        merged = hints_df[[id_hint, hint_col]].merge(
            post_df[[id_post, corr_col]],
            left_on=id_hint,
            right_on=id_post,
            how="inner",
        )

        if merged.empty:
            print("\n[WARN] Merge produced 0 rows")
            print(" post file:", post_path)
            print(" hints file:", hints_path)
            print(" example ids post:", post_df[id_post].head(3).tolist())
            print(" example ids hint:", hints_df[id_hint].head(3).tolist())
            continue

        merged_rows_total += len(merged)

        for is_corr, sub in merged.groupby(corr_col):
            bucket[(model_name, dataset_name, bool(is_corr))].extend(sub[hint_col].tolist())

    print(f"Folders missing hints file: {missing_hints}")
    print(f"Total merged rows: {merged_rows_total}")
    print(f"Buckets created: {len(bucket)}")

    # --- build output ---
    results = []
    for (model_name, dataset_name, is_corr), sentences in sorted(bucket.items()):
        try:
            top5 = top_keywords_keynmf(sentences, encoder=encoder, top_k=5)
        except Exception as e:
            print(f"[WARN] KeyNMF failed for {model_name}/{dataset_name}/is_correct={is_corr} (n={len(sentences)}): {e}")
            top5 = []
        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "is_correct": is_corr,
            "n_hints": len(sentences),
            "top5_keywords": top5,
        })

    out = pd.DataFrame(results)

    if out.empty:
        raise RuntimeError(
            "No results produced. Check printed WARN/SKIP lines above to see why "
            "(missing hints files, missing columns, or merge by id failing)."
        )

    out = out.sort_values(["model", "dataset", "is_correct"], ascending=[True, True, False])
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
