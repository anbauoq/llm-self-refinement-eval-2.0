import os
import json
import argparse
import pandas as pd

RESULTS_ROOT = "results/all_results"

MODELS_REASONING = {
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-0528-Qwen3-8B",
    "DeepSeek-R1-Distill-Llama-8B",
    "Phi-4-mini-reasoning",
}

MODELS_NON_REASONING = {
    "gemma-2-2b-it",
    "Meta-Llama-3.1-8B-Instruct",
    "Phi-4-mini-instruct",
    "Qwen2.5-Math-1.5B-instruct",
    "Qwen2.5-Math-7B-instruct",
}

TOKENIZER_NAME_BY_MODEL = {
    "DeepSeek-R1-0528-Qwen3-8B": "Qwen/Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
    "Phi-4-mini-reasoning": "microsoft/Phi-4-mini-instruct",
    "Qwen2.5-Math-1.5B-instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen2.5-Math-7B-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
}


# ---------------- helpers ----------------

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_model_category(model):
    if model in MODELS_REASONING:
        return "Reasoning"
    if model in MODELS_NON_REASONING:
        return "Non-Reasoning"
    return "Unknown"


def tokenize_lengths(texts, tokenizer):
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [len(ids) for ids in enc["input_ids"]]


def load_ok_and_outputs(path):
    """
    Returns:
      ok_map[qid] = bool(is_correct)
      out_map[qid] = full_output (string)
      n_total
      n_correct
    """
    ok_map = {}
    out_map = {}
    n_total = 0
    n_correct = 0

    for obj in iter_jsonl(path):
        qid = str(obj["id"])
        ok = bool(obj["is_correct"])
        ok_map[qid] = ok
        out_map[qid] = obj["full_output"]
        n_total += 1
        if ok:
            n_correct += 1

    return ok_map, out_map, n_total, n_correct


# ---------------- per-question df (ALL QUESTIONS; nulls only if initially correct) ----------------

def build_per_question_df(results_root):
    """
    One row = one QUESTION (ALL questions from initial_inference.jsonl).

    IMPORTANT RULE (your request):
      - If initially correct (initial_correct == 1): set NULL for all hint/post-hint fields
      - If initially incorrect (initial_correct == 0): DO NOT null-out; we try to fill hint/post-hint fields.
        If hint text is missing for some initially-incorrect questions (data issue), hint_* fields will be NULL,
        but post_* fields will still be filled from post_hint_inference.jsonl.

    Columns:
      model, model_category, dataset, max_tokens,
      corrected, hint_tokens, hint_outcome,
      initial_inference_tokens, post_hint_inference_tokens,
      initial_correct, post_correct
    """
    from transformers import AutoTokenizer

    tokenizers = {}
    rows = []

    for model in sorted(os.listdir(results_root)):
        model_path = os.path.join(results_root, model)
        if not os.path.isdir(model_path):
            continue
        if model not in TOKENIZER_NAME_BY_MODEL:
            continue

        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME_BY_MODEL[model],
                use_fast=True
            )
        tok = tokenizers[model]
        model_category = normalize_model_category(model)

        for dataset in sorted(os.listdir(model_path)):
            dataset_path = os.path.join(model_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for max_tokens in sorted(os.listdir(dataset_path)):
                leaf = os.path.join(dataset_path, max_tokens)
                if not os.path.isdir(leaf):
                    continue

                init_p = os.path.join(leaf, "initial_inference.jsonl")
                post_p = os.path.join(leaf, "post_hint_inference.jsonl")
                hints_p = os.path.join(leaf, "hints.jsonl")
                if not (os.path.isfile(init_p) and os.path.isfile(post_p) and os.path.isfile(hints_p)):
                    continue

                init_ok, init_out, _, _ = load_ok_and_outputs(init_p)
                post_ok, post_out, _, _ = load_ok_and_outputs(post_p)

                # hint_sentence per qid (only exists for some questions)
                hint_text_by_id = {}
                for obj in iter_jsonl(hints_p):
                    qid = str(obj["id"])
                    hint_text_by_id[qid] = obj.get("hint_sentence", "")

                # Precompute token lengths for ALL initial outputs (one per qid)
                all_qids = list(init_ok.keys())
                init_lens_all = tokenize_lengths([init_out.get(q, "") for q in all_qids], tok)

                # For post-hint: we want lengths for ALL questions too (even if later nulled for initially-correct)
                post_lens_all = tokenize_lengths([post_out.get(q, "") for q in all_qids], tok)

                # Compute hint token length only for those with hint_sentence present
                hint_tokens_by_id = {}
                if hint_text_by_id:
                    hint_ids = list(hint_text_by_id.keys())
                    hint_lens = tokenize_lengths([hint_text_by_id[q] for q in hint_ids], tok)
                    hint_tokens_by_id = dict(zip(hint_ids, map(int, hint_lens)))

                for qid, ilen, plen in zip(all_qids, init_lens_all, post_lens_all):
                    was_init = bool(init_ok.get(qid, False))

                    if was_init:
                        # initially correct -> NULL hint/post-hint related fields (your requested rule)
                        rows.append({
                            "model": model,
                            "model_category": model_category,
                            "dataset": dataset,
                            "max_tokens": max_tokens,
                            "corrected": None,
                            "hint_tokens": None,
                            "hint_outcome": None,
                            "initial_inference_tokens": int(ilen),
                            "post_hint_inference_tokens": None,
                            "initial_correct": 1,
                            "post_correct": None,
                        })
                    else:
                        # initially incorrect -> fill post-hint fields (always), hint fields if available
                        was_post = bool(post_ok.get(qid, False))
                        corrected = (not was_init) and was_post

                        hlen = hint_tokens_by_id.get(qid, None)
                        rows.append({
                            "model": model,
                            "model_category": model_category,
                            "dataset": dataset,
                            "max_tokens": max_tokens,
                            "corrected": int(corrected),
                            "hint_tokens": int(hlen) if hlen is not None else None,
                            "hint_outcome": ("Corrected" if corrected else "Not corrected") if hlen is not None else None,
                            "initial_inference_tokens": int(ilen),
                            "post_hint_inference_tokens": int(plen),
                            "initial_correct": 0,
                            "post_correct": int(was_post),
                        })

    df = pd.DataFrame(rows)
    df = df[df["model_category"].isin(["Reasoning", "Non-Reasoning"])].copy()
    return df


# ---------------- metrics.csv (ALL QUESTIONS) ----------------

def build_metrics_all_questions(results_root):
    """
    One row per (model, dataset, max_tokens), computed over ALL questions.

    Uses:
      - n_questions from initial_inference.jsonl (total lines)
      - initial_accuracy = n_correct_init / n_questions
      - n_corrected = count of (init incorrect AND post correct)
      - post_hint_accuracy = (n_correct_init + n_corrected) / n_questions
      - correction_rate = n_corrected / n_incorrect_initial
      - token means computed over ALL full_output in each file
    """
    from transformers import AutoTokenizer

    tokenizers = {}
    rows = []

    for model in sorted(os.listdir(results_root)):
        model_path = os.path.join(results_root, model)
        if not os.path.isdir(model_path):
            continue
        if model not in TOKENIZER_NAME_BY_MODEL:
            continue

        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME_BY_MODEL[model],
                use_fast=True
            )
        tok = tokenizers[model]
        model_category = normalize_model_category(model)

        for dataset in sorted(os.listdir(model_path)):
            dataset_path = os.path.join(model_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for max_tokens in sorted(os.listdir(dataset_path)):
                leaf = os.path.join(dataset_path, max_tokens)
                if not os.path.isdir(leaf):
                    continue

                init_p = os.path.join(leaf, "initial_inference.jsonl")
                post_p = os.path.join(leaf, "post_hint_inference.jsonl")
                hints_p = os.path.join(leaf, "hints.jsonl")
                if not (os.path.isfile(init_p) and os.path.isfile(post_p) and os.path.isfile(hints_p)):
                    continue

                init_ok, init_out, n_questions, n_correct_init = load_ok_and_outputs(init_p)
                post_ok, post_out, _, _ = load_ok_and_outputs(post_p)

                # corrected = initially incorrect AND post correct
                n_corrected = 0
                for qid, ok_init in init_ok.items():
                    ok_post = bool(post_ok.get(qid, False))
                    if (not ok_init) and ok_post:
                        n_corrected += 1

                n_incorrect_initial = n_questions - n_correct_init

                initial_accuracy = (n_correct_init / n_questions) if n_questions else 0.0
                post_hint_accuracy = ((n_correct_init + n_corrected) / n_questions) if n_questions else 0.0
                delta_accuracy = post_hint_accuracy - initial_accuracy
                correction_rate = (n_corrected / n_incorrect_initial) if n_incorrect_initial else 0.0

                # token means over ALL outputs
                init_texts_all = list(init_out.values())
                post_texts_all = list(post_out.values())
                hint_texts_all = [obj["full_output"] for obj in iter_jsonl(hints_p)]

                initial_tokens_mean = float(sum(tokenize_lengths(init_texts_all, tok)) / len(init_texts_all)) if init_texts_all else 0.0
                post_hint_tokens_mean = float(sum(tokenize_lengths(post_texts_all, tok)) / len(post_texts_all)) if post_texts_all else 0.0
                hint_tokens_mean = float(sum(tokenize_lengths(hint_texts_all, tok)) / len(hint_texts_all)) if hint_texts_all else 0.0

                rows.append({
                    "model": model,
                    "model_category": model_category,
                    "dataset": dataset,
                    "max_tokens": max_tokens,
                    "n_questions": int(n_questions),
                    "initial_accuracy": float(initial_accuracy),
                    "post_hint_accuracy": float(post_hint_accuracy),
                    "n_incorrect_initial": int(n_incorrect_initial),
                    "n_corrected": int(n_corrected),
                    "correction_rate": float(correction_rate),
                    "delta_accuracy": float(delta_accuracy),
                    "initial_tokens_mean": float(initial_tokens_mean),
                    "hint_tokens_mean": float(hint_tokens_mean),
                    "post_hint_tokens_mean": float(post_hint_tokens_mean),
                })

    df = pd.DataFrame(rows)
    df = df[df["model_category"].isin(["Reasoning", "Non-Reasoning"])].copy()
    df = df.sort_values(["model", "dataset", "max_tokens"]).reset_index(drop=True)
    return df


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default=RESULTS_ROOT)
    parser.add_argument("--per_question_out", type=str, default="evaluation/outputs/per_question_tokens.csv")
    parser.add_argument("--metrics_out", type=str, default="evaluation/outputs/metrics.csv")
    args = parser.parse_args()

    print("Building per-question dataframe (ALL questions; nulls only for initially correct)...")
    df_questions = build_per_question_df(args.results_root)
    df_questions.to_csv(args.per_question_out, index=False)
    print(f"Saved {args.per_question_out} (rows={len(df_questions)})")

    print("Building metrics.csv over ALL questions...")
    df_metrics = build_metrics_all_questions(args.results_root)
    df_metrics.to_csv(args.metrics_out, index=False)
    print(f"Saved {args.metrics_out} (rows={len(df_metrics)})")

    print("Done.")


if __name__ == "__main__":
    main()