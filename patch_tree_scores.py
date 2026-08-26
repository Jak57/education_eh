#!/usr/bin/env python3
"""
patch_tree_scores.py — patch wb_scores on one or more existing tree JSON files.

Usage:
    # writes trees/my_tree_patched.json  (default: new file with _patched suffix)
    python patch_tree_scores.py trees/my_tree.json

    # custom suffix
    python patch_tree_scores.py trees/my_tree.json --suffix _wb_scored

    # all trees into a separate directory
    python patch_tree_scores.py trees/*.json --output-dir trees/patched/

    # overwrite in-place
    python patch_tree_scores.py trees/my_tree.json --inplace

    # custom CSV location
    python patch_tree_scores.py trees/my_tree.json --csv /path/to/dataset_WB_with_scores.csv

    # UltraFeedback trees (per-model scores come from HuggingFace, not GitHub)
    python patch_tree_scores.py Results/UF_random_prompts_64_base.json --dataset uf

What it does:
  1. Ensures the CSV has every available model score column.
     - --dataset wb (default): if the CSV is missing columns, it fetches only the
       missing ones from the WildBench GitHub eval results and merges them in.
     - --dataset uf: if the CSV is missing score_<model> columns, it extracts them
       from the openbmb/UltraFeedback HF dataset (mean of each completion's four
       GPT-4 aspect ratings, 1-5; missing models midpoint-filled with 3.0).
     - If the CSV does not exist at all, the script exits with a helpful message.
  2. Loads each tree JSON, runs add_scores + build_scores (the same logic as main.py).
  3. Writes the patched JSON back in-place (or to --output-dir if specified).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from score_retrieval import add_scores, build_scores
from tree import Tree

LISTING_URL = (
    "https://api.github.com/repos/allenai/WildBench/contents/"
    "eval_results/v2.0625/score.v2/eval%3Dgpt-4o-2024-05-13"
)

DEFAULT_CSV = {
    "wb": "dataset_WB_with_scores.csv",
    "uf": "dataset_UF_with_scores.csv",
}


def ensure_all_scores(csv_path: Path) -> None:
    """
    Verify the CSV exists and add any missing score_* columns by fetching
    only the absent model JSONs from GitHub.
    """
    if not csv_path.exists():
        sys.exit(
            f"[ERROR] Score CSV not found: {csv_path}\n"
            "Run dataset_prep/prepare_dataset.py first to create the base CSV "
            "(it downloads the WildBench dataset and all model score files)."
        )

    df = pd.read_csv(csv_path)
    existing_cols = set(df.columns)

    print("[INFO] Fetching model listing from GitHub …")
    resp = requests.get(LISTING_URL, timeout=30)
    resp.raise_for_status()
    all_entries = [e for e in resp.json() if e["name"].endswith(".json")]

    missing = [
        e for e in all_entries
        if f"score_{e['name'].replace('.json', '')}" not in existing_cols
    ]

    if not missing:
        print(f"[INFO] CSV already contains all {len(all_entries)} model score columns.")
        return

    print(f"[INFO] Fetching {len(missing)} missing model score files …")
    for entry in missing:
        model_name = entry["name"].replace(".json", "")
        col = f"score_{model_name}"
        print(f"  → {model_name}")
        score_resp = requests.get(entry["download_url"], timeout=60)
        score_resp.raise_for_status()
        data = json.loads(score_resp.text)
        score_df = (
            pd.DataFrame(data)[["session_id", "score"]]
            .rename(columns={"score": col})
        )
        df = df.merge(score_df, on="session_id", how="left")

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Added {len(missing)} new model columns → {csv_path}")


def ensure_all_scores_uf(csv_path: Path) -> None:
    """
    UltraFeedback counterpart of ensure_all_scores: the per-model scores live
    inside the HF dataset itself (each row's 4 completions carry GPT-4 aspect
    annotations), so instead of GitHub we load openbmb/UltraFeedback (cached
    after first download) and merge any missing score_<model> columns.
    """
    if not csv_path.exists():
        sys.exit(
            f"[ERROR] Score CSV not found: {csv_path}\n"
            "Run dataset_prep/prepare_ultrafeedback.py first to create the base CSV "
            "(aspect-mean score columns), then optionally "
            "dataset_prep/prepare_uf_model_scores.py for per-model columns."
        )

    sys.path.insert(0, str(Path(__file__).parent / "dataset_prep"))
    from prepare_uf_model_scores import MID_SCORE, completion_score

    df = pd.read_csv(csv_path)
    if "instructions" not in df.columns:
        sys.exit(f"[ERROR] {csv_path} has no 'instructions' column -- regenerate it "
                 "with dataset_prep/prepare_ultrafeedback.py")

    print("[INFO] Loading openbmb/UltraFeedback from HuggingFace (cached) …")
    from datasets import load_dataset
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    if len(ds) != len(df):
        sys.exit(f"[ERROR] row count mismatch: CSV {len(df)} vs dataset {len(ds)} -- "
                 "regenerate the CSV with dataset_prep/prepare_ultrafeedback.py")

    all_models = {c["model"] for r in ds for c in (r["completions"] or []) if c.get("model")}
    missing = sorted(m for m in all_models if f"score_{m}" not in df.columns)
    if not missing:
        print(f"[INFO] CSV already contains all {len(all_models)} model score columns.")
        return

    print(f"[INFO] Merging {len(missing)} missing model score columns …")
    per_model: dict[str, dict[int, float]] = {m: {} for m in missing}
    for i, row in enumerate(ds):
        for comp in row["completions"] or []:
            m = comp.get("model")
            if m in per_model:
                s = completion_score(comp)
                if s is not None:
                    per_model[m][i] = s
    for m in missing:
        col = f"score_{m}"
        print(f"  → {m}")
        df[col] = pd.Series(per_model[m]).reindex(df.index).fillna(MID_SCORE)

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Added {len(missing)} new model columns → {csv_path}")


def patch_tree(tree_json: Path, csv_path: Path, output_path: Path) -> None:
    print(f"[INFO] Loading tree: {tree_json}")
    tree = Tree.load(tree_json)

    print("[INFO] Attaching scores to leaf nodes …")
    add_scores(tree, str(csv_path))

    print("[INFO] Propagating scores up the hierarchy …")
    build_scores(tree)

    tree.dump(str(output_path))
    print(f"[INFO] Patched tree written → {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch wb_scores on existing tree JSON files without re-running main.py."
    )
    parser.add_argument(
        "tree_jsons",
        nargs="+",
        metavar="TREE_JSON",
        help="One or more tree JSON files to patch.",
    )
    parser.add_argument(
        "--dataset",
        choices=["wb", "uf"],
        default="wb",
        help="Score source: 'wb' fetches WildBench judge scores from GitHub, "
             "'uf' extracts UltraFeedback per-model scores from HuggingFace "
             "(default: wb).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="CSV",
        help="Path to the scores CSV (default: dataset_WB_with_scores.csv for "
             "--dataset wb, dataset_UF_with_scores.csv for --dataset uf).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to write patched files into. "
            "Defaults to the same directory as each input file."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="_patched",
        metavar="SUFFIX",
        help="Suffix appended to the stem of each output filename (default: _patched).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original file instead of creating a new one.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(DEFAULT_CSV[args.dataset])
    if args.dataset == "uf":
        ensure_all_scores_uf(csv_path)
    else:
        ensure_all_scores(csv_path)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for tree_json_str in args.tree_jsons:
        tree_json = Path(tree_json_str)
        if not tree_json.exists():
            print(f"[WARNING] File not found, skipping: {tree_json}")
            continue

        if args.inplace:
            output_path = tree_json
        else:
            stem = tree_json.stem + args.suffix
            base_dir = out_dir if out_dir else tree_json.parent
            output_path = base_dir / f"{stem}.json"

        patch_tree(tree_json, csv_path, output_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
