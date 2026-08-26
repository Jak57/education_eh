#!/usr/bin/env python3
"""
Add per-model score columns to dataset_UF_with_scores.csv -- the UltraFeedback
parallel of the WildBench flow in dataset_prep/prepare_dataset.py (which fetches
per-model GPT-4o judge JSONs from the WildBench GitHub repo and merges them as
score_<model> columns).

UltraFeedback needs no external fetch: each row already carries 4 completions
from a rotating pool of ~17 models, each completion annotated by GPT-4 on four
aspects (1-5). The per-model score for a prompt is the MEAN of that completion's
four aspect ratings, exposed as score_<model> (e.g. score_llama-2-70b-chat,
score_gpt-3.5-turbo) so score_retrieval.add_scores() picks them up unchanged.

Coverage caveat (unlike WildBench, where every model scored every prompt): each
UF prompt was answered by only 4 of the ~17 models, so each score_<model>
column has values for ~4/17 of rows. Missing cells are filled with 3.0 (the 1-5
midpoint, same convention as the aspect columns) because add_scores() would
otherwise default them to 5.0 -- the scale MAXIMUM -- when averaging up the
tree. Interpret per-model averages accordingly: they are pulled toward 3.0 in
proportion to how rarely the model was sampled.

Usage:
    conda run -n EntailmentHierarchy python dataset_prep/prepare_uf_model_scores.py
    ... [--csv dataset_UF_with_scores.csv]
"""
import argparse
import re

import pandas as pd
from datasets import load_dataset

ASPECTS = ("instruction_following", "truthfulness", "honesty", "helpfulness")
MID_SCORE = 3.0


def parse_rating(raw) -> float | None:
    """Annotation ratings are strings like '4', 'N/A', sometimes '4.5'."""
    if raw is None:
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(raw))
    if not m:
        return None
    v = float(m.group())
    return v if 1.0 <= v <= 5.0 else None


def completion_score(comp) -> float | None:
    """Mean of the completion's four aspect ratings (1-5); None if unparseable."""
    ann = comp.get("annotations") or {}
    vals = [v for a in ASPECTS
            if (v := parse_rating((ann.get(a) or {}).get("Rating"))) is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="dataset_UF_with_scores.csv",
                    help="scores CSV from prepare_ultrafeedback.py (updated in place)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.csv}")

    print("[INFO] Loading openbmb/UltraFeedback (train split, cached)...")
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    assert len(ds) == len(df), (
        f"row count mismatch: CSV {len(df)} vs dataset {len(ds)} -- "
        "regenerate the CSV with prepare_ultrafeedback.py first")

    # positional alignment (CSV was written by iterating ds in order); verify
    # on a text sample rather than trusting it blindly
    for i in (0, len(df) // 2, len(df) - 1):
        assert df["instructions"].iloc[i] == (ds[i]["instruction"] or ""), \
            f"instruction mismatch at row {i} -- CSV/dataset out of sync"

    per_model: dict[str, dict[int, float]] = {}
    for i, row in enumerate(ds):
        for comp in row["completions"] or []:
            s = completion_score(comp)
            if s is not None and comp.get("model"):
                per_model.setdefault(comp["model"], {})[i] = s

    print(f"[INFO] Found {len(per_model)} models")
    for model in sorted(per_model):
        col = f"score_{model}"
        series = pd.Series(per_model[model])
        df[col] = series.reindex(df.index).fillna(MID_SCORE)
        print(f"  {col:35} coverage {len(series):6}/{len(df)} "
              f"({len(series)/len(df):.0%}), mean where scored {series.mean():.2f}")

    df.to_csv(args.csv, index=False)
    n_scores = sum(1 for c in df.columns if c.startswith("score_"))
    print(f"[INFO] Wrote {args.csv}: {len(df)} rows, {n_scores} score_* columns")


if __name__ == "__main__":
    main()
