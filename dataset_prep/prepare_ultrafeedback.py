#!/usr/bin/env python3
"""
Prepare UltraFeedback (openbmb/UltraFeedback) inputs for the hierarchy pipeline,
mirroring the WildBench setup (dataset_prep/prepare_dataset.py + xlsx_to_base_json.py):

  1. dataset_UF_with_scores.csv   -- scores CSV shaped exactly like
     dataset_WB_with_scores.csv so score_retrieval.add_scores() works unchanged:
     a text column named `instructions` + numeric columns prefixed `score_`.
     UltraFeedback has no dense per-model judge scores (each row's 4 completions
     come from a rotating pool of models), so the score_* columns are the MEAN
     over the row's 4 completions of each GPT-4 annotation aspect (1-5 scale):
         score_instruction_following, score_truthfulness, score_honesty,
         score_helpfulness, score_overall (mean of the four aspects)
     Missing/unparseable ratings are skipped; an all-missing cell becomes 3.0
     (scale midpoint) here so add_scores()'s 5.0 default never fires.

  2. Results/UF_{random,code,story}_prompts_64_base.json -- three 64-prompt
     level-0 base trees in the same node shape the FINAL_* files used, sampled
     with keyword heuristics that mimic the WildBench random/code/story split
     (UltraFeedback has no task-type tags, only source datasets).

Usage:
    conda run -n EntailmentHierarchy python dataset_prep/prepare_ultrafeedback.py
    ... [--n 64] [--seed 42] [--min-chars 200] [--max-chars 4000]
        [--out-csv dataset_UF_with_scores.csv] [--out-dir Results]
"""
import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ASPECTS = ("instruction_following", "truthfulness", "honesty", "helpfulness")
MID_SCORE = 3.0                     # 1-5 scale midpoint for all-missing cells

CODE_KEYWORDS = (
    "code", "python", "function", "javascript", "sql", "program", "script",
    "algorithm", "debug", "api", "class", "compile",
)
STORY_KEYWORDS = (
    "story", "write a", "poem", "character", "fiction", "essay", "narrative",
    "creative", "dialogue", "plot",
)


# --------------------------------------------------------------------------- #
# Garbled / non-English character detection (same rules as
# extract_specificity_mturk_csv.py: keep ASCII, typography, Latin/Greek letters;
# flag other scripts, math-alphanumerics, combining marks, control/format chars)
# --------------------------------------------------------------------------- #
_ALLOWED_LETTER_PREFIXES = ("LATIN", "GREEK")


def _is_bad_char(c: str) -> bool:
    cp = ord(c)
    if cp < 128:
        return False
    if 0x1D400 <= cp <= 0x1D7FF:
        return True
    cat = unicodedata.category(c)
    if cat in ("Mn", "Mc", "Me"):
        return True
    if cat in ("Cc", "Cf", "Cs", "Co"):
        return True
    if cat[0] == "L":
        try:
            name = unicodedata.name(c)
        except ValueError:
            return True
        if not name.startswith(_ALLOWED_LETTER_PREFIXES):
            return True
    return False


def is_clean(s: str) -> bool:
    return not any(_is_bad_char(c) for c in s)


# --------------------------------------------------------------------------- #
# Scores CSV
# --------------------------------------------------------------------------- #
def parse_rating(raw) -> float | None:
    """Annotation ratings are strings like '4', 'N/A', sometimes '4.5'."""
    if raw is None:
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(raw))
    if not m:
        return None
    v = float(m.group())
    return v if 1.0 <= v <= 5.0 else None


def row_scores(completions) -> dict:
    """Mean rating per aspect over a row's completions (skip unparseable)."""
    per_aspect = {a: [] for a in ASPECTS}
    for comp in completions or []:
        ann = comp.get("annotations") or {}
        for a in ASPECTS:
            v = parse_rating((ann.get(a) or {}).get("Rating"))
            if v is not None:
                per_aspect[a].append(v)
    out = {f"score_{a}": (sum(v) / len(v) if v else MID_SCORE)
           for a, v in per_aspect.items()}
    out["score_overall"] = sum(out.values()) / len(out)
    return out


def build_scores_csv(ds, out_csv: str) -> pd.DataFrame:
    recs = []
    for row in ds:
        rec = {
            "id": row.get("id"),
            "source": row.get("source"),
            "instructions": row.get("instruction") or "",
        }
        rec.update(row_scores(row.get("completions")))
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {len(df)} rows -> {out_csv}")
    return df


# --------------------------------------------------------------------------- #
# Base JSONs (node shape identical to xlsx_to_base_json.py / the FINAL_* base files)
# --------------------------------------------------------------------------- #
def write_base_json(prompts: list[str], out_path: Path) -> None:
    nodes = [
        {"id": idx, "prompt": p, "match": None, "summary": None,
         "score": None, "evaluation": None, "alive": True}
        for idx, p in enumerate(prompts)
    ]
    payload = {"levels": [{"level": 0, "nodes": nodes}]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote {len(nodes)} nodes -> {out_path}")


def matches_any(text: str, keywords: tuple) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


def sample_categories(df: pd.DataFrame, n: int, seed: int,
                      min_chars: int, max_chars: int) -> dict:
    """Return {category: sub-DataFrame} of n sampled prompts each."""
    eligible = df[
        df["instructions"].str.len().between(min_chars, max_chars)
        & df["instructions"].map(is_clean)
    ].drop_duplicates(subset="instructions")
    print(f"[INFO] {len(eligible)}/{len(df)} prompts eligible "
          f"(clean chars, {min_chars}-{max_chars} chars, deduped)")

    rng = random.Random(seed)

    def take(pool: pd.DataFrame, k: int) -> pd.DataFrame:
        idx = rng.sample(list(pool.index), k)
        return pool.loc[idx]

    code_pool = eligible[eligible["instructions"].map(
        lambda t: matches_any(t, CODE_KEYWORDS))]
    story_pool = eligible[eligible["instructions"].map(
        lambda t: matches_any(t, STORY_KEYWORDS) and not matches_any(t, CODE_KEYWORDS))]
    code = take(code_pool, n)
    story = take(story_pool, n)
    used = set(code.index) | set(story.index)
    random_pool = eligible[
        ~eligible.index.isin(used)
        & ~eligible["instructions"].map(lambda t: matches_any(t, CODE_KEYWORDS)
                                        or matches_any(t, STORY_KEYWORDS))
    ]
    rand = take(random_pool, n)

    print(f"[INFO] pools -> code: {len(code_pool)}, story: {len(story_pool)}, "
          f"random: {len(random_pool)}")
    return {"code": code, "story": story, "random": rand}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64, help="prompts per category file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--out-csv", default="dataset_UF_with_scores.csv")
    ap.add_argument("--out-dir", default="Results")
    args = ap.parse_args()

    print("[INFO] Loading openbmb/UltraFeedback (train split)...")
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    print(f"[INFO] {len(ds)} rows loaded")

    df = build_scores_csv(ds, args.out_csv)

    samples = sample_categories(df, args.n, args.seed,
                                args.min_chars, args.max_chars)
    for cat, sub in samples.items():
        dist = Counter(sub["source"])
        print(f"[INFO] {cat}: source distribution {dict(dist)}")
        write_base_json(list(sub["instructions"]),
                        Path(args.out_dir) / f"UF_{cat}_prompts_{args.n}_base.json")


if __name__ == "__main__":
    main()
