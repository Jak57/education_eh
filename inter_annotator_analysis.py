#!/usr/bin/env python3
"""
Inter-annotator agreement analysis for the faithfulness/informativeness MTurk
runs (general_prompt_scoring_faith_info.html batches).

Reuses the loading / blinding-decode / Krippendorff machinery from
analyze_mturk_results.py. Excludes low-quality workers (default: the three
flagged in the Batch_5399666 screen), decodes slots back to true roles, and
reports agreement on every HIT that still has >= 2 assignments:

  * Krippendorff's alpha (interval) -- pooled and per role x dimension
  * exact agreement (same radio) and within-one-step (<= 0.25) rates
  * mean pairwise |difference|
  * worker-pair coverage, and the largest disagreements for manual review

Outputs mturk_analysis_out/iaa_pairs.csv (one row per rated pair per question)
plus the printed report.

Usage:
    python inter_annotator_analysis.py                          # defaults
    python inter_annotator_analysis.py --results Batch_5399666_batch_results.csv \
        --exclude-workers A1IOMFFEKCWOIT A1EX0MEOPF8AHT A2KJ983WWTEK4L
    python inter_annotator_analysis.py --exclude-workers        # exclude nobody
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

from analyze_mturk_results import (QUESTIONS, ROLES, decode_blinding,
                                   krippendorff_alpha_interval,
                                   llm_scores_for_items, load_results)

FLAGGED_DEFAULT = ["A1IOMFFEKCWOIT", "A1EX0MEOPF8AHT", "A2KJ983WWTEK4L"]
DEFAULT_RESULTS = ["Batch_5399666_batch_results.csv"]
STEP = 0.25                       # radio step for the "within one step" rate


def pair_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, role, dimension, annotator-pair).

    Units are ITEMS (source, summary_id), not HITs: batch-2 rows were copied
    verbatim from batch 1 (same texts, same slot blinding), so raters from
    different batches rated identical content and are poolable.
    """
    rows = []
    for (src, sid, role), grp in long.groupby(["source", "summary_id", "role"]):
        for dim in ("faith", "info"):
            rated = grp.dropna(subset=[dim])
            for a, b in combinations(rated.itertuples(), 2):
                va, vb = getattr(a, dim), getattr(b, dim)
                rows.append({
                    "source": src, "summary_id": sid,
                    "role": role, "dim": dim,
                    "worker_1": a.WorkerId, "worker_2": b.WorkerId,
                    "val_1": va, "val_2": vb, "abs_diff": abs(va - vb),
                })
    return pd.DataFrame(rows)


def alpha_for(long: pd.DataFrame, dim: str, roles=None) -> float:
    units = []
    sub = long if roles is None else long[long["role"].isin(roles)]
    for _, grp in sub.groupby(["source", "summary_id", "role"]):
        units.append(grp[dim].dropna().values)
    return krippendorff_alpha_interval(units)


def worker_composites(long: pd.DataFrame) -> pd.DataFrame:
    """Per (item, worker) composite scores, mirroring the pipeline's metrics."""
    rows = []
    for (src, sid, aid), grp in long.groupby(["source", "summary_id", "AssignmentId"]):
        g = grp.set_index("role")
        def get(role, dim):
            return g.loc[role, dim] if role in g.index else np.nan
        rows.append({
            "source": src, "summary_id": sid, "worker": grp["WorkerId"].iloc[0],
            "base_A_id": grp["base_A_id"].iloc[0],
            "candidate_id": grp["candidate_id"].iloc[0],
            "faith1": np.nanmean([get("base_A", "faith"), get("base_B", "faith")]),
            "faith2": np.nanmin([get("base_A", "faith"), get("base_B", "faith")]),
            "info1": np.nanmean([get("base_A", "info"), get("base_B", "info")]),
            "info2": np.nanmin([get("base_A", "info"), get("base_B", "info")]),
            "spec_faith": np.nanmean([get("base_A", "faith"), get("candidate", "faith")]),
            "spec_info": np.nanmean([get("base_A", "info"), get("candidate", "info")]),
        })
    return pd.DataFrame(rows)


HL_METRICS = [("faith1", "llm_faithfulness1"),
              ("faith2", "llm_faithfulness2"),
              ("info1", "llm_informativeness1"),
              ("info2", "llm_informativeness2"),
              ("spec_faith", "llm_spec_faithfulness"),
              ("spec_info", "llm_spec_informativeness")]


def report_human_vs_llm(long: pd.DataFrame, trees_dir: str = "Results") -> None:
    """Human-human vs human-LLM correlation on the same items.

    The LLM is treated as one more annotator: r(human, LLM) pools each
    individual worker's composite against the LLM's score for that item, and is
    directly comparable to r(human, human), which pools all co-rater pairs
    (symmetrized). Items need >= 2 raters; every extra rater adds pairs.
    """
    comp = worker_composites(long)
    llm = llm_scores_for_items(
        comp[["source", "summary_id", "base_A_id", "candidate_id"]].drop_duplicates(),
        trees_dir)
    if llm.empty:
        print("\n(no LLM scores found -- human-vs-LLM section skipped)")
        return
    comp = comp.merge(llm, on=["source", "summary_id"], how="left")

    counts = comp.groupby(["source", "summary_id"])["worker"].nunique()
    multi = set(counts[counts >= 2].index)
    comp["is_multi"] = list(zip(comp["source"], comp["summary_id"]))
    comp["is_multi"] = comp["is_multi"].isin(multi)
    m = comp[comp["is_multi"]]

    print(f"\n=== Human-human vs human-LLM (items with >=2 raters: {len(multi)}; "
          f"rater counts {counts[counts >= 2].value_counts().sort_index().to_dict()}) ===")
    print(f"{'metric':<11} {'r(H,H)':>8} {'pairs':>6} | {'r(H,LLM)':>9} {'n':>5} "
          f"| {'r(avgH,LLM)':>12} {'items':>6}")
    for h, l in HL_METRICS:
        xs, ys = [], []
        for _, grp in m.groupby(["source", "summary_id"]):
            vals = grp[h].dropna().values
            for a, b in combinations(vals, 2):
                xs += [a, b]; ys += [b, a]          # symmetrized pairs
        r_hh = stats.pearsonr(xs, ys)[0] if len(xs) >= 6 else float("nan")

        sub = m[[h, l]].dropna()
        r_hl = stats.pearsonr(sub[h], sub[l])[0] if len(sub) >= 6 else float("nan")

        avg = (m.groupby(["source", "summary_id"])
               .agg(**{h: (h, "mean"), l: (l, "first")}).dropna())
        r_avg, p_avg = (stats.pearsonr(avg[h], avg[l])
                        if len(avg) >= 6 else (float("nan"), 1))
        print(f"{h:<11} {r_hh:>8.3f} {len(xs)//2:>6} | {r_hl:>9.3f} {len(sub):>5} "
              f"| {r_avg:>11.3f}{'*' if p_avg < 0.05 else ' '} {len(avg):>6}")
    print("  r(H,H): pooled co-rater pairs (symmetrized). r(H,LLM): each individual")
    print("  worker vs the LLM on the same items. r(avgH,LLM): workers averaged first.")
    print("  If r(H,LLM) ~ r(H,H), the LLM agrees with a human about as well as")
    print("  humans agree with each other (r(H,H) is the practical ceiling).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=DEFAULT_RESULTS)
    ap.add_argument("--exclude-workers", nargs="*", default=FLAGGED_DEFAULT,
                    help="worker ids to drop (pass with no values to keep everyone)")
    ap.add_argument("--out-dir", default="mturk_analysis_out")
    args = ap.parse_args()

    frames = []
    for p in args.results:
        f = load_results(p)
        f["batch"] = Path(p).stem
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    if args.exclude_workers:
        n0 = len(df)
        df = df[~df["WorkerId"].isin(args.exclude_workers)]
        print(f"Excluded {sorted(set(args.exclude_workers))}: "
              f"{n0 - len(df)} assignments dropped, {len(df)} remain")

    long = decode_blinding(df)
    long = long.merge(df[["AssignmentId", "batch"]].drop_duplicates(),
                      on="AssignmentId")
    item_key = ["source", "summary_id"]
    n_per_item = long.groupby(item_key)["AssignmentId"].nunique()
    usable = n_per_item[n_per_item >= 2].index
    print(f"Items with >=2 remaining assignments: {len(usable)}/{len(n_per_item)} "
          f"| raters per usable item: "
          f"{n_per_item[n_per_item >= 2].value_counts().sort_index().to_dict()}")
    long = long.set_index(item_key).loc[usable].reset_index()

    pairs = pair_table(long)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "iaa_pairs.csv", index=False)

    print("\n=== Inter-annotator agreement (roles decoded, pairs per HIT) ===")
    print(f"{'scope':<26} {'alpha':>7} {'exact%':>7} {'<=0.25%':>8} {'mean|d|':>8} {'n pairs':>8}")
    for dim in ("faith", "info"):
        for label, roles in (
            (f"{dim} -- all roles", None),
            (f"{dim} -- children (A,B)", ["base_A", "base_B"]),
            (f"{dim} -- candidate (C)", ["candidate"]),
        ):
            sub = pairs[pairs["dim"] == dim]
            if roles is not None:
                sub = sub[sub["role"].isin(roles)]
            if not len(sub):
                continue
            alpha = alpha_for(long, dim, roles)
            exact = (sub["abs_diff"] == 0).mean() * 100
            near = (sub["abs_diff"] <= STEP).mean() * 100
            print(f"{label:<26} {alpha:>7.3f} {exact:>6.1f}% {near:>7.1f}% "
                  f"{sub['abs_diff'].mean():>8.3f} {len(sub):>8}")

    print("\n=== Worker-pair coverage (top 10) ===")
    wp = (pairs.groupby(["worker_1", "worker_2"])["summary_id"].nunique()
          .sort_values(ascending=False))
    for (w1, w2), n in wp.head(10).items():
        print(f"  {w1} x {w2}: {n} items")
    print(f"  ... {len(wp)} worker pairs total")

    print("\n=== Largest disagreements (abs diff >= 0.5) ===")
    big = pairs[pairs["abs_diff"] >= 0.5].sort_values("abs_diff", ascending=False)
    print(f"{len(big)} of {len(pairs)} pairs ({len(big)/len(pairs):.0%})")
    for _, r in big.head(12).iterrows():
        print(f"  {r['source'][:22]} item {r['summary_id']} | {r['dim']:5} {r['role']:9} "
              f"| {r['val_1']:g} vs {r['val_2']:g} | {r['worker_1'][:12]} vs {r['worker_2'][:12]}")

    report_human_vs_llm(long)

    print(f"\nWrote {out / 'iaa_pairs.csv'} ({len(pairs)} rows)")


if __name__ == "__main__":
    main()
