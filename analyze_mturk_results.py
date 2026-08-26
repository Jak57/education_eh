#!/usr/bin/env python3
"""
Analyze MTurk worker results for the faithfulness / informativeness experiment
(general_prompt_scoring_faith_info.html + extract_specificity_mturk_csv.py).

What it does:
  1. Loads the MTurk batch results CSV, drops Rejected assignments.
  2. DECODES THE BLINDING: workers answered per display slot (A / B / third);
     Input.slot_*_role maps each slot back to its true role
     (base_A = owner child, base_B = partner child, candidate = third-NN).
  3. Flags low-quality assignments (straightliners, too fast, incomplete).
  4. Aggregates per item (summary) across workers and joins the LLM judge
     scores from the hierarchy JSONs (faithfulness1/2, informativeness1/2,
     entail_1/2, specificity, specificity_faithfulness/informativeness).
  5. Reports human<->LLM correlations (Pearson, Spearman), MAE, position-bias
     (slot effects), and inter-rater agreement (Krippendorff's interval alpha).

Outputs (to --out-dir):
  per_assignment_long.csv  one row per assignment x role, blinding decoded
  per_item_summary.csv     one row per item: human means + LLM scores

Usage:
    python analyze_mturk_results.py                       # defaults below
    python analyze_mturk_results.py --results Batch_5396880_batch_results.csv \
        --trees-dir Results --out-dir mturk_analysis_out --strict
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_RESULTS = "Batch_5396880_batch_results.csv"
DEFAULT_TREES_DIR = "Results"
DEFAULT_OUT_DIR = "mturk_analysis_out"
MIN_WORK_SECONDS = 120          # below this an assignment is flagged "too fast"

ROLES = ("base_A", "base_B", "candidate")
SLOTS = ("A", "B", "third")
QUESTIONS = [f"{d}_{s}" for d in ("faith", "info") for s in SLOTS]


# --------------------------------------------------------------------------- #
# Loading + blinding decode
# --------------------------------------------------------------------------- #
def load_results(path: str, include_rejected: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    n0 = len(df)
    if not include_rejected:
        # exclude rejected answers: rejected in MTurk OR submitted without a reason
        has_reason = df["Answer.reason_A"].fillna("").str.strip() != ""
        n_rej = (df["AssignmentStatus"] == "Rejected").sum()
        n_noreason = (~has_reason & (df["AssignmentStatus"] != "Rejected")).sum()
        df = df[(df["AssignmentStatus"] != "Rejected") & has_reason].copy()
        print(f"Loaded {n0} assignments ({n_rej} rejected + {n_noreason} without a "
              f"reason dropped, {len(df)} kept) | {df['HITId'].nunique()} HITs "
              f"| {df['WorkerId'].nunique()} workers")
    else:
        print(f"Loaded {n0} assignments (kept all) | {df['HITId'].nunique()} HITs "
              f"| {df['WorkerId'].nunique()} workers")
    for q in QUESTIONS:
        df[f"Answer.{q}"] = pd.to_numeric(df[f"Answer.{q}"], errors="coerce")
    return df


def decode_blinding(df: pd.DataFrame) -> pd.DataFrame:
    """One row per assignment x TRUE ROLE, with the slot it was shown in."""
    rows = []
    for _, r in df.iterrows():
        for slot in SLOTS:
            role = r[f"Input.slot_{slot}_role"]
            rows.append({
                "HITId": r["HITId"],
                "AssignmentId": r["AssignmentId"],
                "WorkerId": r["WorkerId"],
                "source": r["Input.source"],
                "summary_id": r["Input.summary_id"],
                "base_A_id": r["Input.base_A_id"],
                "candidate_id": r["Input.candidate_id"],
                "candidate_distinct": r["Input.candidate_distinct"],
                "role": role,
                "slot": slot,
                "faith": r[f"Answer.faith_{slot}"],
                "info": r[f"Answer.info_{slot}"],
                # the reason textarea only exists under display slot A
                "reason": r.get("Answer.reason_A") if slot == "A" else None,
                "work_seconds": r["WorkTimeInSeconds"],
            })
    long = pd.DataFrame(rows)
    assert (long.groupby("AssignmentId")["role"].nunique() == 3).all(), \
        "an assignment is missing a role -- slot_*_role columns look corrupted"
    return long


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Assignment-level quality flags (reported; dropped only under --strict)."""
    ans = df[[f"Answer.{q}" for q in QUESTIONS]]
    flags = pd.DataFrame(index=df.index)
    flags["incomplete"] = ans.isna().any(axis=1)
    # straightliner: same value on every question they did answer
    flags["straightliner"] = ans.nunique(axis=1, dropna=True) <= 1
    flags["too_fast"] = df["WorkTimeInSeconds"] < MIN_WORK_SECONDS
    flags["flagged"] = flags.any(axis=1)
    out = df.copy()
    out[flags.columns] = flags
    print(f"\nQuality flags: {flags['incomplete'].sum()} incomplete, "
          f"{flags['straightliner'].sum()} straightliners, "
          f"{flags['too_fast'].sum()} faster than {MIN_WORK_SECONDS}s "
          f"-> {flags['flagged'].sum()}/{len(flags)} assignments flagged")
    return out


# --------------------------------------------------------------------------- #
# LLM scores from the hierarchy JSONs
# --------------------------------------------------------------------------- #
def llm_scores_for_items(items: pd.DataFrame, trees_dir: str) -> pd.DataFrame:
    """Join LLM judge scores onto (source, summary_id) items.

    summary node  : faithfulness1/2, informativeness1/2, entail_1/2
                    (faith/info are propagated up from the leaves, so the
                     level-1 node holds the child-vs-summary scores)
    owner leaf    : specificity, specificity_faithfulness/_informativeness
                    (NOT propagated; taken from whichever child supplied the
                     candidate, matching candidate_third() in the extractor)
    """
    recs = []
    for source, grp in items.groupby("source"):
        path = Path(trees_dir) / f"{source}.json"
        if not path.exists():
            print(f"  ! tree file missing, LLM join skipped for: {path}")
            continue
        tree = json.load(open(path, encoding="utf-8"))
        levels = tree["levels"]
        level0 = levels[0]["nodes"]
        id2node = {n["id"]: n for lvl in levels for n in lvl["nodes"]}
        for _, it in grp.iterrows():
            summ = id2node.get(it["summary_id"], {})
            owner = id2node.get(it["base_A_id"], {})
            partner_id = next((c for c in summ.get("children", [])
                               if c != it["base_A_id"]), None)
            partner = id2node.get(partner_id, {})
            # the child whose `third` points at the candidate holds the
            # specificity-context scores
            spec_src = {}
            for child in (owner, partner):
                ti = child.get("third")
                if ti is not None and 0 <= ti < len(level0) \
                        and level0[ti]["id"] == it["candidate_id"]:
                    spec_src = child
                    break
            recs.append({
                "source": source, "summary_id": it["summary_id"],
                "llm_entail_1": summ.get("entail_1"),
                "llm_entail_2": summ.get("entail_2"),
                "llm_faithfulness1": summ.get("faithfulness1"),
                "llm_faithfulness2": summ.get("faithfulness2"),
                "llm_informativeness1": summ.get("informativeness1"),
                "llm_informativeness2": summ.get("informativeness2"),
                "llm_specificity": spec_src.get("specificity"),
                "llm_spec_faithfulness": spec_src.get("specificity_faithfulness"),
                "llm_spec_informativeness": spec_src.get("specificity_informativeness"),
            })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def per_item_table(long: pd.DataFrame, trees_dir: str) -> pd.DataFrame:
    """One row per item: human means per role/dimension + composites + LLM."""
    wide = long.pivot_table(index=["source", "summary_id", "base_A_id",
                                   "candidate_id", "AssignmentId"],
                            columns="role", values=["faith", "info"])
    wide.columns = [f"{d}_{r}" for d, r in wide.columns]
    wide = wide.reset_index()

    # per-assignment composites, mirroring how the LLM scores are built:
    # *_1 = avg over the two real children, *_2 = min (entail_1/entail_2 style)
    wide["human_faith1"] = wide[["faith_base_A", "faith_base_B"]].mean(axis=1)
    wide["human_faith2"] = wide[["faith_base_A", "faith_base_B"]].min(axis=1)
    wide["human_info1"] = wide[["info_base_A", "info_base_B"]].mean(axis=1)
    wide["human_info2"] = wide[["info_base_A", "info_base_B"]].min(axis=1)
    # specificity context = owner child + candidate (avg, no min -- like the pipeline)
    wide["human_spec_faith"] = wide[["faith_base_A", "faith_candidate"]].mean(axis=1)
    wide["human_spec_info"] = wide[["info_base_A", "info_candidate"]].mean(axis=1)

    keys = ["source", "summary_id", "base_A_id", "candidate_id"]
    value_cols = [c for c in wide.columns if c not in keys + ["AssignmentId"]]
    agg = wide.groupby(keys).agg(
        n_workers=("AssignmentId", "nunique"),
        **{c: (c, "mean") for c in value_cols},
        **{c + "_std": (c, "std") for c in value_cols},
    ).reset_index()

    llm = llm_scores_for_items(agg[keys].drop_duplicates(), trees_dir)
    if not llm.empty:
        agg = agg.merge(llm, on=["source", "summary_id"], how="left")
    return agg


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def report_role_comparison(long: pd.DataFrame, items: pd.DataFrame) -> None:
    """Summarize the human entailment (faithfulness) scores of A, B, C.

    A = base_A (owner child), B = base_B (partner child), C = candidate
    (third-NN tested for reassignment). Faithfulness is the entailment
    judgment: does the base prompt imply the general prompt's details?
    Paired tests use item-level means so each item counts once.
    """
    print("\n=== Entailment (faithfulness) comparison: A vs B vs C ===")
    label = {"base_A": "A (child)", "base_B": "B (child)", "candidate": "C (candidate)"}
    for dim, name in (("faith", "faithfulness/entailment"), ("info", "informativeness")):
        print(f"-- {name} --")
        g = long.groupby("role")[dim].agg(["count", "mean", "std", "median"])
        for role in ROLES:
            r = g.loc[role]
            print(f"  {label[role]:>14}: mean {r['mean']:.3f}  std {r['std']:.3f}  "
                  f"median {r['median']:.2f}  (n={int(r['count'])} answers)")
        # paired comparisons on item-level means
        cols = {"base_A": f"{dim}_base_A", "base_B": f"{dim}_base_B",
                "candidate": f"{dim}_candidate"}
        for r1, r2 in (("base_A", "base_B"), ("base_A", "candidate"),
                       ("base_B", "candidate")):
            sub = items[[cols[r1], cols[r2]]].dropna()
            diff = sub[cols[r1]] - sub[cols[r2]]
            if len(sub) < 3 or (diff == 0).all():
                print(f"  {label[r1]} vs {label[r2]}: too few/constant data")
                continue
            w = stats.wilcoxon(sub[cols[r1]], sub[cols[r2]])
            print(f"  {label[r1]} vs {label[r2]}: mean diff {diff.mean():+.3f} "
                  f"(Wilcoxon p={w.pvalue:.4f}{'*' if w.pvalue < 0.05 else ''}, "
                  f"n={len(sub)} items)")
    print("  Reading: A and B are the summary's real children; C is the outside")
    print("  candidate. If the hierarchy is sound, A and B should score high and")
    print("  similar, C clearly lower -- unless C truly belongs under the summary.")


def report_correlations(items: pd.DataFrame) -> None:
    pairs = [
        ("human_faith1", "llm_faithfulness1"),
        ("human_faith2", "llm_faithfulness2"),
        ("human_info1", "llm_informativeness1"),
        ("human_info2", "llm_informativeness2"),
        ("human_faith1", "llm_entail_1"),
        ("human_faith2", "llm_entail_2"),
        ("human_spec_faith", "llm_spec_faithfulness"),
        ("human_spec_info", "llm_spec_informativeness"),
        ("human_spec_faith", "llm_specificity"),
        ("faith_candidate", "llm_specificity"),
    ]
    print("\n=== Human vs LLM (item-level, human = mean over workers) ===")
    print(f"{'human':>18} {'llm':>26} {'n':>4} {'pearson':>9} {'spearman':>9} {'MAE':>7}")
    for h, l in pairs:
        if h not in items or l not in items:
            continue
        sub = items[[h, l]].dropna()
        if len(sub) < 3:
            print(f"{h:>18} {l:>26} {len(sub):>4}   (too few points)")
            continue
        pr, pp = stats.pearsonr(sub[h], sub[l])
        sr, sp = stats.spearmanr(sub[h], sub[l])
        mae = (sub[h] - sub[l]).abs().mean()
        star = lambda p: "*" if p < 0.05 else " "
        print(f"{h:>18} {l:>26} {len(sub):>4} {pr:>8.3f}{star(pp)} {sr:>8.3f}{star(sp)} {mae:>7.3f}")
    print("  (* = p < 0.05)")


def report_position_bias(long: pd.DataFrame) -> None:
    print("\n=== Blinding / position-bias checks ===")
    for dim in ("faith", "info"):
        by_slot = long.groupby("slot")[dim].mean().reindex(list(SLOTS))
        by_role = long.groupby("role")[dim].mean().reindex(list(ROLES))
        print(f"{dim:>6} | mean by SLOT  A: {by_slot['A']:.3f}  B: {by_slot['B']:.3f}  "
              f"third: {by_slot['third']:.3f}")
        print(f"{'':>6} | mean by ROLE  base_A: {by_role['base_A']:.3f}  "
              f"base_B: {by_role['base_B']:.3f}  candidate: {by_role['candidate']:.3f}")
    cand = long[long["role"] == "candidate"]
    by_slot = cand.groupby("slot")["faith"].agg(["mean", "count"]).reindex(list(SLOTS))
    print("candidate faith by the slot it landed in "
          "(should be ~equal if workers are truly blind):")
    for slot, row in by_slot.iterrows():
        print(f"    slot {slot:>5}: mean {row['mean']:.3f}  (n={int(row['count'])})")


def krippendorff_alpha_interval(values_by_unit: list) -> float:
    """Krippendorff's alpha for interval data. Units with <2 values are skipped."""
    units = [np.asarray([v for v in u if not pd.isna(v)], dtype=float)
             for u in values_by_unit]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    n = sum(len(u) for u in units)
    d_o = sum(2 * sum((a - b) ** 2 for a, b in combinations(u, 2)) / (len(u) - 1)
              for u in units) / n
    allv = np.concatenate(units)
    d_e = 2 * sum((a - b) ** 2 for a, b in combinations(allv, 2)) / (n * (n - 1))
    return 1 - d_o / d_e if d_e else float("nan")


def report_agreement(long: pd.DataFrame) -> None:
    print("\n=== Inter-rater agreement (HITs with >= 2 assignments) ===")
    for dim in ("faith", "info"):
        units, diffs = [], []
        for _, grp in long.groupby(["HITId", "role"]):
            vals = grp[dim].dropna().values
            units.append(vals)
            diffs += [abs(a - b) for a, b in combinations(vals, 2)]
        alpha = krippendorff_alpha_interval(units)
        mad = np.mean(diffs) if diffs else float("nan")
        print(f"{dim:>6} | Krippendorff alpha (interval): {alpha:.3f} "
              f"| mean pairwise |diff|: {mad:.3f} (over {len(diffs)} pairs)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=[DEFAULT_RESULTS],
                    help="MTurk batch results CSV(s); multiple files are pooled by item")
    ap.add_argument("--exclude-workers", nargs="*", default=[],
                    help="worker ids to drop before any statistics")
    ap.add_argument("--trees-dir", default=DEFAULT_TREES_DIR,
                    help="directory holding the hierarchy JSONs named <source>.json")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--strict", action="store_true",
                    help="exclude quality-flagged assignments from all statistics")
    ap.add_argument("--include-rejected", action="store_true")
    args = ap.parse_args()

    df = pd.concat([load_results(p, args.include_rejected) for p in args.results],
                   ignore_index=True)
    if args.exclude_workers:
        n0 = len(df)
        df = df[~df["WorkerId"].isin(args.exclude_workers)]
        print(f"Excluded {len(args.exclude_workers)} workers: "
              f"{n0 - len(df)} assignments dropped, {len(df)} remain")
    df = flag_quality(df)
    if args.strict:
        df = df[~df["flagged"]].copy()
        print(f"--strict: {len(df)} assignments remain")

    long = decode_blinding(df)
    items = per_item_table(long, args.trees_dir)

    report_role_comparison(long, items)
    report_correlations(items)
    report_position_bias(long)
    report_agreement(long)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "per_assignment_long.csv", index=False)
    items.to_csv(out / "per_item_summary.csv", index=False)
    print(f"\nWrote {out / 'per_assignment_long.csv'} ({len(long)} rows) and "
          f"{out / 'per_item_summary.csv'} ({len(items)} rows)")


if __name__ == "__main__":
    main()
