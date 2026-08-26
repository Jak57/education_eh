#!/usr/bin/env python3
"""
Extract MTurk input data from hierarchy JSON files for the faithfulness /
informativeness scoring template (general_prompt_scoring_faith_info.html).

For each chosen *level-1 node* (a common summary of two leaf/level-0 prompts) we
emit one row with the four ${...} variables used by the template:

    general_prompt   -> the common summary (level-1 node's `prompt`)
    base_prompt_A    -> SHUFFLED: one of {child A, child B, candidate}
    base_prompt_B    -> SHUFFLED: one of {child A, child B, candidate}
    third_prompt     -> SHUFFLED: one of {child A, child B, candidate}

Blinding / shuffle (new):
  The three prompts -- the summary's two real children plus the third-nearest-
  neighbour Candidate tested for reassignment -- are randomly assigned to the
  three template slots, so workers cannot infer from position which prompt is
  the outside candidate. Only we (the requesters) can decode the mapping, via
  the slot_*_role columns. MTurk ignores extra input columns for rendering but
  echoes them back as Input.<column> in the batch results file:

      slot_A_role, slot_B_role, slot_third_role   each in {base_A, base_B, candidate}

  Decoding results: the worker's faith_A / info_A / reason_A answers score the
  prompt whose role is slot_A_role (same for _B and _third). All role-level
  metadata columns (base_A_id, base_B_id, candidate_id, llm_* scores) ALWAYS
  refer to the true roles, never to the display slots.

  If a summary has no distinct candidate (degenerate fallback), only the two
  real children are shuffled between slots A and B and the candidate stays in
  the third slot, so an empty/duplicate prompt never lands in a base slot.

Background (how these map to the pipeline):
  * `children` / `parent` are stored as node IDs.
  * `match` / `third`     are stored as INDICES into that level's `nodes` list.
  * A level-1 node's `prompt` is the common summary generated from its two
    level-0 children. children[0] is the "owner" (the node whose summary became
    this prompt), children[1] is its match.
  * specificity = avg(owner->summary, owner.third->summary). `owner.third` (a
    level-0 index) is the third-nearest-neighbour that was tested for
    reassignment as an extra child of this summary. If its specificity passed
    the threshold it would already be in `children`; here we surface it
    regardless so a human can score it independently.
  * If the owner's `third` happens to be one of the two base prompts, we fall
    back to the partner child's `third` so the Candidate stays distinct.

Usage:
    python extract_specificity_mturk_csv.py
    python extract_specificity_mturk_csv.py --n 20 --seed 42 \
        --out faith_info_mturk_data.csv \
        --files Results/FINAL_random_prompts_64_base_patched.json ...
"""
import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path

DEFAULT_FILES = [
    "Results/FINAL_random_prompts_64_base.json",
    "Results/FINAL_code_prompts_64_base.json",
    "Results/FINAL_story_prompts_64_base.json",
]
DEFAULT_N = 20                    # summaries per file
DEFAULT_SEED = 42                 # reproducible sampling AND slot shuffling
DEFAULT_OUT = "faith_info_mturk_data.csv"
PREFER_DISTINCT_CANDIDATE = True  # put summaries with a distinct candidate first

ROLES = ("base_A", "base_B", "candidate")

# Column order: the first four MUST match the ${...} variables in the HTML template.
FIELDS = [
    "general_prompt", "base_prompt_A", "base_prompt_B", "third_prompt",
    # --- blinding key: which true role sits in each display slot -------------
    "slot_A_role", "slot_B_role", "slot_third_role",
    # --- reference columns (ignored by MTurk, useful for analysis) -----------
    # These always describe the TRUE roles, not the display slots.
    "source", "summary_id", "base_A_id", "base_B_id", "candidate_id",
    "candidate_distinct", "candidate_reassigned",
    "llm_entail_1", "llm_entail_2", "llm_specificity", "third_score",
]

# --------------------------------------------------------------------------- #
# Garbled / non-English character detection.
# A summary is dropped if its subtree (or its candidate prompt) contains any
# "bad" character. We flag genuinely problematic characters while KEEPING
# ordinary typography (curly quotes, dashes, bullets, ellipsis) and accented
# Latin / Greek letters (e.g. e-acute, o-umlaut, gamma).
# --------------------------------------------------------------------------- #
_ALLOWED_LETTER_PREFIXES = ("LATIN", "GREEK")  # keep e-acute, o-umlaut, math vars like gamma


def _is_bad_char(c: str) -> bool:
    cp = ord(c)
    if cp < 128:
        return False                              # plain ASCII is always fine
    if 0x1D400 <= cp <= 0x1D7FF:
        return True                               # math alphanumeric symbols (italic/bold letters)
    cat = unicodedata.category(c)
    if cat in ("Mn", "Mc", "Me"):
        return True                               # combining marks (Devanagari signs, overline, ...)
    if cat in ("Cc", "Cf", "Cs", "Co"):
        return True                               # control / format / zero-width / private-use
    if cat[0] == "L":                             # a letter: allow only Latin/Greek scripts
        try:
            name = unicodedata.name(c)
        except ValueError:
            return True
        if not name.startswith(_ALLOWED_LETTER_PREFIXES):
            return True                           # Devanagari, CJK, Arabic, Hebrew, ...
    return False


def bad_chars(s: str) -> set:
    """Return the set of flagged characters found in *s* (empty == clean)."""
    return {c for c in (s or "") if _is_bad_char(c)}


def flatten_ws(s: str) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces.

    MTurk counts physical lines in the input CSV, so a prompt with embedded
    newlines is read as many tasks. Flattening keeps each record on one line.
    """
    return re.sub(r"\s+", " ", s or "").strip()


def subtree_texts(node: dict, id2node: dict) -> list:
    """All prompt strings in the subtree rooted at *node* (node + descendants)."""
    texts, stack, seen = [], [node], set()
    while stack:
        x = stack.pop()
        if x["id"] in seen:
            continue
        seen.add(x["id"])
        texts.append(x.get("prompt") or "")
        for cid in x.get("children", []):
            child = id2node.get(cid)
            if child is not None:
                stack.append(child)
    return texts


def candidate_third(owner, partner, level0_nodes, base_ids):
    """Pick the Candidate Prompt for a summary.

    Prefer a `third` (a level-0 index) that is NOT already one of the two base
    prompts: try the owner first, then the partner. Fall back to the owner's
    `third` even if it coincides. Returns
    (cand_node_or_None, specificity, third_score, is_distinct).
    """
    for src in (owner, partner):
        ti = src.get("third")
        if ti is None or not (0 <= ti < len(level0_nodes)):
            continue
        cand = level0_nodes[ti]
        if cand["id"] in base_ids:        # coincides with A or B -> try the other child
            continue
        return cand, src.get("specificity"), src.get("third_score"), True

    # fallback: owner's third even if it coincides with a base prompt
    ti = owner.get("third")
    if ti is not None and 0 <= ti < len(level0_nodes):
        cand = level0_nodes[ti]
        return cand, owner.get("specificity"), owner.get("third_score"), cand["id"] not in base_ids
    return None, owner.get("specificity"), owner.get("third_score"), False


def rows_for_file(path):
    tree = json.load(open(path, encoding="utf-8"))
    levels = tree["levels"]
    if len(levels) < 2:
        print(f"  ! {path} has no level 1 -- skipping")
        return [], 0
    level0_nodes = levels[0]["nodes"]
    level1_nodes = levels[1]["nodes"]
    id2node = {n["id"]: n for lvl in levels for n in lvl["nodes"]}
    source = Path(path).stem

    rows = []
    dropped = 0
    for n in level1_nodes:
        children = n.get("children", [])
        if len(children) < 2:             # need an A and a B
            continue
        owner = id2node.get(children[0])
        partner = id2node.get(children[1])
        if owner is None or partner is None:
            continue
        base_ids = {owner["id"], partner["id"]}

        cand, spec, third_score, distinct = candidate_third(
            owner, partner, level0_nodes, base_ids
        )
        cand_prompt = (cand.get("prompt") if cand else "") or ""

        # Drop the whole subtree if any node in it (or the candidate) is garbled.
        bad = set()
        for t in subtree_texts(n, id2node) + [cand_prompt]:
            bad |= bad_chars(t)
        if bad:
            dropped += 1
            continue

        rows.append({
            "general_prompt": flatten_ws(n.get("prompt")),
            # role-keyed texts; assign_slots() maps them onto the display slots
            "_text_base_A": flatten_ws(owner.get("prompt")),
            "_text_base_B": flatten_ws(partner.get("prompt")),
            "_text_candidate": flatten_ws(cand_prompt),
            "source": source,
            "summary_id": n["id"],
            "base_A_id": owner["id"],
            "base_B_id": partner["id"],
            "candidate_id": cand["id"] if cand else "",
            "candidate_distinct": distinct,
            "candidate_reassigned": bool(cand and cand["id"] in children),
            # entail_1/2 are propagated up to the summary node (leaves get cleared by
            # propagate_attrs_up), so read them from the level-1 node `n`.
            "llm_entail_1": n.get("entail_1"),
            "llm_entail_2": n.get("entail_2"),
            # specificity is NOT propagated, so the owner/partner that supplied the
            # candidate still holds the matching level-0 specificity score.
            "llm_specificity": spec,
            "third_score": third_score,
        })

    return rows, dropped


def assign_slots(row: dict, rng: random.Random) -> None:
    """Blind the display order: randomly place the true roles into the slots.

    Mutates *row* in place: fills base_prompt_A / base_prompt_B / third_prompt
    from the role-keyed _text_* fields, records the mapping in slot_*_role, and
    removes the temporary _text_* keys.
    """
    if row["_text_candidate"] and row["candidate_distinct"]:
        order = list(ROLES)
        rng.shuffle(order)                # full 3-way shuffle
    else:
        # degenerate candidate (missing or duplicates a base prompt): keep it in
        # the third slot, shuffle only the two real children between A and B
        order = ["base_A", "base_B"]
        rng.shuffle(order)
        order.append("candidate")

    row["base_prompt_A"] = row["_text_" + order[0]]
    row["base_prompt_B"] = row["_text_" + order[1]]
    row["third_prompt"] = row["_text_" + order[2]]
    row["slot_A_role"], row["slot_B_role"], row["slot_third_role"] = order
    for role in ROLES:
        del row["_text_" + role]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                    help="hierarchy JSON files to read")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="number of summaries to take per file")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="RNG seed for sampling and slot shuffling (reproducible)")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="disable slot blinding (roles stay in their named slots)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    all_rows = []
    for path in args.files:
        if not Path(path).exists():
            print(f"  ! missing: {path}")
            continue
        rows, dropped = rows_for_file(path)
        rng.shuffle(rows)                       # random sample within the file
        if PREFER_DISTINCT_CANDIDATE:
            # stable sort keeps the shuffle order within each group, just floats
            # distinct-candidate summaries ahead of any degenerate ones
            rows.sort(key=lambda r: not r["candidate_distinct"])
        kept = rows[: args.n]
        n_distinct = sum(1 for r in kept if r["candidate_distinct"])
        print(f"{Path(path).name}: {len(rows)} clean summaries available "
              f"({dropped} dropped for garbled chars), took {len(kept)} "
              f"({n_distinct} with a distinct candidate)")
        if len(kept) < args.n:
            print(f"  ! only {len(kept)} summaries < requested {args.n}")
        all_rows.extend(kept)

    # Blind the slot order (after sampling, so --n/--seed keep picking the same
    # summaries whether or not shuffling is enabled).
    identity_rng = random.Random(0)
    for row in all_rows:
        if args.no_shuffle:
            row["base_prompt_A"] = row.pop("_text_base_A")
            row["base_prompt_B"] = row.pop("_text_base_B")
            row["third_prompt"] = row.pop("_text_candidate")
            row["slot_A_role"], row["slot_B_role"], row["slot_third_role"] = ROLES
        else:
            assign_slots(row, rng)

    if not args.no_shuffle:
        placement = {r: {"A": 0, "B": 0, "third": 0} for r in ROLES}
        for row in all_rows:
            placement[row["slot_A_role"]]["A"] += 1
            placement[row["slot_B_role"]]["B"] += 1
            placement[row["slot_third_role"]]["third"] += 1
        print("\nSlot placement (blinding sanity check):")
        for role, slots in placement.items():
            print(f"  {role:9} -> slot A: {slots['A']:3}  slot B: {slots['B']:3}  "
                  f"slot third: {slots['third']:3}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
