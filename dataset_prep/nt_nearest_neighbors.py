#!/usr/bin/env python3
"""
Build a WildBench-style nearest-neighbor sheet for UltraFeedback, for manual
subset curation (pick a seed prompt -> take its 64 neighbors).

Mirrors the WB artifact (wb_nearest_neighbors_64.csv: Question, Result, Score) so the
same manual workflow applies:

  1. Pool: eligible UF prompts (garbled-char clean, 200-4000 chars, deduped --
     same rules as prepare_ultrafeedback.py), read from dataset_UF_with_scores.csv.
  2. Embeddings: all-mpnet-base-v2, L2-normalized, cached to
     outputs/uf_mpnet_embeddings.pkl (first run encodes ~35k prompts).
  3. Queries: a seeded random sample of --n-queries prompts (default 1024,
     matching the WB sheet); neighbors are retrieved from the FULL eligible
     pool, not just the queries.
  4. Output: uf_nearest_neighbors.csv with one row per (query, neighbor),
     top --k neighbors each (default 64), exact cosine (no FAISS approximation).

Also supports a targeted lookup once you've spotted a seed you like:

    # 64 neighbors of the best-matching prompt, written to an xlsx
    python dataset_prep/uf_nearest_neighbors.py --seed-text "pymavlink"

Usage:
    conda run -n EntailmentHierarchy python dataset_prep/uf_nearest_neighbors.py
    ... [--n-queries 1024] [--k 64] [--seed 42] [--out uf_nearest_neighbors.csv]
    ... [--seed-text SUBSTRING]   # lookup mode instead of the full sheet
"""
import argparse
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_ultrafeedback import is_clean

EMB_CACHE = "outputs/nt_mpnet_embeddings.pkl"
# SCORES_CSV = "dataset_UF_with_scores.csv"
SCORES_CSV = "dataset_NT_with_scores_so.csv"


def eligible_prompts(min_chars=200, max_chars=4000) -> list[str]:
    df = pd.read_csv(SCORES_CSV)
    texts = df["instructions"].dropna().astype(str)
    keep = texts[texts.str.len().between(min_chars, max_chars) & texts.map(is_clean)]
    pool = list(dict.fromkeys(keep))          # dedup, stable order
    print(f"[INFO] {len(pool)} eligible prompts "
          f"(clean chars, {min_chars}-{max_chars} chars, deduped)")
    return pool


def load_or_build_embeddings(pool: list[str]) -> np.ndarray:
    cache = Path(EMB_CACHE)
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if data["sentences"] == pool:
            print(f"[INFO] Loaded cached embeddings {data['embeddings'].shape}")
            return data["embeddings"]
        print("[INFO] Cache exists but pool changed -- re-encoding")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    print(f"[INFO] Encoding {len(pool)} prompts with all-mpnet-base-v2 ...")
    emb = model.encode(pool, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"sentences": pool, "embeddings": emb}, f)
    print(f"[INFO] Cached embeddings -> {cache}")
    return emb


def top_k(emb: np.ndarray, query_idx: list[int], k: int):
    """Exact cosine top-k (excluding self) for the given query indices."""
    out = []
    for start in range(0, len(query_idx), 256):
        chunk = query_idx[start:start + 256]
        sims = emb[chunk] @ emb.T                     # (chunk, pool)
        for row, qi in enumerate(chunk):
            s = sims[row]
            nn = np.argpartition(-s, k + 1)[: k + 1]
            nn = nn[np.argsort(-s[nn])]
            nn = [j for j in nn if j != qi][:k]
            out.append((qi, nn, s[nn]))
    return out


def flatten_ws(s: str) -> str:
    """One line per cell so the sheet is scrollable in Excel."""
    return re.sub(r"\s+", " ", s or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-queries", type=int, default=1024,
                    help="how many prompts get a neighbor list (default 1024, like WB)")
    ap.add_argument("--k", type=int, default=64, help="neighbors per query")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="nt_nearest_neighbors.csv")
    ap.add_argument("--seed-text", default=None,
                    help="lookup mode: substring of a prompt; writes that prompt's "
                         "k neighbors to nt_neighbors_of_<idx>.xlsx")
    args = ap.parse_args()

    pool = eligible_prompts()
    emb = load_or_build_embeddings(pool)

    if args.seed_text:
        matches = [i for i, p in enumerate(pool)
                   if args.seed_text.lower() in p.lower()]
        if not matches:
            sys.exit(f"[ERROR] no eligible prompt contains {args.seed_text!r}")
        qi = matches[0]
        print(f"[INFO] {len(matches)} match(es); using pool index {qi}:")
        print("       " + pool[qi][:160].replace("\n", " "))
        (_, nn, scores), = top_k(emb, [qi], args.k)
        out = Path(f"nt_neighbors_of_{qi}.xlsx")
        pd.DataFrame({
            "Result": [flatten_ws(pool[qi])] + [flatten_ws(pool[j]) for j in nn],
            "Score": [1.0] + [float(s) for s in scores],
        }).to_excel(out, index=False)
        print(f"[INFO] Wrote seed + {len(nn)} neighbors -> {out}")
        return

    rng = random.Random(args.seed)
    queries = sorted(rng.sample(range(len(pool)), min(args.n_queries, len(pool))))
    print(f"[INFO] Computing top-{args.k} for {len(queries)} queries "
          f"over the {len(pool)}-prompt pool (exact cosine) ...")
    rows = []
    for qi, nn, scores in top_k(emb, queries, args.k):
        q = flatten_ws(pool[qi])
        for j, s in zip(nn, scores):
            rows.append({"Question": q, "Result": flatten_ws(pool[j]),
                         "Score": round(float(s), 4)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[INFO] Wrote {len(rows)} rows ({len(queries)} queries x {args.k} "
          f"neighbors) -> {args.out}")


if __name__ == "__main__":
    main()
