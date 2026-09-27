"""Build a human-annotatable candidate set for hotel entity resolution (Issue I2).

The release merges hotel names by ``canonical_hotel_key``: accent-free,
lower-cased, hotel stopwords removed, tokens sorted and de-duplicated. The
quality report shows how many names that rule collapses, but a collapse count
says nothing about whether the merges are *correct* or whether real duplicates
were missed. Evaluating that needs pairwise human labels.

This script produces the candidate set. All name pairs are scored with two
cheap, algorithm-independent similarities (token Jaccard and character
sequence ratio) and sampled into three strata:

* ``algo_merge`` -- every pair merged by *either* matcher: the original
  name-only canonical key, or the city- and type-aware resolver that replaced
  it. Labelling all of them exhaustively lets the same gold labels score both,
  so the revision can report the improvement rather than assert it.
* ``top_similar`` -- the most similar unmerged pairs, taken exhaustively. These
  are where missed duplicates concentrate.
* ``hard_negative`` -- a *uniform random* sample of the remaining high-similarity
  pairs. Randomness matters: these labels are extrapolated to the whole stratum
  when recall is estimated, which is only valid for an unbiased sample.
* ``random_negative`` -- a uniform random sample of the low-similarity
  remainder, so the estimate is not built on hard cases alone.

Each stratum is either exhaustive or uniformly sampled, so ``er_evaluate.py``
can reweight labels back to the full pair space without bias.

Two files are written. The annotation sheet is shuffled and carries no
algorithm decision or similarity score, so annotators cannot anchor on what the
pipeline did; the key file keeps that information for scoring.

Run:  python er_candidate_pairs.py [--n-hard 250] [--n-random 150]
Outputs: annotation/er_pairs_to_annotate.csv, annotation/er_pairs_key.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import config as C
from textnorm import canonical_hotel_key, canonical_location, strip_accents

DEFAULT_N_TOP = 100
DEFAULT_N_HARD = 150
DEFAULT_N_RANDOM = 150
# A pair is "hard" when either similarity is high enough that a reviewer would
# expect entity resolution to have considered it.
HARD_JACCARD = 0.34
HARD_SEQRATIO = 0.72


def _name_tokens(name: str) -> frozenset[str]:
    """Discriminative tokens of a hotel name (no stopword removal).

    Stopwords are deliberately kept here: the canonical key drops them, so a
    similarity that also dropped them would inherit the same blind spot.
    """
    base = strip_accents(str(name)).lower()
    base = "".join(ch if ch.isalnum() else " " for ch in base)
    return frozenset(t for t in base.split() if t)


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, strip_accents(a).lower(), strip_accents(b).lower()).ratio()


def load_name_catalogue() -> pd.DataFrame:
    """One row per distinct (hotel name, city) listing.

    Keys are taken from ``interactions_clean.csv`` rather than recomputed, so
    the candidate set always reflects the matcher the pipeline actually ran.
    """
    src = C.OUT_REPORTS / "interactions_clean.csv"
    if not src.exists():
        raise SystemExit("Run quality_control.py first to produce interactions_clean.csv")
    df = pd.read_csv(src)
    cat = (
        df.groupby(["NameHotel", "Location"])
        .agg(
            n_interactions=("NameHotel", "size"),
            platforms=("source", lambda s: " | ".join(sorted(set(s.astype(str))))),
            key_resolved=("hotel_key", "first"),
            key_nameonly=("hotel_key_nameonly", "first"),
            loc_key=("loc_key", "first"),
        )
        .reset_index()
        .rename(columns={"Location": "city"})
    )
    cat["tokens"] = cat["NameHotel"].map(_name_tokens)
    return cat


def score_all_pairs(cat: pd.DataFrame) -> pd.DataFrame:
    """Exhaustive pairwise scoring. ~580 names => ~170k pairs, cheap enough."""
    recs = cat.to_dict("records")
    rows = []
    for a, b in itertools.combinations(recs, 2):
        ta, tb = a["tokens"], b["tokens"]
        union = ta | tb
        if not union:
            continue
        jac = len(ta & tb) / len(union)
        # Character similarity only matters when tokens already overlap or the
        # strings are close; skip the rest to keep the O(n^2) loop fast.
        seq = _seq_ratio(a["NameHotel"], b["NameHotel"]) if jac > 0 else 0.0
        merge_resolved = a["key_resolved"] == b["key_resolved"]
        merge_nameonly = a["key_nameonly"] == b["key_nameonly"]
        rows.append(
            {
                "name_a": a["NameHotel"],
                "name_b": b["NameHotel"],
                "city_a": a["city"],
                "city_b": b["city"],
                "platforms_a": a["platforms"],
                "platforms_b": b["platforms"],
                "n_inter_a": a["n_interactions"],
                "n_inter_b": b["n_interactions"],
                "same_city": a["loc_key"] == b["loc_key"],
                "jaccard": round(jac, 4),
                "seq_ratio": round(seq, 4),
                "merge_nameonly": merge_nameonly,
                "merge_resolved": merge_resolved,
                "algo_merge": merge_nameonly or merge_resolved,
            }
        )
    return pd.DataFrame(rows)


def _uniform_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    take = min(n, len(pool))
    if take == 0:
        return pool.iloc[0:0].copy()
    idx = rng.choice(len(pool), size=take, replace=False)
    return pool.iloc[np.sort(idx)].copy()


def sample_strata(
    pairs: pd.DataFrame, n_top: int, n_hard: int, n_random: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Draw the four strata and return them with their population sizes.

    Every stratum is either exhaustive (``algo_merge``, ``top_similar``) or a
    uniform random draw (``hard_negative``, ``random_negative``). Taking the
    most similar pairs *and* reweighting them as a random sample would inflate
    the estimated number of missed duplicates, so the deterministic top slice
    is carved out into its own weight-1 stratum first.
    """
    rng = np.random.default_rng(seed)

    merges = pairs[pairs["algo_merge"]].copy()
    merges["stratum"] = "algo_merge"

    rest = pairs[~pairs["algo_merge"]]
    hard_mask = (rest["jaccard"] >= HARD_JACCARD) | (rest["seq_ratio"] >= HARD_SEQRATIO)
    hard_pool = rest[hard_mask].sort_values(["jaccard", "seq_ratio"], ascending=False)

    top = hard_pool.head(n_top).copy()
    top["stratum"] = "top_similar"

    remaining_hard = hard_pool.iloc[len(top):]
    hard = _uniform_sample(remaining_hard, n_hard, rng)
    hard["stratum"] = "hard_negative"

    easy_pool = rest[~hard_mask]
    easy = _uniform_sample(easy_pool, n_random, rng)
    easy["stratum"] = "random_negative"

    pools = {
        "algo_merge": {
            "pool_size": int(len(merges)), "n_sampled": int(len(merges)),
            "design": "exhaustive",
        },
        "top_similar": {
            "pool_size": int(len(top)), "n_sampled": int(len(top)),
            "design": "exhaustive (highest-similarity unmerged pairs)",
        },
        "hard_negative": {
            "pool_size": int(len(remaining_hard)), "n_sampled": int(len(hard)),
            "design": "uniform random",
        },
        "random_negative": {
            "pool_size": int(len(easy_pool)), "n_sampled": int(len(easy)),
            "design": "uniform random",
        },
    }

    out = pd.concat([merges, top, hard, easy], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.insert(0, "pair_id", [f"P{i:04d}" for i in range(len(out))])
    return out, pools


def write_outputs(sample: pd.DataFrame) -> dict:
    sheet_cols = [
        "pair_id",
        "name_a", "city_a", "platforms_a",
        "name_b", "city_b", "platforms_b",
    ]
    sheet = sample[sheet_cols].copy()
    # Blank columns the annotators fill in: 1 = same hotel, 0 = different.
    for i in range(1, C.N_ANNOTATORS + 1):
        sheet[f"annotator_{i}"] = ""
    sheet["adjudicated"] = ""
    sheet["notes"] = ""

    key = sample[
        [
            "pair_id", "stratum", "algo_merge", "merge_nameonly", "merge_resolved",
            "jaccard", "seq_ratio", "same_city", "n_inter_a", "n_inter_b",
        ]
    ].copy()

    sheet_path = C.OUT_ANNOTATION / "er_pairs_to_annotate.csv"
    key_path = C.OUT_ANNOTATION / "er_pairs_key.csv"
    sheet.to_csv(sheet_path, index=False, encoding="utf-8")
    key.to_csv(key_path, index=False, encoding="utf-8")

    counts = sample["stratum"].value_counts().to_dict()
    return {
        "sheet": str(sheet_path),
        "key": str(key_path),
        "n_pairs": int(len(sample)),
        "by_stratum": {k: int(v) for k, v in counts.items()},
        "merges_by_matcher": {
            "name_only": int(sample["merge_nameonly"].sum()),
            "resolved": int(sample["merge_resolved"].sum()),
            "disputed": int((sample["merge_nameonly"] != sample["merge_resolved"]).sum()),
        },
        "cross_platform_pairs": int(
            (sample["platforms_a"] != sample["platforms_b"]).sum()
        ),
        "same_city_pairs": int(sample["same_city"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-top", type=int, default=DEFAULT_N_TOP)
    ap.add_argument("--n-hard", type=int, default=DEFAULT_N_HARD)
    ap.add_argument("--n-random", type=int, default=DEFAULT_N_RANDOM)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    cat = load_name_catalogue()
    pairs = score_all_pairs(cat)
    sample, pools = sample_strata(
        pairs, args.n_top, args.n_hard, args.n_random, args.seed
    )
    summary = write_outputs(sample)

    summary["strata_pools"] = pools
    summary["catalogue"] = {
        "distinct_listings": int(len(cat)),
        "distinct_raw_names": int(cat["NameHotel"].nunique()),
        "entities_resolved_matcher": int(cat["key_resolved"].nunique()),
        "entities_name_only_matcher": int(cat["key_nameonly"].nunique()),
        "total_pairs_scored": int(len(pairs)),
        "hard_thresholds": {"jaccard": HARD_JACCARD, "seq_ratio": HARD_SEQRATIO},
        "seed": args.seed,
    }
    (C.OUT_ANNOTATION / "er_pairs_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
