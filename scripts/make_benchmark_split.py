"""Build the public, reproducible train/validation/test split for ViHoRec.

Protocol (temporal leave-one-out, three-way):

  * keep only users with at least ``MIN_INTERACTIONS`` ratings (default 4);
  * remap ``user_id`` / ``hotel_id`` to contiguous integers;
  * per user, in chronological order: the **last** interaction is test, the
    **second-last** is validation, and everything before is train.

The validation fold is the point of this script. The submitted version split
only into train and test, so every hyper-parameter — the hybrid's lambda, the
adaptive rule's (alpha_min, n0, tau), and the neural models' settings — was
selected on the same interactions used to report the result. Publishing
``val.csv`` makes tuning-without-leakage the default rather than a promise.

With min-k = 4, removing two interactions per user still leaves at least two in
train, so no user becomes untrainable.

**Relevance.** The task is implicit next-item ranking: for a user, the single
relevant item is the hotel they reviewed next, regardless of the score they
gave it. Ratings are kept in the files for analysis but are not thresholded, so
a 3/10 review counts as relevant exactly like a 10/10 one — the signal is the
visit, not the sentiment. ``--relevance-threshold`` restricts the evaluation
folds to ratings at or above a cut-off for the sensitivity analysis the paper
reports; it never changes the training fold.

Outputs (release/benchmark/):
  train.csv, val.csv, test.csv  -> columns: userID,itemID,rating,timestamp
  user_map.csv, item_map.csv, split_config.json

Run:  python make_benchmark_split.py [--relevance-threshold 7.0]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import config as C

MIN_INTERACTIONS = 4
# Interactions held out per user for evaluation: the last two in time.
N_HOLDOUT = 2
COLS = ["userID", "itemID", "rating", "timestamp"]
OUT = C.OUT_RELEASE / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)


def epoch_seconds(dates: pd.Series) -> pd.Series:
    """Unix seconds from a date column, independent of pandas' unit default.

    ``to_datetime(...).astype("int64") // 10**9`` is wrong on pandas >= 3, which
    parses to ``datetime64[us]``: dividing microseconds by 1e9 yields values
    1000x too small, so every released timestamp landed in January 1970. The
    split itself survived because the error is monotonic, but the published
    column was unusable for any time-aware model.
    """
    return pd.to_datetime(dates).astype("datetime64[s]").astype("int64")


def _fold_stats(fold: pd.DataFrame) -> dict:
    if fold.empty:
        return {"n": 0}
    ts = pd.to_datetime(fold["timestamp"], unit="s")
    return {
        "n": int(len(fold)),
        "n_users": int(fold["userID"].nunique()),
        "n_items": int(fold["itemID"].nunique()),
        "date_min": str(ts.min().date()),
        "date_max": str(ts.max().date()),
        "rating_mean": round(float(fold["rating"].mean()), 4),
    }


def run(relevance_threshold: float | None = None) -> dict:
    df = pd.read_csv(C.OUT_RELEASE / "interactions.csv")
    df["timestamp"] = epoch_seconds(df["date"])

    counts = df.groupby("user_id").size()
    keep = counts[counts >= MIN_INTERACTIONS].index
    df = df[df["user_id"].isin(keep)].copy()

    user_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item_map = {h: i for i, h in enumerate(sorted(df["hotel_id"].unique()))}
    df["userID"] = df["user_id"].map(user_map)
    df["itemID"] = df["hotel_id"].map(item_map)

    # Stable ordering so "last" and "second-last" are deterministic under ties.
    df = df.sort_values(["userID", "timestamp", "itemID"]).reset_index(drop=True)
    rank_from_end = df.groupby("userID").cumcount(ascending=False)

    test = df[rank_from_end == 0]
    val = df[rank_from_end == 1]
    train = df[rank_from_end >= N_HOLDOUT]

    # Sensitivity variant: keep only well-rated holdouts as relevant. Train is
    # untouched, so the two settings differ in what counts as a hit, not in what
    # the models were allowed to learn from. It is written to its own directory
    # so the canonical release files are never overwritten by an ablation run.
    if relevance_threshold is not None:
        test = test[test["rating"] >= relevance_threshold]
        val = val[val["rating"] >= relevance_threshold]
        out_dir = OUT / f"relevance_ge_{relevance_threshold:g}"
    else:
        out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fold in (("train", train), ("val", val), ("test", test)):
        fold[COLS].to_csv(out_dir / f"{name}.csv", index=False)
    pd.Series(user_map).rename_axis("user_id").reset_index(name="userID").to_csv(
        out_dir / "user_map.csv", index=False
    )
    pd.Series(item_map).rename_axis("hotel_id").reset_index(name="itemID").to_csv(
        out_dir / "item_map.csv", index=False
    )

    # Users whose train history is short enough to be the hard evaluation cases.
    train_hist = train.groupby("userID").size()
    cfg = {
        "protocol": "temporal leave-one-out, three-way (train / val / test)",
        "fold_assignment": {
            "test": "last interaction per user",
            "val": "second-last interaction per user",
            "train": "all earlier interactions",
        },
        "temporal_scope": {
            "ordering": "per user, not global",
            "note": (
                "Each user's own history is split chronologically, so no user's "
                "future leaks into their training data. Across users the folds "
                "overlap in wall-clock time: one user's training review can be "
                "later than another user's test review. This is the standard "
                "leave-last-one-out protocol; a single global cut-off date would "
                "be a different and stricter setting, and is not used here."
            ),
        },
        "relevance": {
            "definition": "implicit next-item: the chronologically next hotel the user reviewed",
            "all_ratings_are_positive": relevance_threshold is None,
            "relevance_threshold": relevance_threshold,
            "note": (
                "Ratings are not thresholded by default; the holdout item is "
                "relevant because it was visited, not because it scored well."
            ),
        },
        "min_interactions": MIN_INTERACTIONS,
        "seed": C.RANDOM_SEED,
        "n_users": len(user_map),
        "n_items": len(item_map),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "train_history_length": {
            "min": int(train_hist.min()),
            "median": float(train_hist.median()),
            "max": int(train_hist.max()),
        },
        "users_with_train_history_2": int((train_hist == 2).sum()),
        "users_evaluated": {
            "val": int(val["userID"].nunique()),
            "test": int(test["userID"].nunique()),
        },
        "sparsity_pct": round(
            100.0 * (1 - len(df) / (len(user_map) * len(item_map))), 4
        ),
        "output_dir": str(out_dir.relative_to(C.OUT_RELEASE)),
        "folds": {
            "train": _fold_stats(train),
            "val": _fold_stats(val),
            "test": _fold_stats(test),
        },
    }
    (out_dir / "split_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--relevance-threshold",
        type=float,
        default=None,
        help="Keep only val/test items rated >= this value (sensitivity analysis)",
    )
    args = ap.parse_args()
    run(args.relevance_threshold)
