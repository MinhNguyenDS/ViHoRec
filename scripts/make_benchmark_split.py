"""Build a public, reproducible train/test split for the ViHoRec benchmark.

Protocol (leave-last-one-out, temporal):
  * keep only users with >= MIN_INTERACTIONS ratings (default 4, matching the
    original study which dropped very sparse users);
  * remap user_id / hotel_id to contiguous integers (0..N-1);
  * for every retained user, the chronologically latest interaction goes to the
    test set and the rest to train (a standard, leakage-free RecSys protocol).

Outputs (dataset_release/release/benchmark/):
  train.csv, test.csv  -> columns: userID,itemID,rating,timestamp
  user_map.csv, item_map.csv, split_config.json

Run:  python make_benchmark_split.py
"""

from __future__ import annotations

import json

import pandas as pd

import config as C

MIN_INTERACTIONS = 4
OUT = C.OUT_RELEASE / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)


def run() -> dict:
    df = pd.read_csv(C.OUT_RELEASE / "interactions.csv")
    df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9

    counts = df.groupby("user_id").size()
    keep = counts[counts >= MIN_INTERACTIONS].index
    df = df[df["user_id"].isin(keep)].copy()

    user_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item_map = {h: i for i, h in enumerate(sorted(df["hotel_id"].unique()))}
    df["userID"] = df["user_id"].map(user_map)
    df["itemID"] = df["hotel_id"].map(item_map)

    # Stable ordering so "latest" is deterministic even when timestamps tie.
    df = df.sort_values(["userID", "timestamp", "itemID"]).reset_index(drop=True)
    is_last = df.groupby("userID").cumcount(ascending=False) == 0
    test = df[is_last]
    train = df[~is_last]

    cols = ["userID", "itemID", "rating", "timestamp"]
    train[cols].to_csv(OUT / "train.csv", index=False)
    test[cols].to_csv(OUT / "test.csv", index=False)
    pd.Series(user_map).rename_axis("user_id").reset_index(name="userID").to_csv(
        OUT / "user_map.csv", index=False
    )
    pd.Series(item_map).rename_axis("hotel_id").reset_index(name="itemID").to_csv(
        OUT / "item_map.csv", index=False
    )

    cfg = {
        "protocol": "leave-last-one-out (temporal)",
        "min_interactions": MIN_INTERACTIONS,
        "seed": C.RANDOM_SEED,
        "n_users": len(user_map),
        "n_items": len(item_map),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "sparsity_pct": round(
            100.0 * (1 - len(df) / (len(user_map) * len(item_map))), 4
        ),
    }
    (OUT / "split_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return cfg


if __name__ == "__main__":
    run()
