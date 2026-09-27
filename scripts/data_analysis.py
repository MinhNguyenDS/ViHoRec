"""Dataset characterization and data-centric ablations for ViHoRec.

Produces the evidence a resource-paper reviewer expects beyond a single
benchmark table:

  1. Characterization: scale, sparsity, popularity Gini, long-tail head share,
     single-interaction share, rating statistics  (-> reports/analysis_report.md/json,
     Image/LongTail.png).
  2. Ablation A - entity resolution ON vs OFF: does merging cross-site hotel
     name variants change benchmark metrics? (isolates the paper's core claim).
  3. Ablation B - minimum-interaction threshold sweep: effect on scale/sparsity
     and a personalised baseline.
  4. Ablation C - temporal vs random leave-one-out: shows the temporal protocol
     is the harder, more realistic setting.

Run:  python data_analysis.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import make_benchmark_split as mbs
import plot_style as ps
import run_baselines as rb

IMG_DIR = C.PAPER_IMG_DIR
SRC = C.OUT_REPORTS / "interactions_clean.csv"
MIN_K = 4


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["ts"] = mbs.epoch_seconds(df["Date_parsed"])
    return df


def build_split(df: pd.DataFrame, user_col: str, item_col: str,
                min_k: int = MIN_K, mode: str = "temporal", seed: int = 42):
    counts = df.groupby(user_col).size()
    keep = counts[counts >= min_k].index
    d = df[df[user_col].isin(keep)].copy()
    umap = {u: i for i, u in enumerate(sorted(d[user_col].unique()))}
    imap = {v: i for i, v in enumerate(sorted(d[item_col].unique()))}
    d["userID"] = d[user_col].map(umap)
    d["itemID"] = d[item_col].map(imap)

    if mode == "temporal":
        d = d.sort_values(["userID", "ts", "itemID"])
        is_test = d.groupby("userID").cumcount(ascending=False) == 0
    else:  # random held-out-one
        rng = np.random.default_rng(seed)
        pick = d.groupby("userID").sample(n=1, random_state=seed).index
        is_test = d.index.isin(pick)
    test = d[is_test]
    train = d[~is_test]
    return train, dict(zip(test.userID, test.itemID)), len(umap), len(imap)


def _bench(train, test_items, n_users, n_items) -> dict:
    mat = rb.build_matrix(train, n_users, n_items)
    m = rb.evaluate(rb.score_userknn(mat), mat, test_items)
    return {"Recall@10": m["Recall@10"], "NDCG@10": m["NDCG@10"], "MRR": m["MRR"]}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return round(float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n), 4)


def characterize(df: pd.DataFrame) -> dict:
    per_user = df.groupby("CustomerName").size()
    per_item = df.groupby("hotel_key").size()
    n_u, n_i, n = len(per_user), len(per_item), len(df)
    pop_sorted = per_item.sort_values(ascending=False).to_numpy()
    head = int(np.ceil(0.2 * n_i))
    return {
        "n_users": n_u,
        "n_items": n_i,
        "n_interactions": n,
        "sparsity_pct": round(100 * (1 - n / (n_u * n_i)), 4),
        "interactions_per_user_mean": round(float(per_user.mean()), 2),
        "interactions_per_user_median": float(per_user.median()),
        "interactions_per_item_mean": round(float(per_item.mean()), 2),
        "item_popularity_gini": gini(per_item.to_numpy()),
        "head20pct_interaction_share_pct": round(
            100 * pop_sorted[:head].sum() / n, 2),
        "single_interaction_users_pct": round(
            100 * float((per_user == 1).mean()), 2),
        # Alias kept so older report consumers do not break. The protocol
        # itself is short-history; this figure is a corpus property.
        "cold_start_users_1_interaction_pct": round(
            100 * float((per_user == 1).mean()), 2),
        "rating_mean": round(float(df["Rating_clean"].mean()), 3),
        "rating_std": round(float(df["Rating_clean"].std()), 3),
    }


def longtail_figure(df: pd.DataFrame) -> None:
    ps.apply_acl_style()
    per_item = df.groupby("hotel_key").size().sort_values(ascending=False).to_numpy()
    cum = np.cumsum(per_item) / per_item.sum()
    x = np.arange(1, len(per_item) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(ps.PAGE_WIDTH, 2.35))
    axes[0].plot(x, per_item, color=ps.VIH_BLUE, linewidth=1.2)
    axes[0].set_xlabel("Hotel rank")
    axes[0].set_ylabel("Interactions")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.35)

    lor_x = np.concatenate([[0], x / len(per_item)])
    lor_y = np.concatenate([[0], cum])
    axes[1].plot(lor_x, lor_y, color=ps.VIH_BLUE, linewidth=1.2, label="ViHoRec")
    axes[1].plot([0, 1], [0, 1], "--", color=ps.VIH_GRAY, linewidth=1.0, label="Equality")
    axes[1].set_xlabel("Cumulative hotel share")
    axes[1].set_ylabel("Cumulative interaction share")
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].grid(alpha=0.35)
    fig.tight_layout(pad=0.6)
    ps.save_fig(fig, IMG_DIR / "LongTail.png")
    plt.close(fig)


def run() -> dict:
    df = load_clean()
    report = {"characterization": characterize(df)}
    longtail_figure(df)

    # Ablation A: entity resolution ON (canonical key) vs OFF (raw name).
    on = build_split(df, "CustomerName", "hotel_key")
    off = build_split(df, "CustomerName", "NameHotel")
    report["ablation_entity_resolution"] = {
        "ER_on": {"n_items": on[3], **_bench(*on)},
        "ER_off": {"n_items": off[3], **_bench(*off)},
    }

    # Ablation B: minimum-interaction threshold sweep.
    sweep = {}
    for k in (2, 3, 4, 5, 8):
        tr, ti, nu, ni = build_split(df, "CustomerName", "hotel_key", min_k=k)
        sparsity = round(100 * (1 - (len(tr) + len(ti)) / (nu * ni)), 3)
        sweep[k] = {"n_users": nu, "n_items": ni,
                    "sparsity_pct": sparsity, "Recall@10": _bench(tr, ti, nu, ni)["Recall@10"]}
    report["ablation_min_interactions"] = sweep

    # Ablation C: temporal vs random leave-one-out.
    tmp = build_split(df, "CustomerName", "hotel_key", mode="temporal")
    rnd = build_split(df, "CustomerName", "hotel_key", mode="random")
    report["ablation_split_protocol"] = {
        "temporal": _bench(*tmp),
        "random": _bench(*rnd),
    }

    (C.OUT_REPORTS / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
