"""Cold-start-stratified evaluation on the ViHoRec public split.

The single global metric hides how models behave for users with little history.
This script buckets test users by their number of TRAIN interactions and reports
per-bucket ranking metrics, exposing the cold-start regime that dominates
ViHoRec (most users have very short histories).

Buckets (by #train interactions after leave-last-one-out): 3, 4-5, 6-10, 11+.

Run:  python cold_start_eval.py
Outputs: reports/cold_start.json, Image/ColdStart.png
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import plot_style as ps
import run_baselines as rb

OUT = C.OUT_RELEASE / "benchmark"
IMG_DIR = C.PAPER_IMG_DIR
BUCKETS = [(3, 3, "3"), (4, 5, "4--5"), (6, 10, "6--10"), (11, 10**9, "11+")]
TOPK = 10


def _bucket(n: int) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= n <= hi:
            return label
    return BUCKETS[-1][2]


def per_user_metrics(scores, mat, test_items) -> dict:
    """Return {user: (recall@10, ndcg@10, rr)} with seen items masked."""
    scores = scores.copy()
    scores[mat > 0] = -np.inf
    order = np.argsort(-scores, axis=1)
    out = {}
    for u, target in test_items.items():
        full_rank = int(np.where(order[u] == target)[0][0]) + 1
        topk = order[u, :TOPK]
        hit = target in topk
        if hit:
            rank = int(np.where(topk == target)[0][0]) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        out[u] = (1.0 if hit else 0.0, ndcg, 1.0 / full_rank)
    return out


def aggregate(per_user: dict, train_count: dict) -> pd.DataFrame:
    rows = {}
    for u, (rec, ndcg, rr) in per_user.items():
        rows.setdefault(_bucket(train_count[u]), []).append((rec, ndcg, rr))
    data = {}
    for _, _, label in BUCKETS:
        vals = rows.get(label, [])
        if vals:
            arr = np.array(vals)
            data[label] = {"n_users": len(vals),
                           "Recall@10": round(float(arr[:, 0].mean()), 4),
                           "NDCG@10": round(float(arr[:, 1].mean()), 4),
                           "MRR": round(float(arr[:, 2].mean()), 4)}
        else:
            data[label] = {"n_users": 0, "Recall@10": 0.0, "NDCG@10": 0.0, "MRR": 0.0}
    return pd.DataFrame(data).T


def figure(results: dict) -> None:
    ps.apply_acl_style()
    labels = [b[2] for b in BUCKETS]
    styles = {
        "MostPop": {"color": ps.VIH_GRAY, "marker": "s", "linestyle": "--"},
        "UserKNN": {"color": ps.VIH_BLUE, "marker": "o", "linestyle": "-"},
        "BPR-MF": {"color": ps.VIH_ACCENT, "marker": "^", "linestyle": "-."},
    }
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.5))
    for name, df in results.items():
        sty = styles.get(name, {})
        ax.plot(labels, [df.loc[l, "Recall@10"] for l in labels],
                label=name, **sty)
    ax.set_xlabel("Train interactions per user")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, max(0.18, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="upper left")
    ps.save_fig(fig, IMG_DIR / "ColdStart.png")
    plt.close(fig)


def run() -> dict:
    tr, te, n_users, n_items = rb.load()
    mat = rb.build_matrix(tr, n_users, n_items)
    test_items = dict(zip(te.userID, te.itemID))
    train_count = tr.groupby("userID").size().to_dict()

    model_scores = {
        "MostPop": rb.score_mostpop(mat, n_users),
        "UserKNN": rb.score_userknn(mat),
        "BPR-MF": rb.train_bpr(tr, n_users, n_items, seed=0),
    }
    results = {name: aggregate(per_user_metrics(s, mat, test_items), train_count)
               for name, s in model_scores.items()}
    figure(results)

    report = {name: df.to_dict("index") for name, df in results.items()}
    (C.OUT_REPORTS / "cold_start.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, df in results.items():
        print(f"\n== {name} ==")
        print(df.to_string())
    return report


if __name__ == "__main__":
    run()
