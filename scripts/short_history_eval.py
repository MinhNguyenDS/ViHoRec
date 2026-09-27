"""History-stratified evaluation on the ViHoRec public short-history split.

The protocol is **not** strict user cold-start: every evaluated user has at
least two training interactions (min-k = 4, last and second-last held out).
This script buckets test users by their train-history length so the paper can
report how models behave on the shortest *eligible* histories without calling
those users cold-start.

Buckets (train interactions after the three-way split): 2, 3--4, 5--10, 11+.

Hybrid-fixed ``λ`` is selected on the validation fold, then every model is
scored once on test. Adaptive Hybrid defaults stay frozen here; their
``(α_min, n0, τ)`` search lives in ``run_adaptive_hybrid.py`` and must also
use validation.

Run:  python short_history_eval.py
Outputs: reports/short_history.json, Image/ShortHistory.png
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import content_baseline as cb
import hybrid_baseline as hb
import plot_style as ps
import run_baselines as rb

OUT = C.OUT_RELEASE / "benchmark"
IMG_DIR = C.PAPER_IMG_DIR
# After holding out last + second-last, the shortest train history is 2.
BUCKETS = [(2, 2, "2"), (3, 4, "3--4"), (5, 10, "5--10"), (11, 10**9, "11+")]
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
            data[label] = {
                "n_users": len(vals),
                "Recall@10": round(float(arr[:, 0].mean()), 4),
                "NDCG@10": round(float(arr[:, 1].mean()), 4),
                "MRR": round(float(arr[:, 2].mean()), 4),
            }
        else:
            data[label] = {"n_users": 0, "Recall@10": 0.0, "NDCG@10": 0.0, "MRR": 0.0}
    return pd.DataFrame(data).T


def _plot_lines(results: dict, names: list[str], styles: dict, outfile: str,
                ylim: float = 0.20) -> None:
    ps.apply_acl_style()
    labels = [b[2] for b in BUCKETS]
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.5))
    for name in names:
        if name not in results:
            continue
        df = results[name]
        sty = styles.get(name, {"marker": "o", "linestyle": "-"})
        ax.plot(labels, [df.loc[l, "Recall@10"] for l in labels],
                label=name, **sty)
    ax.set_xlabel("Train interactions per user")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, max(ylim, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    ps.save_fig(fig, IMG_DIR / outfile)
    plt.close(fig)


def figure(results: dict, fixed_name: str) -> None:
    """Write ShortHistory.png and keep AdaptiveColdStart.png as an alias name."""
    main_names = ["MostPop", "UserKNN", "BPR-MF", fixed_name, "AdaptiveHybrid"]
    main_styles = {
        "MostPop": {"color": ps.VIH_GRAY, "marker": "s", "linestyle": "--"},
        "UserKNN": {"color": ps.VIH_BLUE, "marker": "o", "linestyle": "-"},
        "BPR-MF": {"color": "#8B5E3C", "marker": "^", "linestyle": "-."},
        fixed_name: {"color": "#5B8C5A", "marker": "v", "linestyle": "-."},
        "AdaptiveHybrid": {"color": ps.VIH_ACCENT, "marker": "D", "linestyle": "-"},
    }
    _plot_lines(results, main_names, main_styles, "ShortHistory.png", ylim=0.22)
    # Keep the submitted figure name so existing LaTeX includes still compile
    # until the manuscript caption is renamed.
    _plot_lines(results, main_names, main_styles, "ColdStart.png", ylim=0.22)

    hybrid_names = ["UserKNN", "Content-TFIDF", fixed_name, "AdaptiveHybrid"]
    hybrid_styles = {
        "UserKNN": {"color": ps.VIH_BLUE, "marker": "o", "linestyle": "-"},
        "Content-TFIDF": {"color": ps.VIH_GRAY, "marker": "s", "linestyle": "--"},
        fixed_name: {"color": "#5B8C5A", "marker": "^", "linestyle": "-."},
        "AdaptiveHybrid": {"color": ps.VIH_ACCENT, "marker": "D", "linestyle": "-"},
    }
    _plot_lines(results, hybrid_names, hybrid_styles, "AdaptiveShortHistory.png", ylim=0.22)
    _plot_lines(results, hybrid_names, hybrid_styles, "AdaptiveColdStart.png", ylim=0.22)


def run() -> dict:
    tr, va, n_users, n_items = rb.load("val")
    _, te, _, _ = rb.load("test")
    mat = rb.build_matrix(tr, n_users, n_items)
    val_items = dict(zip(va.userID, va.itemID))
    test_items = dict(zip(te.userID, te.itemID))
    train_count = tr.groupby("userID").size().to_dict()

    comps = hb.prepare_components(tr, n_users, n_items)
    best_lam = rb.select_fixed_lambda(tr, n_users, n_items, comps, mat, val_items)
    adapt_hp, _, _ = hb.select_adaptive_params(
        tr, n_users, n_items, comps, mat, val_items)
    fixed_scores = hb.score_fixed_hybrid(
        tr, n_users, n_items, best_lam, components=comps)

    model_scores = {
        "MostPop": rb.score_mostpop(mat, n_users),
        "UserKNN": rb.score_userknn(mat),
        "BPR-MF": rb.train_bpr(tr, n_users, n_items, seed=0),
        "Content-TFIDF": cb.score_content_tfidf(tr, n_users, n_items),
        f"Hybrid-fixed (lam={best_lam:.1f})": fixed_scores,
        "AdaptiveHybrid": hb.score_adaptive_hybrid(
            tr, n_users, n_items, components=comps, **adapt_hp),
    }
    results = {
        name: aggregate(per_user_metrics(s, mat, test_items), train_count)
        for name, s in model_scores.items()
    }
    fixed_name = f"Hybrid-fixed (lam={best_lam:.1f})"
    figure(results, fixed_name)

    report = {name: df.to_dict("index") for name, df in results.items()}
    report["_meta"] = {
        "protocol": "short-history / temporal leave-one-out, three-way",
        "not_strict_cold_start": True,
        "best_fixed_lambda_selected_on": "val",
        "best_fixed_lambda": best_lam,
        "adaptive_selected_on": "val",
        "adaptive_alpha_min": adapt_hp["alpha_min"],
        "adaptive_n0": adapt_hp["n0"],
        "adaptive_tau": adapt_hp["tau"],
        "buckets": [b[2] for b in BUCKETS],
        "users_per_bucket": {
            label: int((pd.Series(train_count).map(_bucket) == label).sum())
            for _, _, label in BUCKETS
        },
    }
    (C.OUT_REPORTS / "short_history.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, df in results.items():
        print(f"\n== {name} ==")
        print(df.to_string())
    print(f"\nBest fixed lam={best_lam:.1f} (selected on val)")
    return report


if __name__ == "__main__":
    run()
