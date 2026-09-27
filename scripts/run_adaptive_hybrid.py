"""Adaptive Hybrid experiments: λ sweep, sensitivity, and paper figures.

Runs on the public three-way short-history split. Every search
(λ and (α_min, n0, τ)) is decided on ``val.csv``; the test fold is scored
once with the chosen settings.

  1) Fixed-λ Hybrid sweep on validation (λ ∈ {0.0, 0.1, ..., 1.0})
  2) Adaptive Hybrid grid on validation; defaults remain a documented baseline
  3) Unconstrained Adaptive ablation (α_min=0) for claim-evidence honesty
  4) History-stratified comparison on test

Outputs under release/benchmark/ and reports/, plus paper Image/ figures.

Run:  python run_adaptive_hybrid.py
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
from short_history_eval import aggregate, per_user_metrics, BUCKETS

OUT = C.OUT_RELEASE / "benchmark"
REPORTS = C.OUT_REPORTS
IMG_DIR = C.PAPER_IMG_DIR
LAMBDAS = [i / 10 for i in range(11)]
ALPHA_MIN_GRID = hb.ALPHA_MIN_GRID
N0_GRID = hb.N0_GRID
TAU_GRID = hb.TAU_GRID


def _write_markdown(res: pd.DataFrame, path) -> None:
    cols = list(res.columns)
    lines = ["| Method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for method, row in res.iterrows():
        lines.append(
            "| " + str(method) + " | "
            + " | ".join(
                f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating))
                else str(row[c])
                for c in cols
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def figure_lambda_sweep(sweep: pd.DataFrame, adaptive_r10: float) -> None:
    ps.apply_acl_style()
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.4))
    ax.plot(sweep["lambda"], sweep["Recall@10"], color=ps.VIH_BLUE,
            marker="o", label="Fixed Hybrid")
    ax.axhline(adaptive_r10, color=ps.VIH_ACCENT, linestyle="--",
               label=f"AdaptiveHybrid ({adaptive_r10:.3f})")
    ax.set_xlabel(r"Fixed CF weight $\lambda$")
    ax.set_ylabel("Recall@10")
    ax.set_xticks(LAMBDAS[::2])
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="best")
    ps.save_fig(fig, IMG_DIR / "HybridLambda.png")
    plt.close(fig)


def figure_alpha_schedule(
    alpha_min: float = hb.DEFAULT_ALPHA_MIN,
    n0: float = hb.DEFAULT_N0,
    tau: float = hb.DEFAULT_TAU,
) -> None:
    ps.apply_acl_style()
    ns = np.arange(1, 21)
    alpha = hb.alpha_logistic(ns, n0=n0, tau=tau, alpha_min=alpha_min)
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.2))
    ax.plot(ns, alpha, color=ps.VIH_BLUE, marker="o", markersize=3.5)
    ax.axhline(alpha_min, color=ps.VIH_GRAY, linestyle=":", alpha=0.8)
    ax.axvline(n0, color=ps.VIH_GRAY, linestyle=":", alpha=0.5)
    ax.set_xlabel("Train interactions $n$")
    ax.set_ylabel(r"CF weight $\alpha(n)$")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.35)
    ax.set_title(
        rf"$\alpha_{{\min}}={alpha_min:g},\ n_0={n0:g},\ \tau={tau:g}$",
        fontsize=8)
    ps.save_fig(fig, IMG_DIR / "AdaptiveAlpha.png")
    plt.close(fig)


def run() -> dict:
    tr, va, n_users, n_items = rb.load("val")
    _, te, _, _ = rb.load("test")
    mat = rb.build_matrix(tr, n_users, n_items)
    val_items = dict(zip(va.userID, va.itemID))
    test_items = dict(zip(te.userID, te.itemID))
    train_count = tr.groupby("userID").size().to_dict()
    comps = hb.prepare_components(tr, n_users, n_items)
    short_label = BUCKETS[0][2]  # shortest train-history bucket ("2")

    # --- Fixed-λ sweep on validation ---
    sweep_rows = []
    for lam in LAMBDAS:
        scores = hb.score_fixed_hybrid(tr, n_users, n_items, lam, components=comps)
        metrics = rb.evaluate(scores, mat, val_items)
        buckets = aggregate(per_user_metrics(scores, mat, val_items), train_count)
        row = {"lambda": lam, **metrics}
        for label in [b[2] for b in BUCKETS]:
            row[f"R@10[{label}]"] = buckets.loc[label, "Recall@10"]
        sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "hybrid_lambda_sweep.csv", index=False)
    best_idx = int(sweep["Recall@10"].idxmax())
    best_lam = float(sweep.loc[best_idx, "lambda"])

    # --- Adaptive Hybrid grid on validation; defaults stay a documented point ---
    adapt_scores = hb.score_adaptive_hybrid(
        tr, n_users, n_items, components=comps)
    adapt_metrics = rb.evaluate(adapt_scores, mat, test_items)
    adapt_buckets = aggregate(
        per_user_metrics(adapt_scores, mat, test_items), train_count)

    # --- Unconstrained ablation (content-heavy for short histories) ---
    uncon_scores = hb.score_adaptive_hybrid(
        tr, n_users, n_items, alpha_min=0.0, n0=5.0, tau=1.5, components=comps)
    uncon_metrics = rb.evaluate(uncon_scores, mat, test_items)
    uncon_buckets = aggregate(
        per_user_metrics(uncon_scores, mat, test_items), train_count)

    # --- Sensitivity: choose (α_min, n0, τ) on validation ---
    sens_rows = []
    best_adapt = None
    best_adapt_val = -1.0
    for amin in ALPHA_MIN_GRID:
        for n0 in N0_GRID:
            for tau in TAU_GRID:
                scores = hb.score_adaptive_hybrid(
                    tr, n_users, n_items, n0=n0, tau=tau, alpha_min=amin,
                    components=comps)
                val_metrics = rb.evaluate(scores, mat, val_items)
                buckets = aggregate(
                    per_user_metrics(scores, mat, val_items), train_count)
                sens_rows.append({
                    "alpha_min": amin, "n0": n0, "tau": tau,
                    "val_Recall@10": val_metrics["Recall@10"],
                    "val_NDCG@10": val_metrics["NDCG@10"],
                    "val_MRR": val_metrics["MRR"],
                    f"val_R@10[{short_label}]": buckets.loc[short_label, "Recall@10"],
                    "val_R@10[11+]": buckets.loc["11+", "Recall@10"],
                })
                if val_metrics["Recall@10"] > best_adapt_val:
                    best_adapt_val = val_metrics["Recall@10"]
                    best_adapt = {"alpha_min": amin, "n0": n0, "tau": tau}
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "hybrid_sensitivity.csv", index=False)

    tuned_adapt_scores = hb.score_adaptive_hybrid(
        tr, n_users, n_items, components=comps, **best_adapt)
    tuned_adapt_metrics = rb.evaluate(tuned_adapt_scores, mat, test_items)
    tuned_adapt_buckets = aggregate(
        per_user_metrics(tuned_adapt_scores, mat, test_items), train_count)

    # --- Comparison table ---
    userknn_scores = rb.score_userknn(mat)
    content_scores = cb.score_content_tfidf(tr, n_users, n_items)
    fixed_scores = hb.score_fixed_hybrid(
        tr, n_users, n_items, best_lam, components=comps)

    comparison = {
        "UserKNN-cosine": rb.evaluate(userknn_scores, mat, test_items),
        "Content-TFIDF": rb.evaluate(content_scores, mat, test_items),
        f"Hybrid-fixed (lam={best_lam:.1f})": rb.evaluate(fixed_scores, mat, test_items),
        "AdaptiveHybrid-uncon": uncon_metrics,
        "AdaptiveHybrid-default": adapt_metrics,
        "AdaptiveHybrid-valtuned": tuned_adapt_metrics,
    }
    cmp_df = pd.DataFrame(comparison).T
    cmp_df.index.name = "Method"
    cmp_df.to_csv(OUT / "adaptive_hybrid_results.csv")
    _write_markdown(cmp_df, OUT / "adaptive_hybrid_results.md")

    # Stratified comparison
    strat = {}
    for name, scores in {
        "UserKNN": userknn_scores,
        "Content-TFIDF": content_scores,
        f"Hybrid-fixed (lam={best_lam:.1f})": fixed_scores,
        "AdaptiveHybrid-uncon": uncon_scores,
        "AdaptiveHybrid-default": adapt_scores,
        "AdaptiveHybrid-valtuned": tuned_adapt_scores,
    }.items():
        strat[name] = aggregate(
            per_user_metrics(scores, mat, test_items), train_count
        ).to_dict("index")

    report = {
        "selection_fold": "val",
        "report_fold": "test",
        "best_fixed_lambda": best_lam,
        "adaptive_default": {
            "alpha_min": hb.DEFAULT_ALPHA_MIN,
            "n0": hb.DEFAULT_N0,
            "tau": hb.DEFAULT_TAU,
            "global": adapt_metrics,
            "buckets": adapt_buckets.to_dict("index"),
        },
        "adaptive_valtuned": {
            **best_adapt,
            "val_Recall@10": best_adapt_val,
            "global": tuned_adapt_metrics,
            "buckets": tuned_adapt_buckets.to_dict("index"),
        },
        "adaptive_unconstrained": {
            "alpha_min": 0.0,
            "n0": 5.0,
            "tau": 1.5,
            "global": uncon_metrics,
            "buckets": uncon_buckets.to_dict("index"),
        },
        "comparison_global": comparison,
        "comparison_stratified": strat,
        "sensitivity": sens.to_dict("records"),
        "lambda_sweep": sweep.to_dict("records"),
    }
    (REPORTS / "adaptive_hybrid.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_lambda_sweep(sweep, tuned_adapt_metrics["Recall@10"])
    figure_alpha_schedule(**best_adapt)

    print("=== Global comparison ===")
    print(cmp_df.to_string())
    print(f"\nBest fixed lam={best_lam:.1f} (selected on val)")
    print(f"Val-tuned Adaptive {best_adapt} (val Recall@10={best_adapt_val:.4f})")
    print("\n=== AdaptiveHybrid-default buckets (test) ===")
    print(adapt_buckets.to_string())
    print("\n=== AdaptiveHybrid-valtuned buckets (test) ===")
    print(tuned_adapt_buckets.to_string())
    print("\n=== Unconstrained Adaptive buckets ===")
    print(uncon_buckets.to_string())
    print(f"\n=== Sensitivity top-8 by val R@10[{short_label}] ===")
    print(sens.sort_values([f"val_R@10[{short_label}]", "val_Recall@10"],
                           ascending=False).head(8).to_string(index=False))
    print(f"\nWrote figures to {IMG_DIR}")
    return report


if __name__ == "__main__":
    run()
