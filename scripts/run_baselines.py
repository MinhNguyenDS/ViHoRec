"""Reproducible baseline benchmark for the ViHoRec public split.

Dependency-free top-K rankers over the three-way temporal short-history split
produced by make_benchmark_split.py. Hyper-parameters that have a search
(Hybrid-fixed ``λ``) are selected on ``val.csv``; every reported number is
then computed once on ``test.csv``.
  * Random    - uniform random ranking (sanity floor);
  * MostPop   - non-personalised popularity ranking;
  * ItemKNN   - cosine item-item similarity on the binary user-item matrix;
  * UserKNN   - cosine user-user similarity;
  * BPR-MF    - Bayesian Personalised Ranking matrix factorisation (learned),
                averaged over several seeds with standard deviation;
  * Content-TFIDF - TF-IDF content baseline over hotel metadata;
  * Hybrid-fixed  - late fusion of UserKNN + Content-TFIDF with λ selected on val;
  * AdaptiveHybrid - history-gated fusion; (α_min, n0, τ) selected on val;
  * EASE      - closed-form item-item ridge (Steck 2019); λ selected on val.

Every reported test number is accompanied by a user-level 95% bootstrap CI
and a paired Wilcoxon signed-rank test versus UserKNN (`eval_stats.py`).

Reports MAP@K, NDCG@K, Precision@K, Recall@K (K in {5,10}) and MRR (full rank).
All models run on the SAME public split, so the numbers are directly comparable.

Run:  python run_baselines.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import content_baseline as cb
import ease_baseline as ease
import eval_stats as es
import hybrid_baseline as hb

OUT = C.OUT_RELEASE / "benchmark"
KS = (5, 10)
TOPN = 10
BPR_SEEDS = (0, 1, 2)
REF_MODEL = "UserKNN-cosine"


def load(eval_fold: str = "test"):
    """Return ``(train, eval_fold, n_users, n_items)`` for the public split.

    Every hyper-parameter must be chosen with ``eval_fold="val"``; ``"test"`` is
    for the final table only. The universe size comes from the id maps rather
    than from the folds present, so a model always scores the full catalogue
    even when a fold happens to contain no interaction with the last item.
    """
    if eval_fold not in {"val", "test"}:
        raise ValueError(f"eval_fold must be 'val' or 'test', got {eval_fold!r}")
    tr = pd.read_csv(OUT / "train.csv")
    ev = pd.read_csv(OUT / f"{eval_fold}.csv")
    n_users = len(pd.read_csv(OUT / "user_map.csv"))
    n_items = len(pd.read_csv(OUT / "item_map.csv"))
    return tr, ev, n_users, n_items


def build_matrix(tr, n_users, n_items):
    m = np.zeros((n_users, n_items), dtype=np.float32)
    m[tr.userID.to_numpy(), tr.itemID.to_numpy()] = 1.0
    return m


def _cosine(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = a / norm
    return unit @ unit.T


def score_random(mat, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(mat.shape).astype(np.float32)


def score_mostpop(mat, n_users):
    pop = mat.sum(axis=0)
    return np.tile(pop, (n_users, 1))


def score_itemknn(mat):
    sim = _cosine(mat.T)
    np.fill_diagonal(sim, 0.0)
    return mat @ sim


def score_userknn(mat):
    sim = _cosine(mat)
    np.fill_diagonal(sim, 0.0)
    return sim @ mat


def train_bpr(tr, n_users, n_items, factors=32, lr=0.05, reg=0.01,
              epochs=40, seed=0):
    """Vectorised BPR matrix factorisation; returns the full score matrix."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, (n_users, factors)).astype(np.float32)
    Q = rng.normal(0, 0.1, (n_items, factors)).astype(np.float32)
    users = tr.userID.to_numpy()
    pos = tr.itemID.to_numpy()
    n = len(users)
    for _ in range(epochs):
        idx = rng.integers(0, n, size=n)
        u, i = users[idx], pos[idx]
        j = rng.integers(0, n_items, size=n)
        pu, qi, qj = P[u], Q[i], Q[j]
        x = np.sum(pu * (qi - qj), axis=1)
        sig = (1.0 / (1.0 + np.exp(x)))[:, None]
        np.add.at(P, u, lr * (sig * (qi - qj) - reg * pu))
        np.add.at(Q, i, lr * (sig * pu - reg * qi))
        np.add.at(Q, j, lr * (-sig * pu - reg * qj))
    return P @ Q.T


def evaluate(scores, mat, test_items) -> dict:
    """Mask seen items, then compute ranking metrics against 1 held-out item."""
    scores = scores.copy()
    scores[mat > 0] = -np.inf
    order = np.argsort(-scores, axis=1)
    topn = order[:, :TOPN]

    out = {}
    rr = []
    for u, target in test_items.items():
        full_rank = int(np.where(order[u] == target)[0][0]) + 1
        rr.append(1.0 / full_rank)
    out["MRR"] = round(float(np.mean(rr)), 4)

    for k in KS:
        precisions, recalls, aps, ndcgs = [], [], [], []
        for u, target in test_items.items():
            topk = topn[u, :k]
            hit = target in topk
            precisions.append((1.0 / k) if hit else 0.0)
            recalls.append(1.0 if hit else 0.0)
            if hit:
                rank = int(np.where(topk == target)[0][0]) + 1
                aps.append(1.0 / rank)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                aps.append(0.0)
                ndcgs.append(0.0)
        out[f"MAP@{k}"] = round(float(np.mean(aps)), 4)
        out[f"NDCG@{k}"] = round(float(np.mean(ndcgs)), 4)
        out[f"Precision@{k}"] = round(float(np.mean(precisions)), 4)
        out[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)
    return out


def _mean_std(dicts: list[dict]) -> tuple[dict, dict]:
    keys = dicts[0].keys()
    mean = {k: round(float(np.mean([d[k] for d in dicts])), 4) for k in keys}
    std = {k: round(float(np.std([d[k] for d in dicts])), 4) for k in keys}
    return mean, std


def select_fixed_lambda(tr, n_users, n_items, comps, mat, val_items) -> float:
    """Pick Hybrid-fixed λ on the validation fold. Never look at test."""
    best_lam, best_r10 = 0.5, -1.0
    for lam in [i / 10 for i in range(11)]:
        metrics = evaluate(
            hb.score_fixed_hybrid(tr, n_users, n_items, lam, components=comps),
            mat, val_items)
        if metrics["Recall@10"] > best_r10:
            best_lam, best_r10 = lam, metrics["Recall@10"]
    return best_lam


def run() -> pd.DataFrame:
    """Val-tune searchable HPs, score test once, write means + CI + Wilcoxon."""
    tr, va, n_users, n_items = load("val")
    _, te, _, _ = load("test")
    mat = build_matrix(tr, n_users, n_items)
    val_items = dict(zip(va.userID, va.itemID))
    test_items = dict(zip(te.userID, te.itemID))
    comps = hb.prepare_components(tr, n_users, n_items)

    best_lam = select_fixed_lambda(tr, n_users, n_items, comps, mat, val_items)
    adapt_hp, adapt_val_r10, _ = hb.select_adaptive_params(
        tr, n_users, n_items, comps, mat, val_items)
    ease_lam = ease.select_ease_lambda(mat, val_items)

    score_bank = {
        "MostPop": score_mostpop(mat, n_users),
        "ItemKNN-cosine": score_itemknn(mat),
        "UserKNN-cosine": score_userknn(mat),
        "Content-TFIDF": cb.score_content_tfidf(tr, n_users, n_items),
        f"Hybrid-fixed (lam={best_lam:.1f})": hb.score_fixed_hybrid(
            tr, n_users, n_items, best_lam, components=comps),
        "AdaptiveHybrid": hb.score_adaptive_hybrid(
            tr, n_users, n_items, components=comps, **adapt_hp),
        f"EASE (lam={ease_lam:g})": ease.score_ease(mat, ease_lam),
    }

    rows, stds = {}, {}
    per_user = {}  # method -> (rec, ndcg, mrr, users)

    rand_scores = [score_random(mat, s) for s in BPR_SEEDS]
    rows["Random"], stds["Random"] = _mean_std(
        [evaluate(s, mat, test_items) for s in rand_scores])
    recs, ndcgs, mrrs = [], [], []
    users = None
    for s in rand_scores:
        r, n, m, users = es.user_metric_vectors(s, mat, test_items)
        recs.append(r); ndcgs.append(n); mrrs.append(m)
    per_user["Random"] = (
        np.mean(recs, axis=0), np.mean(ndcgs, axis=0), np.mean(mrrs, axis=0), users)

    bpr_scores = [train_bpr(tr, n_users, n_items, seed=s) for s in BPR_SEEDS]
    rows["BPR-MF"], stds["BPR-MF"] = _mean_std(
        [evaluate(s, mat, test_items) for s in bpr_scores])
    recs, ndcgs, mrrs = [], [], []
    for s in bpr_scores:
        r, n, m, users = es.user_metric_vectors(s, mat, test_items)
        recs.append(r); ndcgs.append(n); mrrs.append(m)
    per_user["BPR-MF"] = (
        np.mean(recs, axis=0), np.mean(ndcgs, axis=0), np.mean(mrrs, axis=0), users)

    for name, scores in score_bank.items():
        rows[name] = evaluate(scores, mat, test_items)
        per_user[name] = es.user_metric_vectors(scores, mat, test_items)

    res = pd.DataFrame(rows).T
    res.index.name = "Method"
    res.to_csv(OUT / "baseline_results.csv")
    pd.DataFrame(stds).T.to_csv(OUT / "baseline_results_std.csv")
    _write_markdown(res, OUT / "baseline_results.md")

    ref_rec = per_user[REF_MODEL][0]
    ci_rows, test_rows, user_cols = [], [], {"userID": per_user[REF_MODEL][3]}
    for name, (rec, ndcg, mrr, users) in per_user.items():
        r_ci = es.bootstrap_ci(rec)
        n_ci = es.bootstrap_ci(ndcg)
        vs = es.wilcoxon_paired(rec, ref_rec) if name != REF_MODEL else {
            "p": None, "n_nonzero": 0, "n_pos": 0, "n_neg": 0, "n_zero": len(rec),
            "mean_diff": 0.0, "statistic": None,
        }
        star = es.stars(vs["p"]) if name != REF_MODEL else ""
        ci_rows.append({
            "Method": name,
            "Recall@10": r_ci["mean"],
            "Recall@10_ci_lo": r_ci["ci_lo"],
            "Recall@10_ci_hi": r_ci["ci_hi"],
            "NDCG@10": n_ci["mean"],
            "NDCG@10_ci_lo": n_ci["ci_lo"],
            "NDCG@10_ci_hi": n_ci["ci_hi"],
            "MRR": round(float(mrr.mean()), 4),
            "p_vs_UserKNN_R@10": vs["p"],
            "sig_vs_UserKNN": star,
            "wilcoxon_n_nonzero": vs["n_nonzero"],
            "mean_diff_vs_UserKNN": vs["mean_diff"],
        })
        if name != REF_MODEL:
            test_rows.append({
                "comparison": f"{name} vs {REF_MODEL}",
                "metric": "Recall@10",
                **vs,
                "stars": star,
            })
        user_cols[f"{name}_R@10"] = rec
        user_cols[f"{name}_N@10"] = ndcg

    # Adaptive vs best fixed hybrid (same users).
    adapt_name = "AdaptiveHybrid"
    fixed_name = f"Hybrid-fixed (lam={best_lam:.1f})"
    ah_vs_fix = es.wilcoxon_paired(per_user[adapt_name][0], per_user[fixed_name][0])
    test_rows.append({
        "comparison": f"{adapt_name} vs {fixed_name}",
        "metric": "Recall@10",
        **ah_vs_fix,
        "stars": es.stars(ah_vs_fix["p"]),
    })

    ci_df = pd.DataFrame(ci_rows).set_index("Method")
    ci_df.to_csv(OUT / "baseline_ci.csv")
    pd.DataFrame(test_rows).to_csv(OUT / "pairwise_tests.csv", index=False)
    pd.DataFrame(user_cols).to_csv(OUT / "per_user_metrics.csv", index=False)
    _write_ci_markdown(ci_df, OUT / "baseline_ci.md", best_lam, adapt_hp, ease_lam, adapt_val_r10)

    print(res.to_string())
    print("\nStd (stochastic models):")
    print(pd.DataFrame(stds).T.to_string())
    print("\n=== Recall@10 with 95% bootstrap CI; Wilcoxon vs UserKNN ===")
    print(ci_df.to_string())
    print(f"\nBest fixed Hybrid lam={best_lam:.1f} (val Recall@10)")
    print(f"Val-tuned Adaptive {adapt_hp} (val Recall@10={adapt_val_r10:.4f})")
    print(f"EASE lam={ease_lam:g} (selected on val Recall@10)")
    print(f"Adaptive vs {fixed_name}: p={ah_vs_fix['p']} {es.stars(ah_vs_fix['p'])}")
    return res


def _write_ci_markdown(ci_df, path, best_lam, adapt_hp, ease_lam, adapt_val_r10) -> None:
    lines = [
        "# ViHoRec baselines (three-way split; val-tuned, test once)",
        "",
        f"Hybrid-fixed λ = {best_lam:.1f}. "
        f"Adaptive Hybrid {adapt_hp} (val Recall@10 = {adapt_val_r10:.4f}). "
        f"EASE λ = {ease_lam:g}.",
        "",
        "95% CIs are percentile bootstrap over users (10,000 resamples). "
        "p is two-sided Wilcoxon signed-rank on per-user Recall@10 vs UserKNN. "
        r"\* p<0.05, \*\* p<0.01, \*\*\* p<0.001.",
        "",
        "| Method | R@10 | 95% CI | N@10 | 95% CI | MRR | vs UserKNN |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for method, row in ci_df.iterrows():
        p = row["p_vs_UserKNN_R@10"]
        p_s = "—" if pd.isna(p) or p is None or p == "" else f"{float(p):.4f}{row['sig_vs_UserKNN']}"
        lines.append(
            f"| {method} | {row['Recall@10']:.4f} | "
            f"[{row['Recall@10_ci_lo']:.4f}, {row['Recall@10_ci_hi']:.4f}] | "
            f"{row['NDCG@10']:.4f} | "
            f"[{row['NDCG@10_ci_lo']:.4f}, {row['NDCG@10_ci_hi']:.4f}] | "
            f"{row['MRR']:.4f} | {p_s} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(res: pd.DataFrame, path) -> None:
    cols = list(res.columns)
    lines = ["| Method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for method, row in res.iterrows():
        lines.append("| " + method + " | "
                     + " | ".join(f"{row[c]:.4f}" for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
