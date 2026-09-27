"""Bootstrap CIs and paired Wilcoxon tests for leave-one-out ranking metrics.

Per-user Recall@10 / NDCG@10 are bounded, zero-inflated, and not Gaussian, so
the paper reports percentile bootstrap 95% CIs and a paired Wilcoxon signed-rank
test rather than a t-test. Users are the resampling unit (and the pairing unit).

No extra dependency beyond numpy + scipy (scipy is used only here).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

N_BOOT = 10_000
BOOT_SEED = 42
ALPHA = 0.05
TOPK = 10


def per_user_r10_ndcg_mrr(scores, mat, test_items) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Aligned arrays of per-user Recall@10, NDCG@10, MRR (seen items masked)."""
    scores = scores.copy()
    scores[mat > 0] = -np.inf
    order = np.argsort(-scores, axis=1)
    users = sorted(test_items)
    rec, ndcg, mrr = [], [], []
    for u in users:
        target = test_items[u]
        full_rank = int(np.where(order[u] == target)[0][0]) + 1
        topk = order[u, :TOPK]
        hit = target in topk
        rec.append(1.0 if hit else 0.0)
        if hit:
            rank = int(np.where(topk == target)[0][0]) + 1
            ndcg.append(1.0 / np.log2(rank + 1))
        else:
            ndcg.append(0.0)
        mrr.append(1.0 / full_rank)
    return (
        np.array(rec, dtype=np.float64),
        np.array(ndcg, dtype=np.float64),
        np.array(mrr, dtype=np.float64),
        users,
    )


def user_metric_vectors(scores, mat, test_items) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Aligned arrays of per-user Recall@10, NDCG@10, MRR."""
    return per_user_r10_ndcg_mrr(scores, mat, test_items)


def bootstrap_ci(
    x: np.ndarray,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    seed: int = BOOT_SEED,
) -> dict:
    """Percentile bootstrap CI for the mean, resampling users with replacement."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": round(float(x.mean()), 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "n": int(n),
        "n_boot": int(n_boot),
    }


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank on paired per-user scores (a vs b).

    Zeros (ties) are dropped (Wilcoxon's original method). With leave-one-out
    Recall@10 many pairs tie at 0; the test is then driven by users where the
    two models disagree, which is the comparison the table needs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n_zero = int((diff == 0).sum())
    usable = diff != 0
    if usable.sum() < 10:
        return {
            "p": None,
            "statistic": None,
            "n_nonzero": int(usable.sum()),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_zero": n_zero,
            "mean_diff": round(float(diff.mean()), 4),
        }
    res = stats.wilcoxon(diff[usable], alternative="two-sided", zero_method="wilcox")
    return {
        "p": round(float(res.pvalue), 4),
        "statistic": round(float(res.statistic), 4),
        "n_nonzero": int(usable.sum()),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "mean_diff": round(float(diff.mean()), 4),
    }


def stars(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def ci_tex(ci: dict) -> str:
    return f"{ci['mean']:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]"
