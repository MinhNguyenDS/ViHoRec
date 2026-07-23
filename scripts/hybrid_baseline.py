"""Adaptive Hybrid baseline: history-gated fusion of UserKNN and Content-TFIDF.

Late-fuses collaborative filtering (UserKNN) and content (Content-TFIDF) with a
per-user weight alpha(n) that increases with train-history length n. Short
histories receive a larger content residual; longer histories approach pure CF.

    s_AH(u, i) = alpha(n_u) * tilde{s}_CF(u, i) + (1 - alpha(n_u)) * tilde{s}_CB(u, i)

where tilde{s} is per-user min-max normalisation and

    alpha(n) = alpha_min + (1 - alpha_min) * sigmoid((n - n0) / tau)

The CF floor alpha_min is required because Content-TFIDF is weaker than UserKNN
even for cold users on ViHoRec; unconstrained content-heavy gating hurts ranking.
Defaults: alpha_min=0.9, n0=4, tau=2.0.

Fixed-lambda hybrids (alpha ≡ λ) are provided as controls.

Run standalone:  python hybrid_baseline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import content_baseline as cb
import run_baselines as rb

DEFAULT_ALPHA_MIN = 0.9
DEFAULT_N0 = 4.0
DEFAULT_TAU = 2.0
EPS = 1e-8


def alpha_logistic(
    n,
    n0: float = DEFAULT_N0,
    tau: float = DEFAULT_TAU,
    alpha_min: float = DEFAULT_ALPHA_MIN,
) -> np.ndarray:
    """Bounded history gate: CF weight grows with train-history length n.

    Returns values in [alpha_min, 1]. Smaller n => closer to alpha_min (more
    content residual); larger n => closer to 1 (CF-dominant).
    """
    n = np.asarray(n, dtype=np.float64)
    tau = max(float(tau), EPS)
    alpha_min = float(np.clip(alpha_min, 0.0, 1.0))
    gate = 1.0 / (1.0 + np.exp(-(n - float(n0)) / tau))
    return alpha_min + (1.0 - alpha_min) * gate


def minmax_per_user(scores: np.ndarray) -> np.ndarray:
    """Per-user min-max normalisation; finite scores only (-inf stays -inf)."""
    out = np.full_like(scores, -np.inf, dtype=np.float32)
    for u in range(scores.shape[0]):
        row = scores[u]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        vals = row[finite]
        lo, hi = float(vals.min()), float(vals.max())
        denom = hi - lo
        if denom < EPS:
            out[u, finite] = 0.0
        else:
            out[u, finite] = ((vals - lo) / denom).astype(np.float32)
    return out


def train_counts(tr: pd.DataFrame, n_users: int) -> np.ndarray:
    """Return array of length n_users with #train interactions per user."""
    counts = np.zeros(n_users, dtype=np.float64)
    vc = tr.groupby("userID").size()
    counts[vc.index.to_numpy()] = vc.to_numpy(dtype=np.float64)
    return counts


def _component_scores(tr: pd.DataFrame, n_users: int, n_items: int):
    mat = rb.build_matrix(tr, n_users, n_items)
    s_cf = rb.score_userknn(mat).astype(np.float32)
    s_cb = cb.score_content_tfidf(tr, n_users, n_items).astype(np.float32)
    return mat, minmax_per_user(s_cf), minmax_per_user(s_cb)


def fuse(s_cf_n: np.ndarray, s_cb_n: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fuse normalised CF/CB scores with per-user alpha in [0, 1]."""
    a = alpha.astype(np.float32).reshape(-1, 1)
    # Where CB is -inf (no metadata), fall back to CF so hybrid can still rank.
    cb_safe = np.where(np.isfinite(s_cb_n), s_cb_n, s_cf_n)
    cf_safe = np.where(np.isfinite(s_cf_n), s_cf_n, cb_safe)
    return (a * cf_safe + (1.0 - a) * cb_safe).astype(np.float32)


def score_fixed_hybrid(
    tr: pd.DataFrame,
    n_users: int,
    n_items: int,
    lam: float,
    *,
    components: tuple | None = None,
) -> np.ndarray:
    """Fixed-weight hybrid: alpha(u) ≡ lam for all users (lam = CF weight)."""
    if components is None:
        _, s_cf_n, s_cb_n = _component_scores(tr, n_users, n_items)
    else:
        _, s_cf_n, s_cb_n = components
    alpha = np.full(n_users, float(lam), dtype=np.float64)
    return fuse(s_cf_n, s_cb_n, alpha)


def score_adaptive_hybrid(
    tr: pd.DataFrame,
    n_users: int,
    n_items: int,
    n0: float = DEFAULT_N0,
    tau: float = DEFAULT_TAU,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    *,
    components: tuple | None = None,
) -> np.ndarray:
    """History-gated Adaptive Hybrid (bounded logistic alpha)."""
    if components is None:
        _, s_cf_n, s_cb_n = _component_scores(tr, n_users, n_items)
    else:
        _, s_cf_n, s_cb_n = components
    n = train_counts(tr, n_users)
    alpha = alpha_logistic(n, n0=n0, tau=tau, alpha_min=alpha_min)
    return fuse(s_cf_n, s_cb_n, alpha)


def prepare_components(tr: pd.DataFrame, n_users: int, n_items: int):
    """Precompute (mat, norm_cf, norm_cb) once for sweeps."""
    return _component_scores(tr, n_users, n_items)


def run() -> dict:
    tr, te, n_users, n_items = rb.load()
    mat = rb.build_matrix(tr, n_users, n_items)
    test_items = dict(zip(te.userID, te.itemID))
    comps = prepare_components(tr, n_users, n_items)

    rows = {
        "UserKNN-cosine": rb.evaluate(rb.score_userknn(mat), mat, test_items),
        "Content-TFIDF": rb.evaluate(
            cb.score_content_tfidf(tr, n_users, n_items), mat, test_items),
        "AdaptiveHybrid": rb.evaluate(
            score_adaptive_hybrid(tr, n_users, n_items, components=comps),
            mat, test_items),
        "AdaptiveHybrid-unconstrained": rb.evaluate(
            score_adaptive_hybrid(
                tr, n_users, n_items, alpha_min=0.0, n0=5.0, tau=1.5,
                components=comps),
            mat, test_items),
    }
    for lam in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
        rows[f"Hybrid-fixed lam={lam:.1f}"] = rb.evaluate(
            score_fixed_hybrid(tr, n_users, n_items, lam, components=comps),
            mat, test_items)

    res = pd.DataFrame(rows).T
    print(res.to_string())
    return rows


if __name__ == "__main__":
    run()
