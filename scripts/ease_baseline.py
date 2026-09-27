"""Closed-form EASE (Steck, WWW 2019) on the ViHoRec public split.

Item–item ridge regression with a zero diagonal. 535 items, so the inverse is
cheap and the model stays in the dependency-free table. ``λ`` is selected on
``val.csv``; test is scored once.

Run:  python ease_baseline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import run_baselines as rb

LAMBDAS = (10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)


def score_ease(mat: np.ndarray, lam: float = 500.0) -> np.ndarray:
    """Return user–item scores ``X B`` for binary interaction matrix ``mat``."""
    x = mat.astype(np.float64)
    n_items = x.shape[1]
    gram = x.T @ x
    gram.flat[:: n_items + 1] += float(lam)
    precision = np.linalg.inv(gram)
    diag = np.diag(precision).copy()
    diag[diag == 0.0] = 1e-12
    b = precision / (-diag[:, None])
    np.fill_diagonal(b, 0.0)
    return (x @ b).astype(np.float32)


def select_ease_lambda(mat, val_items, lambdas=LAMBDAS) -> float:
    best_lam, best_r = 500.0, -1.0
    for lam in lambdas:
        metrics = rb.evaluate(score_ease(mat, lam), mat, val_items)
        if metrics["Recall@10"] > best_r:
            best_lam, best_r = float(lam), metrics["Recall@10"]
    return best_lam


def run() -> pd.DataFrame:
    tr, va, n_users, n_items = rb.load("val")
    _, te, _, _ = rb.load("test")
    mat = rb.build_matrix(tr, n_users, n_items)
    val_items = dict(zip(va.userID, va.itemID))
    test_items = dict(zip(te.userID, te.itemID))
    lam = select_ease_lambda(mat, val_items)
    metrics = rb.evaluate(score_ease(mat, lam), mat, test_items)
    print(f"EASE lam={lam:g} (selected on val Recall@10)")
    print(pd.Series(metrics).to_string())
    return pd.Series(metrics, name=f"EASE (lam={lam:g})").to_frame().T


if __name__ == "__main__":
    run()
