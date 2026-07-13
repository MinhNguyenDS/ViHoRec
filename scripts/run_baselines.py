"""Reproducible baseline benchmark for the ViHoRec public split.

Dependency-free top-K rankers over the leave-last-one-out split produced by
make_benchmark_split.py:
  * Random    - uniform random ranking (sanity floor);
  * MostPop   - non-personalised popularity ranking;
  * ItemKNN   - cosine item-item similarity on the binary user-item matrix;
  * UserKNN   - cosine user-user similarity;
  * BPR-MF    - Bayesian Personalised Ranking matrix factorisation (learned),
                averaged over several seeds with standard deviation.

Reports MAP@K, NDCG@K, Precision@K, Recall@K (K in {5,10}) and MRR (full rank).
All models run on the SAME public split, so the numbers are directly comparable.

Run:  python run_baselines.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import content_baseline as cb

OUT = C.OUT_RELEASE / "benchmark"
KS = (5, 10)
TOPN = 10
BPR_SEEDS = (0, 1, 2)


def load():
    tr = pd.read_csv(OUT / "train.csv")
    te = pd.read_csv(OUT / "test.csv")
    n_users = int(max(tr.userID.max(), te.userID.max())) + 1
    n_items = int(max(tr.itemID.max(), te.itemID.max())) + 1
    return tr, te, n_users, n_items


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


def run() -> pd.DataFrame:
    tr, te, n_users, n_items = load()
    mat = build_matrix(tr, n_users, n_items)
    test_items = dict(zip(te.userID, te.itemID))

    rows, stds = {}, {}
    rand_runs = [evaluate(score_random(mat, s), mat, test_items) for s in BPR_SEEDS]
    rows["Random"], stds["Random"] = _mean_std(rand_runs)
    rows["MostPop"] = evaluate(score_mostpop(mat, n_users), mat, test_items)
    rows["ItemKNN-cosine"] = evaluate(score_itemknn(mat), mat, test_items)
    rows["UserKNN-cosine"] = evaluate(score_userknn(mat), mat, test_items)

    bpr_runs = [evaluate(train_bpr(tr, n_users, n_items, seed=s), mat, test_items)
                for s in BPR_SEEDS]
    rows["BPR-MF"], stds["BPR-MF"] = _mean_std(bpr_runs)

    rows["Content-TFIDF"] = evaluate(
        cb.score_content_tfidf(tr, n_users, n_items), mat, test_items)

    res = pd.DataFrame(rows).T
    res.index.name = "Method"
    res.to_csv(OUT / "baseline_results.csv")
    pd.DataFrame(stds).T.to_csv(OUT / "baseline_results_std.csv")
    _write_markdown(res, OUT / "baseline_results.md")
    print(res.to_string())
    print("\nStd (stochastic models):")
    print(pd.DataFrame(stds).T.to_string())
    return res


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
