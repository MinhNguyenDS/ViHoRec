"""Content-based TF-IDF baseline for the ViHoRec public split.

Builds hotel text profiles from the four attribute groups documented in the
paper (Facilities, Around, Vicinity, Price), aggregates a user profile from
training interactions, and ranks by cosine similarity. Dependency-free
(scikit-learn not required; uses a sparse TF-IDF implementation in numpy).

Run standalone:  python content_baseline.py
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

import config as C
import run_baselines as rb
from textnorm import canonical_hotel_key

CONTENT_COLS = ("Facilities", "Around", "Vicinity", "Price")


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _build_vocab(docs: list[list[str]], max_features: int = 4000) -> dict[str, int]:
    counts = Counter(tok for doc in docs for tok in doc)
    # Keep most frequent tokens (simple document-frequency filter).
    top = [w for w, _ in counts.most_common(max_features)]
    return {w: i for i, w in enumerate(top)}


def _tfidf_matrix(docs: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    n_docs = len(docs)
    n_feat = len(vocab)
    if n_feat == 0:
        return np.zeros((n_docs, 1), dtype=np.float32)
    tf = np.zeros((n_docs, n_feat), dtype=np.float32)
    for i, doc in enumerate(docs):
        if not doc:
            continue
        counts = Counter(doc)
        for w, c in counts.items():
            if w in vocab:
                tf[i, vocab[w]] = c / len(doc)
    df = np.sum(tf > 0, axis=0)
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    return tf * idf


def _hotel_text_lookup() -> dict[str, str]:
    """Map canonical hotel key -> concatenated content text."""
    meta = pd.read_csv(C.CONTENT_RAW_FILE)
    out: dict[str, str] = {}
    for _, row in meta.iterrows():
        parts = [str(row.get(c, "")) for c in CONTENT_COLS]
        text = " ".join(parts)
        key = canonical_hotel_key(row["NameHotel"])
        out[key] = text
    return out


def _item_to_content_key() -> dict[int, str]:
    """Map benchmark itemID -> canonical hotel key via release hotels table."""
    hotels = pd.read_csv(C.OUT_RELEASE / "hotels.csv")
    imap = pd.read_csv(C.OUT_RELEASE / "benchmark" / "item_map.csv")
    merged = imap.merge(hotels, on="hotel_id")
    return {int(r.itemID): canonical_hotel_key(r.name) for r in merged.itertuples()}


def score_content_tfidf(tr: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    content = _hotel_text_lookup()
    item_keys = _item_to_content_key()

    item_docs = []
    covered_items = []
    for item_id in range(n_items):
        key = item_keys.get(item_id, "")
        toks = _tokenize(content.get(key, ""))
        item_docs.append(toks)
        covered_items.append(bool(toks))

    vocab = _build_vocab(item_docs)
    item_mat = _tfidf_matrix(item_docs, vocab)
    if item_mat.shape[1] == 0:
        return np.zeros((n_users, n_items), dtype=np.float32)

    # L2-normalise item vectors.
    norms = np.linalg.norm(item_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_mat = item_mat / norms

    user_mat = np.zeros((n_users, item_mat.shape[1]), dtype=np.float32)
    for u, grp in tr.groupby("userID"):
        vecs = item_mat[grp.itemID.to_numpy()]
        if len(vecs):
            user_mat[int(u)] = vecs.mean(axis=0)

    u_norms = np.linalg.norm(user_mat, axis=1, keepdims=True)
    u_norms[u_norms == 0] = 1.0
    user_mat = user_mat / u_norms

    scores = user_mat @ item_mat.T
    # Items without metadata get zero score (cannot be recommended by content).
    for j, ok in enumerate(covered_items):
        if not ok:
            scores[:, j] = -np.inf
    return scores.astype(np.float32)


def coverage_stats() -> dict:
    imap = pd.read_csv(C.OUT_RELEASE / "benchmark" / "item_map.csv")
    content = _hotel_text_lookup()
    keys = _item_to_content_key()
    n = len(imap)
    covered = sum(1 for i in range(n) if content.get(keys.get(i, ""), ""))
    return {"benchmark_items": n, "with_content_metadata": covered,
            "coverage_pct": round(100 * covered / n, 2)}


def run() -> dict:
    tr, te, n_users, n_items = rb.load()
    mat = rb.build_matrix(tr, n_users, n_items)
    test_items = dict(zip(te.userID, te.itemID))
    metrics = rb.evaluate(score_content_tfidf(tr, n_users, n_items), mat, test_items)
    stats = coverage_stats()
    print("Content metadata coverage:", stats)
    print("Content-TFIDF metrics:", metrics)
    return {"coverage": stats, "metrics": metrics}


if __name__ == "__main__":
    run()
