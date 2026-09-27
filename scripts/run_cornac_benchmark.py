"""Reproduce the SOTA collaborative-filtering benchmark on the ViHoRec PUBLIC
split, using Cornac. This script exists to close the split-mismatch gap: the
extended results in the paper were originally produced under a different split,
and running this file regenerates comparable numbers on the released split.

WHY A SEPARATE SCRIPT?
    Cornac (and its neural models) pull in torch/tensorflow, which are heavy to
    install locally. This file is therefore meant to run on Google Colab or any
    GPU environment where those wheels install cleanly.

HOW TO RUN (Google Colab)
    1) Upload the public split:
         dataset_release/release/benchmark/train.csv
         dataset_release/release/benchmark/val.csv
         dataset_release/release/benchmark/test.csv
    2) Install dependencies:
         !pip install cornac torch
    3) Point TRAIN_CSV / VAL_CSV / TEST_CSV below at the uploaded files and run:
         !python run_cornac_benchmark.py
    Output: cornac_results.csv (one row per model, ranking metrics).

PROTOCOL
    Same public three-way short-history split as run_baselines.py. Neural
    hyper-parameters, when searched, must be chosen on val.csv; this script
    evaluates the published settings once on test.csv. Ranking metrics use
    the full item catalogue with seen items excluded. Relevance is implicit
    next-item: every crawled rating (>= 1) is a positive (threshold 0.5).
"""

from __future__ import annotations

import os

import pandas as pd

# Resolve the split files relative to this repo by default; override on Colab.
try:
    import config as C
    _BENCH = C.OUT_RELEASE / "benchmark"
    TRAIN_CSV = str(_BENCH / "train.csv")
    VAL_CSV = str(_BENCH / "val.csv")
    TEST_CSV = str(_BENCH / "test.csv")
    OUT_CSV = str(_BENCH / "cornac_results.csv")
except Exception:  # standalone on Colab
    TRAIN_CSV = "train.csv"
    VAL_CSV = "val.csv"
    TEST_CSV = "test.csv"
    OUT_CSV = "cornac_results.csv"

RATING_THRESHOLD = 0.5  # every crawled rating (>=1) counts as a positive
SEED = 42
TOP_K = 10


def _load_tuples(path: str):
    df = pd.read_csv(path)
    return list(zip(df.userID.astype(str), df.itemID.astype(str),
                    df.rating.astype(float)))


def build_models():
    """Instantiate the model zoo; skip any model whose backend is unavailable."""
    from cornac.models import (MostPop, UserKNN, ItemKNN, BPR, WMF, PMF, MF,
                               EASE)
    models = [
        MostPop(),
        UserKNN(k=200, similarity="cosine", name="UserKNN-cos"),
        ItemKNN(k=100, similarity="cosine", name="ItemKNN-cos"),
        MF(k=32, max_iter=100, learning_rate=0.01, lambda_reg=0.02, seed=SEED),
        PMF(k=40, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=SEED),
        WMF(k=50, max_iter=100, seed=SEED),
        BPR(k=32, max_iter=200, learning_rate=0.01, lambda_reg=0.01, seed=SEED),
        EASE(lamb=500),
    ]
    # Optional neural models (need torch/tensorflow); add if importable.
    for adder in (_maybe_neumf, _maybe_bivae, _maybe_vaecf):
        m = adder()
        if m is not None:
            models.append(m)
    return models


def _maybe_neumf():
    try:
        from cornac.models import NeuMF
        return NeuMF(num_factors=8, layers=[32, 16, 8], num_epochs=50,
                     batch_size=256, lr=0.001, seed=SEED)
    except Exception:
        return None


def _maybe_bivae():
    try:
        from cornac.models import BiVAECF
        return BiVAECF(k=50, encoder_structure=[100], n_epochs=100,
                       batch_size=128, learning_rate=0.001, seed=SEED)
    except Exception:
        return None


def _maybe_vaecf():
    try:
        from cornac.models import VAECF
        return VAECF(k=10, autoencoder_structure=[20], n_epochs=100,
                     batch_size=100, learning_rate=0.001, seed=SEED)
    except Exception:
        return None


def run() -> pd.DataFrame:
    import cornac
    from cornac.eval_methods import BaseMethod
    from cornac.metrics import MAP, MRR, Precision, Recall, NDCG

    bm = BaseMethod.from_splits(
        train_data=_load_tuples(TRAIN_CSV),
        test_data=_load_tuples(TEST_CSV),
        exclude_unknowns=True,
        rating_threshold=RATING_THRESHOLD,
        seed=SEED,
        verbose=True,
    )
    metrics = [MAP(), MRR(),
               Precision(k=5), Recall(k=5), NDCG(k=5),
               Precision(k=TOP_K), Recall(k=TOP_K), NDCG(k=TOP_K)]

    exp = cornac.Experiment(eval_method=bm, models=build_models(),
                            metrics=metrics, user_based=True)
    exp.run()

    rows = {}
    for res in exp.result:
        rows[res.model_name] = {k: round(float(v), 4)
                                for k, v in res.metric_avg_results.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "Method"
    table.to_csv(OUT_CSV)
    print("\nSaved:", OUT_CSV)
    print(table.to_string())
    return table


if __name__ == "__main__":
    run()
