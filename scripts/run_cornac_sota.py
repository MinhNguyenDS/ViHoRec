"""Cornac generative / hybrid SOTA baselines on the ViHoRec public split.

Evaluates the 2023–2025 next-item models from PreferredAI/cornac on the same
frozen three-way split as ``run_baselines.py`` / ``run_neural_baselines.py``:

    train.csv  -> fit
    val.csv    -> early-stop / best-on-val (never test)
    test.csv   -> scored once

Sequences follow Cornac ``NextItemEvaluation.leave_last_out`` construction
*without* re-splitting: session = user, train = seq[:-2], val prefix = seq[:-1],
test prefix = full seq. That puts the validation interaction in the test-time
history (it is not a training label). Do **not** call ``leave_last_out`` on
pooled UIRT rows: that would ignore ``val.csv`` / ``test.csv``.

Content embeddings are hotel name + city + Facilities/Around/Vicinity/Price,
encoded with multilingual MiniLM (Vietnamese). LETTER also gets truncated-SVD
item factors from the **train** user–item matrix (not random vectors).

SANSA (RecSys 2023) is collaborative filtering, not generative retrieval.
It is fit with Cornac ``Dataset.build`` on ``train.csv`` only (same masking
protocol as EASE / UserKNN). Sequential models below still use LLOO prefixes.

Skipped on this release, with reasons written to the report:

    Companion  — needs SentimentModality / aspect–opinion tuples; no reviews
    HypAR      — review-hypergraph; no review text
    DiffGRM    — paper-scale GPU recipe; skipped unless CUDA is available
                (or ``--try-diffgrm`` on CPU, which is not a paper-faithful run)

Run (conda env ViHoRec)::

    python -u run_cornac_sota.py
    python -u run_cornac_sota.py --models sansa
    python -u run_cornac_sota.py --smoke --models tiger
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import time
from collections import OrderedDict
from datetime import datetime, timezone

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

import config as C
import content_baseline as cb
import eval_stats as es
import run_baselines as rb
from textnorm import canonical_hotel_key

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT = C.OUT_RELEASE / "benchmark"
REPORTS = C.OUT_REPORTS
CACHE = REPORTS / "cache"
SEED = 42
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
METRIC_ORDER = [
    "MRR", "MAP@5", "NDCG@5", "Precision@5", "Recall@5",
    "MAP@10", "NDCG@10", "Precision@10", "Recall@10",
]
SKIP_REASONS = {
    "Companion": (
        "Needs SentimentModality (aspect–opinion tuples from reviews). "
        "ViHoRec does not release review text."
    ),
    "HypAR": (
        "Review-hypergraph model; the public release has no review documents."
    ),
    "DiffGRM": (
        "Paper-scale GPU recipe (d_model=256, 100 epochs, beam 128). "
        "This environment is torch CPU-only."
    ),
}
SANSA_LAMBDAS = (1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_fold(name: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"{name}.csv")
    df["userID"] = df["userID"].astype(int)
    df["itemID"] = df["itemID"].astype(int)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def _usit_row(user: int, item: int, ts: int) -> tuple[str, str, str, int]:
    u = str(int(user))
    return (u, u, str(int(item)), int(ts))


def frozen_usit_splits():
    """Build USIT tuples from the published CSVs, matching Cornac LLOO prefixes.

    Per user, chronological train rows then the val holdout then the test
    holdout. Train sessions are ``seq[:-2]``; val sessions ``seq[:-1]``;
    test sessions the full sequence. Session id = user id.
    """
    train = _load_fold("train")
    val = _load_fold("val")
    test = _load_fold("test")
    val_by = val.set_index("userID")
    test_by = test.set_index("userID")
    train_data, val_data, test_data = [], [], []
    n_users = 0
    for uid, grp in train.groupby("userID", sort=True):
        if uid not in val_by.index or uid not in test_by.index:
            continue
        n_users += 1
        ordered = grp.sort_values("timestamp", kind="mergesort")
        seq = [
            _usit_row(uid, r.itemID, r.timestamp)
            for r in ordered.itertuples(index=False)
        ]
        vr = val_by.loc[uid]
        tr = test_by.loc[uid]
        seq.append(_usit_row(uid, vr.itemID, vr.timestamp))
        seq.append(_usit_row(uid, tr.itemID, tr.timestamp))
        train_data.extend(seq[:-2])
        val_data.extend(seq[:-1])
        test_data.extend(seq)
    return train_data, val_data, test_data, n_users


def item_texts(n_items: int) -> tuple[list[str], int]:
    """Name + city + attribute groups, one string per benchmark itemID."""
    hotels = pd.read_csv(C.OUT_RELEASE / "hotels.csv")
    imap = pd.read_csv(OUT / "item_map.csv")
    merged = imap.merge(hotels, on="hotel_id", how="left")
    content = cb._hotel_text_lookup()
    texts = ["hotel"] * n_items
    n_with_meta = 0
    for r in merged.itertuples(index=False):
        iid = int(r.itemID)
        if iid < 0 or iid >= n_items:
            continue
        name = "" if pd.isna(r.name) else str(r.name)
        loc = "" if pd.isna(r.location) else str(r.location)
        extra = content.get(canonical_hotel_key(name), "")
        blob = " ".join(p for p in (name, loc, extra) if p).strip()
        texts[iid] = blob if blob else f"hotel {iid}"
        if extra.strip():
            n_with_meta += 1
    return texts, n_with_meta


def encode_items(texts: list[str], device: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"item_emb_{ENCODER_NAME.replace('/', '_')}.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if int(z["n"]) == len(texts) and str(z["encoder"]) == ENCODER_NAME:
            print(f"Loaded cached item embeddings: {cache_path}")
            return z["features"].astype(np.float32)
    from sentence_transformers import SentenceTransformer

    print(f"Encoding {len(texts)} hotels with {ENCODER_NAME} on {device}")
    encoder = SentenceTransformer(ENCODER_NAME, device=device)
    features = encoder.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    del encoder
    np.savez_compressed(
        cache_path, features=features, n=np.int32(len(texts)),
        encoder=np.array(ENCODER_NAME),
    )
    return features


def cf_item_embeddings(n_users: int, n_items: int, dim: int = 32) -> np.ndarray:
    """Train-only truncated SVD item factors for LETTER's collaborative loss."""
    tr, _, _, _ = rb.load("val")
    mat = rb.build_matrix(tr, n_users, n_items)
    k = int(min(dim, max(1, min(mat.shape) - 1)))
    # sklearn is already a LETTER/RPG dependency.
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=k, random_state=SEED)
    return svd.fit_transform(mat.T).astype(np.float32)


def _metric_row(result) -> dict:
    avg = result.metric_avg_results
    row = {}
    for key in METRIC_ORDER:
        if key in avg:
            row[key] = round(float(avg[key]), 4)
        else:
            row[key] = float("nan")
    if "MAP" in avg:
        row["MAP"] = round(float(avg["MAP"]), 4)
    row["Train (s)"] = round(float(avg.get("Train (s)", float("nan"))), 1)
    row["Test (s)"] = round(float(avg.get("Test (s)", float("nan"))), 1)
    return row


def _user_r10(result, uid_map: dict, n_users: int) -> np.ndarray:
    """Per-user Recall@10 aligned to integer userID in ``[0, n_users)``."""
    raw = result.metric_user_results.get("Recall@10", {})
    vec = np.full(n_users, np.nan, dtype=np.float64)
    inv = {idx: int(uid) for uid, idx in uid_map.items()}
    for idx, scores in raw.items():
        uid = inv.get(int(idx))
        if uid is None or uid < 0 or uid >= n_users:
            continue
        vec[uid] = float(np.mean(scores)) if scores else np.nan
    return vec


def _user_n10(result, uid_map: dict, n_users: int) -> np.ndarray:
    raw = result.metric_user_results.get("NDCG@10", {})
    vec = np.full(n_users, np.nan, dtype=np.float64)
    inv = {idx: int(uid) for uid, idx in uid_map.items()}
    for idx, scores in raw.items():
        uid = inv.get(int(idx))
        if uid is None or uid < 0 or uid >= n_users:
            continue
        vec[uid] = float(np.mean(scores)) if scores else np.nan
    return vec


def build_eval_method(features: np.ndarray, item_ids: list[str], verbose: bool):
    import cornac
    from cornac.data import FeatureModality
    from cornac.eval_methods import NextItemEvaluation

    train_data, val_data, test_data, n_seq_users = frozen_usit_splits()
    print(
        f"USIT tuples: train={len(train_data)} val={len(val_data)} "
        f"test={len(test_data)} users={n_seq_users}"
    )
    eval_method = NextItemEvaluation.from_splits(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        fmt="USIT",
        exclude_unknowns=True,
        verbose=verbose,
        seed=SEED,
        mode="last",
        item_feature=FeatureModality(features=features, ids=item_ids),
    )
    return eval_method, cornac, n_seq_users


def _metrics():
    from cornac.metrics import MAP, MRR, NDCG, Precision, Recall

    return [
        MAP(), MRR(),
        Precision(k=5), Recall(k=5), NDCG(k=5),
        Precision(k=10), Recall(k=10), NDCG(k=10),
    ]


def _inject_cholmod_stub() -> None:
    """SANSA imports ``sksparse.cholmod`` even for ICF (COLAMD permutation).

    SuiteSparse headers are not available on this Windows env, so CHOLMOD
    numeric factorization cannot run. With 535 items an identity permutation
    is a faithful substitute for COLAMD.
    """
    if "sksparse.cholmod" in sys.modules:
        return
    import types

    class _Factor:
        def __init__(self, n: int):
            self._n = n

        def P(self):
            return np.arange(self._n, dtype=np.int64)

        def L(self):
            raise RuntimeError("CHOLMOD numeric factor unavailable")

        def cholesky_AAt_inplace(self, *args, **kwargs):
            raise RuntimeError("CHOLMOD numeric factor unavailable")

        def cholesky_inplace(self, *args, **kwargs):
            raise RuntimeError("CHOLMOD numeric factor unavailable")

    def _factor(A, **_kwargs):
        return _Factor(int(A.shape[0]))

    cholmod = types.ModuleType("sksparse.cholmod")
    cholmod.analyze_AAt = _factor
    cholmod.analyze = _factor
    pkg = types.ModuleType("sksparse")
    pkg.cholmod = cholmod
    sys.modules["sksparse"] = pkg
    sys.modules["sksparse.cholmod"] = cholmod


def _build_sansa_dataset(tr: pd.DataFrame, n_users: int, n_items: int):
    from cornac.data import Dataset

    uid_map = OrderedDict((str(i), i) for i in range(n_users))
    iid_map = OrderedDict((str(i), i) for i in range(n_items))
    data = [
        (str(int(u)), str(int(i)), 1.0)
        for u, i in zip(tr.userID.to_numpy(), tr.itemID.to_numpy())
    ]
    return Dataset.build(
        data, fmt="UIR", global_uid_map=uid_map, global_iid_map=iid_map, seed=SEED,
    )


def score_sansa_dense(mat: np.ndarray, lam: float, use_abs: bool = False) -> np.ndarray:
    """Exact dense SANSA / NSAE (CHOLMOD + density=1 limit of Spišák et al.).

    535 items: the sparse ICF/CHOLMOD approximation is unnecessary; this is
    the closed-form item–item map Cornac SANSA constructs before pruning.
    """
    x = mat.astype(np.float64)
    n = x.shape[1]
    gram = x.T @ x
    gram.flat[:: n + 1] += float(lam)
    l_chol = np.linalg.cholesky(gram)
    d = np.diag(l_chol).copy()
    d[d == 0.0] = 1e-12
    unit_l = l_chol / d[np.newaxis, :]
    np.fill_diagonal(unit_l, 1.0)
    d2 = d ** 2
    w = np.linalg.inv(unit_l)
    w_r = w / d2[:, np.newaxis]
    diag = ((w ** 2) / d2[:, np.newaxis]).sum(axis=0)
    diag[diag == 0.0] = 1e-12
    w_r = w_r * (-1.0 / diag)[np.newaxis, :]
    scores = (x @ w.T @ w_r).astype(np.float32)
    if use_abs:
        scores = np.abs(scores)
    return scores


def _fit_sansa_library(train_set, lam: float, density: float):
    _inject_cholmod_stub()
    from cornac.models import SANSA

    model = SANSA(
        name="SANSA",
        l2=float(lam),
        weight_matrix_density=float(density),
        compute_gramian=True,
        factorizer_class="ICF",
        use_absolute_value_scores=False,
        verbose=False,
        seed=SEED,
    )
    model.fit(train_set)
    raw = model.forward(model.X)
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    scores = np.asarray(raw, dtype=np.float32)
    if model.use_absolute_value_scores:
        scores = np.abs(scores)
    return scores


def run_sansa(mat, val_items, test_items, tr, n_users, n_items, smoke: bool):
    """Tune λ on val, score test once.

    Cornac SANSA ICF is the 150k-item sparse path. On this 535-item catalog
    X^T X is ~20% dense, so ICF is unstable (it keeps bumping λ). The
    RecSys'23 paper uses exact CHOLMOD at this scale; SuiteSparse is not
    available on Windows here, so we run the same closed-form LDL as
    ``recom_sansa.py`` with a dense Cholesky (density=1, no pruning).
    """
    del tr  # Dataset not needed for the dense path
    lambdas = (100.0, 1000.0) if smoke else SANSA_LAMBDAS
    t0 = time.perf_counter()
    best = {"val_Recall@10": -1.0, "backend": "dense-exact LDL"}
    best_scores = None
    for lam in lambdas:
        scores = score_sansa_dense(mat, lam)
        val_r = rb.evaluate(scores, mat, val_items)["Recall@10"]
        print(f"  SANSA l2={lam:g} dense-exact val R@10={val_r:.4f}", flush=True)
        if val_r > best["val_Recall@10"]:
            best = {
                "l2": float(lam),
                "backend": "dense-exact LDL (CHOLMOD density=1 analogue)",
                "val_Recall@10": float(val_r),
                "library_icf": False,
                "note": (
                    "Cornac SANSA ICF skipped: 535-item gramian is 19.5% dense; "
                    "ICF repeatedly fails and inflates λ. Exact dense LDL is "
                    "the small-catalog SANSA/NSAE closed form."
                ),
            }
            best_scores = scores
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    test_row = rb.evaluate(best_scores, mat, test_items)
    test_s = time.perf_counter() - t1
    test_row["Train (s)"] = round(train_s, 1)
    test_row["Test (s)"] = round(test_s, 1)
    rec, ndcg, _, users = es.user_metric_vectors(best_scores, mat, test_items)
    r10 = np.full(n_users, np.nan)
    n10 = np.full(n_users, np.nan)
    for i, u in enumerate(users):
        r10[int(u)] = rec[i]
        n10[int(u)] = ndcg[i]
    print(
        f"  SANSA pack check: evaluate R@10={test_row['Recall@10']} "
        f"vec_mean={float(rec.mean()):.4f} aligned={float(np.nanmean(r10)):.4f} "
        f"n={len(users)}/{n_users}",
        flush=True,
    )
    return test_row, r10, n10, best


def instantiate_models(names, device, features_dim, cf_emb, item_ids, smoke: bool, n_items: int):
    """Return ``(name -> factory)`` so a failed model does not keep GPU/CPU state."""
    n_epochs = 1 if smoke else None
    factories = {}

    if "tiger" in names:
        def make_tiger():
            from cornac.models import TIGER
            from cornac.models.tiger import GRID_CONFIG

            cfg = dict(GRID_CONFIG)
            if n_epochs is not None:
                cfg["n_epochs"] = n_epochs
                cfg["val_eval_every"] = 1
            elif device == "cpu":
                # GRID uses val every epoch; on CPU that doubles wall time
                # (~2 min ranking × 50). Every 5 epochs still picks best-on-val.
                cfg["val_eval_every"] = 5
            return TIGER(
                **{
                    **cfg,
                    "max_len": 20,
                    "n_beams": 20,
                    "device": device,
                    "verbose": True,
                    "seed": SEED,
                }
            )

        factories["TIGER"] = make_tiger

    if "letter" in names:
        def make_letter():
            from cornac.models import LETTER

            # Constrained k-means requires n_catalog // (K*2) >= 1. The
            # sequential train catalog is 508 items (not 535 in item_map),
            # so Beauty's K=256 fails (508//512=0). K=128 → 508//256=1.
            codebook = 128 if n_items < 1024 else 256
            n_clusters = 8 if codebook < 256 else 10
            return LETTER(
                cf_embeddings=cf_emb,
                cf_embedding_ids=item_ids,
                cf_weight=0.02,
                diversity_weight=1e-3,
                n_clusters=n_clusters,
                rqvae_num_levels=4,
                rqvae_codebook_size=codebook,
                rqvae_latent_dim=32,
                rqvae_n_epochs=2 if smoke else 200,
                rqvae_kmeans_jobs=1,
                n_epochs=n_epochs or 50,
                batch_size=256,
                max_len=20,
                scoring="beam",
                n_beams=20,
                model_selection="best",
                val_metric="ndcg",
                val_k=10,
                val_eval_every=1 if smoke else 5,
                device=device,
                verbose=True,
                seed=SEED,
            )

        factories["LETTER"] = make_letter

    if "rpg" in names:
        def make_rpg():
            from cornac.models import RPG

            # pca_dim=512 is a no-op on 384-d MiniLM (RPG skips PCA when
            # pca_dim >= embedding dim). 384 is divisible by 32, so OPQ32 is valid.
            return RPG(
                n_codebook=32,
                codebook_size=256,
                pca_dim=max(features_dim, 32),
                n_epochs=n_epochs or 50,
                batch_size=256,
                max_len=20,
                scoring="exact",
                n_beams=20,
                model_selection="best",
                val_metric="ndcg",
                val_k=10,
                val_eval_every=1 if smoke else 5,
                early_stopping_patience=None if smoke else 10,
                device=device,
                verbose=True,
                seed=SEED,
            )

        factories["RPG"] = make_rpg

    if "diffgrm" in names:
        def make_diffgrm():
            from cornac.models import DiffGRM
            from cornac.models.diffgrm import DIFFGRM_SPORTS_CONFIG

            cfg = dict(DIFFGRM_SPORTS_CONFIG)
            cfg.update({
                "view_loss_reduction": "token_mean",
                "scoring": "released",
                "device": device,
                "verbose": True,
                "seed": SEED,
                "max_len": 20,
            })
            if n_epochs is not None:
                cfg["n_epochs"] = n_epochs
                cfg["batch_size"] = min(int(cfg.get("batch_size", 64)), 64)
                cfg["beam_size"] = 8
            elif device == "cpu":
                cfg["n_epochs"] = 15
                cfg["batch_size"] = 64
                cfg["d_model"] = 64
                cfg["beam_size"] = 16
                cfg["val_eval_every"] = 5
            return DiffGRM(**cfg)

        factories["DiffGRM"] = make_diffgrm

    if "sasrec" in names:
        def make_sasrec():
            from cornac.models import SASRec

            return SASRec(
                embedding_dim=64,
                n_epochs=n_epochs or 50,
                batch_size=256,
                max_len=20,
                model_selection="best",
                val_eval_every=1 if smoke else 5,
                val_k=10,
                val_metric="recall",
                device=device,
                verbose=True,
                seed=SEED,
            )

        factories["SASRec"] = make_sasrec

    if "bert4rec" in names:
        def make_bert4rec():
            from cornac.models import BERT4Rec

            return BERT4Rec(
                embedding_dim=64,
                n_epochs=n_epochs or 50,
                batch_size=256,
                max_len=20,
                model_selection="best",
                val_eval_every=1 if smoke else 5,
                val_k=10,
                val_metric="recall",
                device=device,
                verbose=True,
                seed=SEED,
            )

        factories["BERT4Rec"] = make_bert4rec

    return factories


def run_one(name, factory, eval_method, cornac, metrics, n_users):
    model = factory()
    exp = cornac.Experiment(
        eval_method=eval_method,
        models=[model],
        metrics=metrics,
        user_based=True,
        verbose=True,
    )
    exp.run()
    result = exp.result[0]
    row = _metric_row(result)
    r10 = _user_r10(result, eval_method.global_uid_map, n_users)
    n10 = _user_n10(result, eval_method.global_uid_map, n_users)
    return row, r10, n10


def _write_outputs(rows: dict, skipped: dict, per_user: dict, meta: dict) -> None:
    ref_path = OUT / "per_user_metrics.csv"
    ci_rows, test_rows = [], []
    user_cols = {}
    if rows:
        table = pd.DataFrame(rows).T
        table.index.name = "Method"
        cols = [c for c in METRIC_ORDER if c in table.columns] + [
            c for c in table.columns if c not in METRIC_ORDER
        ]
        table = table[cols]
        table.to_csv(OUT / "cornac_sota_results.csv")
        print("\nSaved", OUT / "cornac_sota_results.csv")
        print(table.to_string())
    else:
        table = pd.DataFrame()

    if ref_path.exists() and per_user:
        ref = pd.read_csv(ref_path)
        ref_rec = ref["UserKNN-cosine_R@10"].to_numpy(dtype=np.float64)
        ref_users = ref["userID"].to_numpy(dtype=int)
        for name, (rec, ndcg) in per_user.items():
            rec_al = rec[ref_users]
            ndcg_al = ndcg[ref_users]
            ok = np.isfinite(rec_al)
            rec_al, ndcg_al = rec_al[ok], ndcg_al[ok]
            r_ci = es.bootstrap_ci(rec_al)
            n_ci = es.bootstrap_ci(ndcg_al)
            vs = es.wilcoxon_paired(rec_al, ref_rec[ok])
            star = es.stars(vs["p"])
            ci_rows.append({
                "Method": name,
                "Recall@10": r_ci["mean"],
                "Recall@10_ci_lo": r_ci["ci_lo"],
                "Recall@10_ci_hi": r_ci["ci_hi"],
                "NDCG@10": n_ci["mean"],
                "NDCG@10_ci_lo": n_ci["ci_lo"],
                "NDCG@10_ci_hi": n_ci["ci_hi"],
                "p_vs_UserKNN_R@10": vs["p"],
                "sig_vs_UserKNN": star,
                "wilcoxon_n_nonzero": vs["n_nonzero"],
                "mean_diff_vs_UserKNN": vs["mean_diff"],
            })
            test_rows.append({
                "comparison": f"{name} vs UserKNN-cosine",
                "metric": "Recall@10",
                **vs,
                "stars": star,
            })
            user_cols[f"{name}_R@10"] = rec[ref_users]
            user_cols[f"{name}_N@10"] = ndcg[ref_users]
        if ci_rows:
            pd.DataFrame(ci_rows).to_csv(OUT / "cornac_sota_ci.csv", index=False)
        if user_cols:
            extra = pd.DataFrame({"userID": ref_users, **user_cols})
            extra.to_csv(OUT / "cornac_sota_per_user.csv", index=False)

    payload = {
        "written_at": _now(),
        "encoder": ENCODER_NAME,
        "seed": SEED,
        "selection_fold": "val",
        "report_fold": "test",
        "protocol": (
            "Frozen public split. NextItemEvaluation.from_splits USIT, "
            "mode=last, exclude_unknowns=True. Test history includes the "
            "val interaction (Cornac leave-last-out prefixes); val is not "
            "used as a training label."
        ),
        "results": rows,
        "skipped": skipped,
        "ci": ci_rows,
        "pairwise_vs_userknn": test_rows,
        "meta": meta,
    }
    (REPORTS / "cornac_sota.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(table, skipped, OUT / "cornac_sota.md", ci_rows)


def _write_markdown(table: pd.DataFrame, skipped: dict, path, ci_rows: list | None = None) -> None:
    lines = [
        "# Cornac generative / CF baselines (public three-way split)",
        "",
        "Fit on `train.csv`. Best-on-val checkpoint when the model supports it. "
        "`test.csv` scored once. Item text = hotel name + city + Facilities/"
        "Around/Vicinity/Price, encoded with "
        f"`{ENCODER_NAME}`.",
        "",
        "Sequential prefixes follow Cornac leave-last-out: the validation "
        "review is in the test-time history but is not a training label. "
        "That is one extra observed item relative to GRU4Rec in "
        "`run_neural_baselines.py` (train history only).",
        "",
        "SANSA is item–item CF (RecSys 2023). It uses the same train-only "
        "history and seen-item mask as EASE / UserKNN. SuiteSparse/CHOLMOD "
        "cannot build on this Windows env, and Cornac SANSA ICF is unstable "
        "on the 19.5%-dense 535-item gramian (the 150k-item sparse path). "
        "Reported SANSA is the exact dense LDL — the CHOLMOD density=1 "
        "closed form of Spišák et al. at this catalog size. Val-tuned λ=1000.",
        "",
    ]
    if not table.empty:
        lines += [
            "| Method | MRR | NDCG@5 | Recall@5 | NDCG@10 | Recall@10 | Train (s) | Test (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method, row in table.iterrows():
            def _fmt(key):
                v = row.get(key, float("nan"))
                return "—" if pd.isna(v) else f"{float(v):.4f}"

            def _secs(key):
                v = row.get(key, float("nan"))
                return "—" if pd.isna(v) else f"{float(v):.0f}"

            lines.append(
                f"| {method} | {_fmt('MRR')} | {_fmt('NDCG@5')} | {_fmt('Recall@5')} | "
                f"{_fmt('NDCG@10')} | {_fmt('Recall@10')} | {_secs('Train (s)')} | "
                f"{_secs('Test (s)')} |"
            )
        lines.append("")
    if ci_rows:
        lines += [
            "95% bootstrap CI over users; Wilcoxon on per-user Recall@10 vs UserKNN "
            "(R@10 = 0.1291). \\* p<0.05.",
            "",
            "| Method | R@10 | 95% CI | N@10 | vs UserKNN |",
            "|---|---:|---|---:|---|",
        ]
        for row in ci_rows:
            p = row.get("p_vs_UserKNN_R@10")
            star = row.get("sig_vs_UserKNN") or ""
            p_s = "—" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{float(p):.4f}{star}"
            lines.append(
                f"| {row['Method']} | {row['Recall@10']:.4f} | "
                f"[{row['Recall@10_ci_lo']:.4f}, {row['Recall@10_ci_hi']:.4f}] | "
                f"{row['NDCG@10']:.4f} | {p_s} |"
            )
        lines += [
            "",
            "Reference (same freeze, train-only history): UserKNN 0.1291, Hybrid-fixed 0.1353, "
            "GRU4Rec 0.1604, EASE λ=1000 0.1203.",
            "",
        ]
    if skipped:
        lines += ["## Skipped", ""]
        for name, reason in skipped.items():
            lines.append(f"- **{name}**: {reason}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        default="tiger,letter,rpg",
        help="Comma list: tiger,letter,rpg,diffgrm,sansa,sasrec,bert4rec",
    )
    p.add_argument(
        "--try-diffgrm",
        action="store_true",
        help="Include DiffGRM even on CPU (reduced budget; not paper-scale).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch per model to verify the pipeline.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} smoke={args.smoke}")

    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    skipped = dict(SKIP_REASONS)
    if device == "cuda" or args.try_diffgrm:
        skipped.pop("DiffGRM", None)
        if "diffgrm" not in wanted and args.try_diffgrm:
            wanted.append("diffgrm")
        if device == "cuda" and "diffgrm" not in wanted:
            wanted.append("diffgrm")
    elif "diffgrm" in wanted:
        print("DiffGRM requested on CPU: running reduced budget (--try-diffgrm implied).")
        skipped.pop("DiffGRM", None)

    tr, va, n_users, n_items = rb.load("val")
    _, te, _, _ = rb.load("test")
    mat = rb.build_matrix(tr, n_users, n_items)
    val_items = dict(zip(va.userID, va.itemID))
    test_items = dict(zip(te.userID, te.itemID))
    want_sansa = "sansa" in wanted
    seq_wanted = [m for m in wanted if m != "sansa"]
    name_map = {
        "tiger": "TIGER", "letter": "LETTER", "rpg": "RPG", "diffgrm": "DiffGRM",
        "sasrec": "SASRec", "bert4rec": "BERT4Rec", "sansa": "SANSA",
    }
    rerun = {name_map[m] for m in wanted if m in name_map}

    rows, per_user, meta = {}, {}, {
        "device": device,
        "smoke": args.smoke,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": int(len(tr)),
        "encoder": ENCODER_NAME,
        "seed": SEED,
    }
    prev_csv = OUT / "cornac_sota_results.csv"
    prev_users = OUT / "cornac_sota_per_user.csv"
    if prev_csv.exists() and not args.smoke:
        old = pd.read_csv(prev_csv, index_col=0)
        for name, rec in old.iterrows():
            if name in rerun:
                continue
            rows[name] = {
                k: (None if pd.isna(v) else (float(v) if k != "Method" else v))
                for k, v in rec.items()
            }
        print("Kept previous rows:", list(rows))
    if prev_users.exists() and not args.smoke:
        pu = pd.read_csv(prev_users)
        uids = pu["userID"].to_numpy(dtype=int)
        for name in list(rows):
            rcol, ncol = f"{name}_R@10", f"{name}_N@10"
            if rcol not in pu.columns:
                continue
            rec = np.full(n_users, np.nan)
            nd = np.full(n_users, np.nan)
            rec[uids] = pu[rcol].to_numpy(dtype=np.float64)
            nd[uids] = pu[ncol].to_numpy(dtype=np.float64)
            per_user[name] = (rec, nd)
    _write_outputs(rows, skipped, per_user, meta)

    if want_sansa:
        print("\n========== SANSA ==========", flush=True)
        try:
            row, r10, n10, info = run_sansa(
                mat, val_items, test_items, tr, n_users, n_items, args.smoke,
            )
            rows["SANSA"] = row
            per_user["SANSA"] = (r10, n10)
            meta["sansa"] = info
            skipped.pop("SANSA", None)
            print(
                f"SANSA test R@10={row['Recall@10']} N@10={row['NDCG@10']} "
                f"l2={info.get('l2')} {info.get('backend')} val={info.get('val_Recall@10')}",
                flush=True,
            )
        except Exception as exc:
            skipped["SANSA"] = f"Runtime error: {type(exc).__name__}: {exc}"
            meta.setdefault("errors", {})["SANSA"] = traceback.format_exc()
            print(f"SANSA FAILED: {exc}", flush=True)
            traceback.print_exc()
        _write_outputs(rows, skipped, per_user, meta)

    if not seq_wanted:
        print("\nDone. Report:", REPORTS / "cornac_sota.json")
        return

    texts, n_meta = item_texts(n_items)
    print(f"items={n_items} with_attribute_metadata={n_meta}/{n_items}")
    features = encode_items(texts, device)
    if device == "cuda":
        torch.cuda.empty_cache()
    item_ids = [str(i) for i in range(n_items)]
    cf_emb = cf_item_embeddings(n_users, n_items)
    eval_method, cornac, n_seq_users = build_eval_method(
        features, item_ids, verbose=True
    )
    metrics = _metrics()
    factories = instantiate_models(
        seq_wanted, device, features.shape[1], cf_emb, item_ids, args.smoke,
        getattr(eval_method, "total_items", n_items),
    )
    meta.update({
        "n_seq_users": n_seq_users,
        "content_metadata_items": n_meta,
        "feature_dim": int(features.shape[1]),
        "cf_dim": int(cf_emb.shape[1]),
        "cornac": getattr(cornac, "__version__", "unknown"),
    })

    for name, factory in factories.items():
        print(f"\n========== {name} ==========", flush=True)
        try:
            row, r10, n10 = run_one(
                name, factory, eval_method, cornac, metrics, n_users
            )
            rows[name] = row
            per_user[name] = (r10, n10)
            skipped.pop(name, None)
            print(f"{name} Recall@10={row.get('Recall@10')} NDCG@10={row.get('NDCG@10')}",
                  flush=True)
        except Exception as exc:
            skipped[name] = f"Runtime error: {type(exc).__name__}: {exc}"
            meta.setdefault("errors", {})[name] = traceback.format_exc()
            print(f"{name} FAILED: {exc}", flush=True)
            traceback.print_exc()
        _write_outputs(rows, skipped, per_user, meta)

    print("\nDone. Report:", REPORTS / "cornac_sota.json")


if __name__ == "__main__":
    main()
