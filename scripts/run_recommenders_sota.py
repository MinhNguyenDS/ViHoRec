"""Recommenders-team baselines on the ViHoRec public freeze.

Microsoft / recommenders-team/recommenders does not ship 2025–2026 research
models (latest NEWS: April 2025). The newest *implementations* on ``main`` are
the PyTorch UniRec ports (SASRec / SSEPT, PR merged Feb 2026) plus the
PyTorch NCF / LightGCN stack.

This script evaluates three of those on the same protocol as
``run_neural_baselines.py``:

    train.csv only for fit
    val.csv for epoch / HP selection
    test.csv scored once
    full catalogue, mask train items, train-only history (not Cornac LLOO)

Skipped from the same library (written to the report):

    LightGCN  — already in neural_results.csv (R@10=0.1454); DeepRec yaml
    SUM / SLi-Rec / NextItNet — TensorFlow DeepRec sequential, not the 2026
                                PyTorch line
    NRMS / NAML / LSTUR — news recommenders; need article text

Run::

    python -u run_recommenders_sota.py
    python -u run_recommenders_sota.py --smoke --models sasrec
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
import eval_stats as es
import run_baselines as rb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT = C.OUT_RELEASE / "benchmark"
REPORTS = C.OUT_REPORTS
SEED = 42
METRIC_ORDER = [
    "MRR", "MAP@5", "NDCG@5", "Precision@5", "Recall@5",
    "MAP@10", "NDCG@10", "Precision@10", "Recall@10",
]
SKIP_REASONS = {
    "LightGCN": (
        "Already evaluated in-repo (neural_results.csv, test R@10=0.1454). "
        "Recommenders LightGCN is the same 2020 model on the DeepRec yaml stack."
    ),
    "SUM": (
        "TensorFlow DeepRec sequential (Lian et al., 2021). Not on the 2026 "
        "PyTorch UniRec line; extra TF dependency on Python 3.13."
    ),
    "SLi-Rec": (
        "TensorFlow DeepRec (Microsoft, 2019). Same TF stack as SUM."
    ),
    "NextItNet": (
        "TensorFlow DeepRec dilated CNN (Yuan et al., 2019)."
    ),
    "NRMS/NAML/LSTUR": (
        "News recommenders; they need article text. ViHoRec has hotel metadata, "
        "not news bodies."
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _user_sequences(tr: pd.DataFrame, n_users: int) -> list[list[int]]:
    seqs: list[list[int]] = [[] for _ in range(n_users)]
    for uid, grp in tr.groupby("userID", sort=True):
        ordered = grp.sort_values("timestamp", kind="mergesort")
        seqs[int(uid)] = [int(x) for x in ordered.itemID.to_numpy()]
    return seqs


class InProcessSampler:
    """WarpSampler without multiprocessing (Windows-safe)."""

    def __init__(self, user_train, usernum, itemnum, batch_size, maxlen, seed=SEED):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.rng = np.random.RandomState(seed)
        self.users = [u for u in range(1, usernum + 1) if len(user_train.get(u, [])) > 1]
        if not self.users:
            raise RuntimeError("No user has more than one training interaction.")

    def _random_neq(self, left, right, seen):
        t = self.rng.randint(left, right)
        while t in seen:
            t = self.rng.randint(left, right)
        return t

    def _sample(self):
        user = int(self.users[self.rng.randint(0, len(self.users))])
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.user_train[user][-1]
        idx = self.maxlen - 1
        ts = set(self.user_train[user])
        for i in reversed(self.user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = self._random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return user, seq, pos, neg

    def next_batch(self):
        batch = [self._sample() for _ in range(self.batch_size)]
        return tuple(zip(*batch))

    def close(self):
        return


def _frozen_sasrec_dataset(seqs, n_users, n_items, val_items, test_items):
    """1-indexed SASRec dataset from the published split (no re-split)."""
    user_train, user_valid, user_test = {}, {}, {}
    for u in range(n_users):
        uid = u + 1
        user_train[uid] = [i + 1 for i in seqs[u]]
        user_valid[uid] = [int(val_items[u]) + 1]
        user_test[uid] = [int(test_items[u]) + 1]

    class _DS:
        pass

    ds = _DS()
    ds.usernum = n_users
    ds.itemnum = n_items
    ds.user_train = user_train
    ds.user_valid = user_valid
    ds.user_test = user_test
    return ds


def _score_sasrec(model, seqs, n_users, n_items, seq_max_len, users_1idx=None):
    from recommenders.models.sasrec.model import pad_sequences

    seq_1 = [[i + 1 for i in s] for s in seqs]
    padded = pad_sequences(seq_1, maxlen=seq_max_len, padding="pre", truncating="pre")
    cands = np.arange(1, n_items + 1, dtype=np.int64)
    scores = np.zeros((n_users, n_items), dtype=np.float32)
    model.eval()
    bs = 64
    for start in range(0, n_users, bs):
        sl = slice(start, min(start + bs, n_users))
        inp = {
            "input_seq": padded[sl],
            "candidate": np.tile(cands, (sl.stop - sl.start, 1)),
        }
        if users_1idx is not None:
            inp["user"] = users_1idx[sl, None]
        logits = model.predict(inp)
        scores[sl] = logits.detach().cpu().numpy().astype(np.float32)
    return scores


def train_sasrec_family(
    kind: str,
    seqs,
    n_users,
    n_items,
    mat,
    val_items,
    smoke: bool,
):
    from recommenders.models.sasrec.model import SASREC
    from recommenders.models.sasrec.ssept import SSEPT

    _seed()
    seq_max_len = 20
    hidden = 64
    num_epochs = 1 if smoke else 50
    batch_size = 128
    val_every = 1 if smoke else 5
    ds = _frozen_sasrec_dataset(seqs, n_users, n_items, val_items, val_items)
    if kind == "ssept":
        item_dim = 64
        user_dim = 16
        hid = item_dim + user_dim
        model = SSEPT(
            item_num=n_items,
            user_num=n_users,
            seq_max_len=seq_max_len,
            num_blocks=2,
            embedding_dim=item_dim,
            attention_dim=hid,
            attention_num_heads=1,
            conv_dims=[hid, hid],
            dropout_rate=0.2,
            l2_reg=0.0,
            num_neg_test=n_items,
            user_embedding_dim=user_dim,
            item_embedding_dim=item_dim,
        )
    else:
        model = SASREC(
            item_num=n_items,
            seq_max_len=seq_max_len,
            num_blocks=2,
            embedding_dim=hidden,
            attention_dim=hidden,
            attention_num_heads=1,
            conv_dims=[hidden, hidden],
            dropout_rate=0.2,
            l2_reg=0.0,
            num_neg_test=n_items,
        )
    sampler = InProcessSampler(
        ds.user_train, n_users, n_items, batch_size=batch_size,
        maxlen=seq_max_len, seed=SEED,
    )
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)
    best_state, best_val, best_epoch = None, -1.0, 0
    t0 = time.perf_counter()
    num_steps = max(1, int(len(ds.user_train) / batch_size))
    users_1idx = np.arange(1, n_users + 1, dtype=np.int64)
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for _ in range(num_steps):
            u, seq, pos, neg = sampler.next_batch()
            inputs, _ = model.create_combined_dataset(u, seq, pos, neg)
            inp = {
                key: torch.LongTensor(inputs[key]).to(device)
                for key in inputs
                if key in ("users", "input_seq", "positive", "negative")
            }
            opt.zero_grad()
            pos_logits, neg_logits, loss_mask = model(inp, training=True)
            loss = model.loss_function(pos_logits, neg_logits, loss_mask)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if epoch % val_every == 0 or epoch == num_epochs:
            scores = _score_sasrec(
                model, seqs, n_users, n_items, seq_max_len,
                users_1idx=users_1idx if kind == "ssept" else None,
            )
            val_r = rb.evaluate(scores, mat, val_items)["Recall@10"]
            print(
                f"  {kind} epoch={epoch} loss={np.mean(losses):.4f} val R@10={val_r:.4f}",
                flush=True,
            )
            if val_r > best_val:
                best_val, best_epoch = val_r, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sampler.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    test_scores = _score_sasrec(
        model, seqs, n_users, n_items, seq_max_len,
        users_1idx=users_1idx if kind == "ssept" else None,
    )
    test_s = time.perf_counter() - t1
    meta = {
        "val_Recall@10": float(best_val),
        "best_epoch": int(best_epoch),
        "seq_max_len": seq_max_len,
        "embedding_dim": hidden,
        "n_epochs": num_epochs,
        "Train (s)": round(train_s, 1),
        "Test (s)": round(test_s, 1),
        "source": "recommenders.models.sasrec (PyTorch UniRec)",
    }
    return test_scores, meta


def train_ncf(tr, n_users, n_items, mat, val_items, smoke: bool):
    from recommenders.models.ncf.ncf_singlenode import NCF

    _seed()
    epochs = 2 if smoke else 80
    patience = 2 if smoke else 10
    batch = 2048
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        model_type="NeuMF",
        n_factors=8,
        layer_sizes=[16, 8, 4],
        n_epochs=1,
        batch_size=batch,
        learning_rate=0.001,
        verbose=0,
        seed=SEED,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    bce = nn.BCELoss()
    users = torch.tensor(tr.userID.to_numpy(), dtype=torch.long)
    pos = torch.tensor(tr.itemID.to_numpy(), dtype=torch.long)
    n = len(users)
    best_state, best_val, wait, best_epoch = None, -1.0, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        neg = torch.randint(0, n_items, (n,))
        perm = torch.randperm(n)
        epoch_loss = []
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            u = torch.cat([users[idx], users[idx]])
            i = torch.cat([pos[idx], neg[idx]])
            y = torch.cat([
                torch.ones(len(idx), 1),
                torch.zeros(len(idx), 1),
            ])
            u = u.to(model.device)
            i = i.to(model.device)
            y = y.to(model.device)
            pred = model(u, i)
            loss = bce(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss.append(float(loss.item()))
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            scores = _ncf_all_scores(model, n_users, n_items)
            val_r = rb.evaluate(scores, mat, val_items)["Recall@10"]
            print(
                f"  NCF epoch={epoch} loss={np.mean(epoch_loss):.4f} val R@10={val_r:.4f}",
                flush=True,
            )
            if val_r > best_val:
                best_val, wait, best_epoch = val_r, 0, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    test_scores = _ncf_all_scores(model, n_users, n_items)
    test_s = time.perf_counter() - t1
    return test_scores, {
        "val_Recall@10": float(best_val),
        "best_epoch": int(best_epoch),
        "Train (s)": round(train_s, 1),
        "Test (s)": round(test_s, 1),
        "source": "recommenders.models.ncf.ncf_singlenode.NCF (PyTorch NeuMF)",
    }


def _ncf_all_scores(model, n_users, n_items) -> np.ndarray:
    model.eval()
    out = np.zeros((n_users, n_items), dtype=np.float32)
    with torch.no_grad():
        items = torch.arange(n_items, device=model.device)
        for u in range(n_users):
            uu = torch.full((n_items,), u, dtype=torch.long, device=model.device)
            out[u] = model(uu, items).detach().cpu().numpy().reshape(-1)
    return out


def _pack(name, scores, mat, test_items, n_users, meta):
    row = rb.evaluate(scores, mat, test_items)
    row["Train (s)"] = meta.get("Train (s)")
    row["Test (s)"] = meta.get("Test (s)")
    rec, ndcg, _, users = es.user_metric_vectors(scores, mat, test_items)
    r10 = np.full(n_users, np.nan)
    n10 = np.full(n_users, np.nan)
    for i, u in enumerate(users):
        r10[u] = rec[i]
        n10[u] = ndcg[i]
    return row, (r10, n10)


def _write_outputs(rows, skipped, per_user, meta, n_users):
    table = pd.DataFrame()
    if rows:
        table = pd.DataFrame(rows).T
        table.index.name = "Method"
        cols = [c for c in METRIC_ORDER if c in table.columns] + [
            c for c in table.columns if c not in METRIC_ORDER
        ]
        table = table[cols]
        table.to_csv(OUT / "recommenders_sota_results.csv")
        print("\nSaved", OUT / "recommenders_sota_results.csv")
        print(table.to_string())

    ci_rows, test_rows = [], []
    user_cols = {}
    ref_path = OUT / "per_user_metrics.csv"
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
            pd.DataFrame(ci_rows).to_csv(OUT / "recommenders_sota_ci.csv", index=False)
        if user_cols:
            extra = pd.DataFrame({"userID": ref_users, **user_cols})
            extra.to_csv(OUT / "recommenders_sota_per_user.csv", index=False)

    payload = {
        "written_at": _now(),
        "seed": SEED,
        "selection_fold": "val",
        "report_fold": "test",
        "library": "https://github.com/recommenders-team/recommenders",
        "protocol": (
            "Frozen public split. Train on train.csv only; val for early-stop; "
            "test scored once. Train-only history (same as GRU4Rec), full "
            "catalogue, mask seen train items."
        ),
        "results": rows,
        "skipped": skipped,
        "ci": ci_rows,
        "pairwise_vs_userknn": test_rows,
        "meta": meta,
    }
    (REPORTS / "recommenders_sota.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(table, skipped, ci_rows)


def _write_markdown(table: pd.DataFrame, skipped: dict, ci_rows: list) -> None:
    lines = [
        "# Recommenders-team baselines (public three-way split)",
        "",
        "Library: [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders). "
        "The repo does not contain 2025–2026 *papers*; NEWS last updated April 2025. "
        "The models below are the newest **implementations** on `main`: PyTorch "
        "SASRec / SSEPT (UniRec port, 2026) and PyTorch NeuMF.",
        "",
        "Protocol matches `run_neural_baselines.py`: train on `train.csv`, "
        "select epoch on `val.csv`, score `test.csv` once. Train-only history, "
        "full 535-item catalogue, mask seen train items. Recommenders' built-in "
        "sampled HR@10 (100 negatives) is **not** used.",
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
            "Reference (same freeze, train-only history): UserKNN 0.1291, "
            "GRU4Rec 0.1604, LightGCN 0.1454, NeuMF 0.1228, TIGER 0.1566.",
            "",
        ]
    if skipped:
        lines += ["## Skipped", ""]
        for name, reason in skipped.items():
            lines.append(f"- **{name}**: {reason}")
        lines.append("")
    (OUT / "recommenders_sota.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="sasrec,ssept,ncf")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    skipped = dict(SKIP_REASONS)
    tr, va, n_users, n_items = rb.load("val")
    _, te, _, _ = rb.load("test")
    mat = rb.build_matrix(tr, n_users, n_items)
    val_items = dict(zip(va.userID, va.itemID))
    test_items = dict(zip(te.userID, te.itemID))
    seqs = _user_sequences(tr, n_users)
    rows, per_user, meta = {}, {}, {
        "smoke": args.smoke,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": int(len(tr)),
        "seed": SEED,
        "torch": torch.__version__,
    }
    name_map = {"sasrec": "SASRec", "ssept": "SSEPT", "ncf": "NCF-NeuMF"}
    rerun = {name_map[m] for m in wanted if m in name_map}
    prev_csv = OUT / "recommenders_sota_results.csv"
    prev_users = OUT / "recommenders_sota_per_user.csv"
    if prev_csv.exists() and not args.smoke:
        old = pd.read_csv(prev_csv, index_col=0)
        for name, rec in old.iterrows():
            if name in rerun:
                continue
            rows[name] = {
                k: (None if pd.isna(v) else float(v))
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
    jobs = []
    if "sasrec" in wanted:
        jobs.append(("SASRec", lambda: train_sasrec_family(
            "sasrec", seqs, n_users, n_items, mat, val_items, args.smoke)))
    if "ssept" in wanted:
        jobs.append(("SSEPT", lambda: train_sasrec_family(
            "ssept", seqs, n_users, n_items, mat, val_items, args.smoke)))
    if "ncf" in wanted:
        jobs.append(("NCF-NeuMF", lambda: train_ncf(
            tr, n_users, n_items, mat, val_items, args.smoke)))
    _write_outputs(rows, skipped, per_user, meta, n_users)
    for name, fn in jobs:
        print(f"\n========== {name} ==========", flush=True)
        try:
            scores, info = fn()
            row, vecs = _pack(name, scores, mat, test_items, n_users, info)
            rows[name] = row
            per_user[name] = vecs
            meta[name] = info
            skipped.pop(name, None)
            print(f"{name} test R@10={row['Recall@10']} N@10={row['NDCG@10']}", flush=True)
        except Exception as exc:
            skipped[name] = f"Runtime error: {type(exc).__name__}: {exc}"
            meta.setdefault("errors", {})[name] = traceback.format_exc()
            print(f"{name} FAILED: {exc}", flush=True)
            traceback.print_exc()
        _write_outputs(rows, skipped, per_user, meta, n_users)
    print("\nDone. Report:", REPORTS / "recommenders_sota.json")


if __name__ == "__main__":
    main()
