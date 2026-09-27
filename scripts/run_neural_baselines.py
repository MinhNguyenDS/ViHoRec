"""Torch neural baselines on the ViHoRec three-way public split.

LightGCN, NeuMF, and GRU4Rec train on ``train.csv`` only. Any searchable
setting (learning rate, early-stop epoch) is chosen on ``val.csv``; ``test.csv``
is scored once. Seen training items are masked at ranking time, matching
``run_baselines.py``.

GRU4Rec is the sequential attempt required by the revision (SASRec-class).
With train histories of length 2–10 it may be unstable; that outcome is a
result, not a crash.

Run:  python run_neural_baselines.py
Requires: torch (CPU is enough at this scale).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
import eval_stats as es
import run_baselines as rb

OUT = C.OUT_RELEASE / "benchmark"
SEED = 42
DEVICE = torch.device("cpu")


def _seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_numpy(scores: torch.Tensor) -> np.ndarray:
    return scores.detach().cpu().numpy().astype(np.float32)


def build_norm_adj(tr: pd.DataFrame, n_users: int, n_items: int) -> torch.Tensor:
    u = tr.userID.to_numpy()
    i = tr.itemID.to_numpy() + n_users
    src = np.concatenate([u, i])
    dst = np.concatenate([i, u])
    n = n_users + n_items
    idx = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    deg = np.bincount(np.concatenate([src, dst]), minlength=n).astype(np.float64)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-8), -0.5)
    val = (deg_inv_sqrt[src] * deg_inv_sqrt[dst]).astype(np.float32)
    return torch.sparse_coo_tensor(
        idx, torch.tensor(val), (n, n)
    ).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, adj: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        acc = x
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            acc = acc + x
        return acc / (self.n_layers + 1)

    def all_scores(self, adj: torch.Tensor) -> torch.Tensor:
        e = self.propagate(adj)
        return e[: self.n_users] @ e[self.n_users :].T


def train_lightgcn(tr, n_users, n_items, mat, val_items, lr=0.01, dim=64,
                   n_layers=3, epochs=150, patience=20, l2=1e-4):
    _seed()
    adj = build_norm_adj(tr, n_users, n_items)
    model = LightGCN(n_users, n_items, dim=dim, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    users = torch.tensor(tr.userID.to_numpy(), dtype=torch.long)
    pos = torch.tensor(tr.itemID.to_numpy(), dtype=torch.long)
    best_state, best_val, wait = None, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        neg = torch.randint(0, n_items, (len(users),))
        e = model.propagate(adj)
        u_e, i_e = e[:n_users], e[n_users:]
        pos_s = (u_e[users] * i_e[pos]).sum(1)
        neg_s = (u_e[users] * i_e[neg]).sum(1)
        bpr = -F.logsigmoid(pos_s - neg_s).mean()
        reg = l2 * (u_e[users].pow(2).mean() + i_e[pos].pow(2).mean() + i_e[neg].pow(2).mean())
        loss = bpr + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                scores = _to_numpy(model.all_scores(adj))
            r10 = rb.evaluate(scores, mat, val_items)["Recall@10"]
            if r10 > best_val:
                best_val, wait = r10, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = _to_numpy(model.all_scores(adj))
    return test_scores, {"lr": lr, "dim": dim, "n_layers": n_layers, "val_Recall@10": best_val}


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_dim=8, mlp_dims=(32, 16, 8)):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.gmf_u = nn.Embedding(n_users, gmf_dim)
        self.gmf_i = nn.Embedding(n_items, gmf_dim)
        self.mlp_u = nn.Embedding(n_users, mlp_dims[0] // 2)
        self.mlp_i = nn.Embedding(n_items, mlp_dims[0] // 2)
        layers = []
        in_d = mlp_dims[0]
        for d in mlp_dims[1:]:
            layers += [nn.Linear(in_d, d), nn.ReLU()]
            in_d = d
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(gmf_dim + mlp_dims[-1], 1)
        for emb in (self.gmf_u, self.gmf_i, self.mlp_u, self.mlp_i):
            nn.init.normal_(emb.weight, std=0.1)

    def score_ui(self, u, i):
        gmf = self.gmf_u(u) * self.gmf_i(i)
        mlp = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)

    def all_scores(self) -> torch.Tensor:
        # 798 x 535 is small enough to score densely.
        u = torch.arange(self.n_users)
        i = torch.arange(self.n_items)
        uu = u.repeat_interleave(self.n_items)
        ii = i.repeat(self.n_users)
        s = self.score_ui(uu, ii)
        return s.view(self.n_users, self.n_items)


def train_neumf(tr, n_users, n_items, mat, val_items, lr=0.001, epochs=80,
                patience=10, batch=2048):
    _seed()
    model = NeuMF(n_users, n_items)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    users = tr.userID.to_numpy()
    pos = tr.itemID.to_numpy()
    n = len(users)
    best_state, best_val, wait = None, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            u = torch.tensor(users[idx], dtype=torch.long)
            i = torch.tensor(pos[idx], dtype=torch.long)
            j = torch.randint(0, n_items, (len(idx),))
            loss = -F.logsigmoid(model.score_ui(u, i) - model.score_ui(u, j)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                scores = _to_numpy(model.all_scores())
            r10 = rb.evaluate(scores, mat, val_items)["Recall@10"]
            if r10 > best_val:
                best_val, wait = r10, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = _to_numpy(model.all_scores())
    return test_scores, {"lr": lr, "val_Recall@10": best_val}


class GRU4Rec(nn.Module):
    def __init__(self, n_items, dim=64, hidden=64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_items)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def logits(self, seq):
        x = self.item_emb(seq)
        h, _ = self.gru(x)
        last = h[:, -1]
        return self.out(last)


def _user_sequences(tr: pd.DataFrame, n_users: int, pad: int, max_len: int = 20):
    seqs = [[] for _ in range(n_users)]
    # Chronological within user: train.csv is not guaranteed sorted.
    ordered = tr.sort_values(["userID", "timestamp"] if "timestamp" in tr.columns
                             else ["userID"])
    for u, i in zip(ordered.userID.to_numpy(), ordered.itemID.to_numpy()):
        seqs[int(u)].append(int(i))
    arr = np.full((n_users, max_len), pad, dtype=np.int64)
    for u, s in enumerate(seqs):
        s = s[-max_len:]
        if s:
            arr[u, -len(s):] = s
    return arr, seqs


def train_gru4rec(tr, n_users, n_items, mat, val_items, lr=0.001, epochs=40,
                  patience=8, max_len=20):
    _seed()
    pad = n_items
    arr, seqs = _user_sequences(tr, n_users, pad, max_len=max_len)
    model = GRU4Rec(n_items)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Next-item CE on prefixes that have a next train item.
    pairs_u, pairs_seq, pairs_y = [], [], []
    for u, s in enumerate(seqs):
        if len(s) < 2:
            continue
        for t in range(1, len(s)):
            ctx = s[max(0, t - max_len):t]
            padded = [pad] * (max_len - len(ctx)) + ctx
            pairs_u.append(u)
            pairs_seq.append(padded)
            pairs_y.append(s[t])
    if not pairs_y:
        raise RuntimeError("GRU4Rec: no sequences of length >= 2")
    seq_t = torch.tensor(np.array(pairs_seq), dtype=torch.long)
    y_t = torch.tensor(np.array(pairs_y), dtype=torch.long)
    best_state, best_val, wait = None, -1.0, 0
    n = len(y_t)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, 512):
            idx = perm[start:start + 512]
            logits = model.logits(seq_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 2 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                full = torch.tensor(arr, dtype=torch.long)
                scores = _to_numpy(model.logits(full))
            r10 = rb.evaluate(scores, mat, val_items)["Recall@10"]
            if r10 > best_val:
                best_val, wait = r10, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = _to_numpy(model.logits(torch.tensor(arr, dtype=torch.long)))
    return test_scores, {"lr": lr, "val_Recall@10": best_val, "max_len": max_len}


def _select_lightgcn(tr, n_users, n_items, mat, val_items):
    """Tiny lr grid on val; each setting early-stops on val Recall@10."""
    best_scores, best_meta, best_r = None, None, -1.0
    for lr in (0.01, 0.001):
        scores, meta = train_lightgcn(
            tr, n_users, n_items, mat, val_items, lr=lr)
        if meta["val_Recall@10"] > best_r:
            best_r = meta["val_Recall@10"]
            best_scores, best_meta = scores, meta
    return best_scores, best_meta


def run() -> pd.DataFrame:
    tr, va, n_users, n_items = rb.load("val")
    _, te, _, _ = rb.load("test")
    mat = rb.build_matrix(tr, n_users, n_items)
    val_items = dict(zip(va.userID, va.itemID))
    test_items = dict(zip(te.userID, te.itemID))

    jobs = {
        "LightGCN": lambda: _select_lightgcn(tr, n_users, n_items, mat, val_items),
        "NeuMF": lambda: train_neumf(tr, n_users, n_items, mat, val_items),
        "GRU4Rec": lambda: train_gru4rec(tr, n_users, n_items, mat, val_items),
    }
    rows, meta, per_user = {}, {}, {}
    for name, fn in jobs.items():
        try:
            scores, info = fn()
            rows[name] = rb.evaluate(scores, mat, test_items)
            per_user[name] = es.user_metric_vectors(scores, mat, test_items)
            meta[name] = info
            print(f"{name}: test R@10={rows[name]['Recall@10']:.4f} val={info.get('val_Recall@10')}")
        except Exception as exc:
            meta[name] = {"error": str(exc)}
            print(f"{name} failed: {exc}")

    table = pd.DataFrame(rows).T
    if not table.empty:
        table.index.name = "Method"
        table.to_csv(OUT / "neural_results.csv")
    (C.OUT_REPORTS / "neural_baselines.json").write_text(
        json.dumps({"selection_fold": "val", "report_fold": "test",
                    "models": meta, "test": rows}, indent=2),
        encoding="utf-8")
    if not table.empty:
        print(table.to_string())
    return table


if __name__ == "__main__":
    run()
