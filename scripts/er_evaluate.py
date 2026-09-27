"""Score hotel entity resolution against human labels (Issue I2).

Consumes the annotated sheet produced by ``er_candidate_pairs.py`` and reports
what the editor asked for: precision, recall, and F1 of the canonical-key
matcher measured against a manually established ground truth, plus the
inter-annotator agreement that makes that ground truth credible.

Two recall numbers are reported, and they answer different questions:

``sample_recall``
    Recall restricted to the annotated pairs. Easy to compute, but pessimistic
    and not a property of the corpus: hard negatives were oversampled on
    purpose, so missed duplicates are over-represented.

``estimated_recall``
    Recall over all 168k candidate pairs. Every annotated negative is reweighted
    by ``pool_size / n_sampled`` for its stratum (a Horvitz-Thompson estimate),
    so the hard-negative oversampling is undone. This is the number the paper
    should quote, with the sampling design stated alongside it.

Precision needs no weighting: every pair the matcher merges is annotated.

Run:  python er_evaluate.py [--sheet annotation/er_pairs_to_annotate.csv]
Outputs: reports/er_validation.{json,md}
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys

import numpy as np
import pandas as pd

import config as C

ADJUDICATED = "adjudicated"
MIN_RATERS_FOR_GOLD = 2  # a single label is never treated as ground truth


def annotator_columns(df: pd.DataFrame) -> list[str]:
    """``annotator_1``, ``annotator_2``, ... present in the frame, in order."""
    cols = [c for c in df.columns if re.fullmatch(r"annotator_\d+", str(c))]
    return sorted(cols, key=lambda c: int(str(c).split("_")[1]))


def _clean_labels(s: pd.Series) -> pd.Series:
    """Coerce a label column to nullable 0/1, tolerating blanks and yes/no."""
    mapped = (
        s.astype("string")
        .str.strip()
        .str.lower()
        .replace({"yes": "1", "y": "1", "same": "1", "true": "1",
                  "no": "0", "n": "0", "different": "0", "false": "0",
                  "": pd.NA, "nan": pd.NA})
    )
    return pd.to_numeric(mapped, errors="coerce").astype("Int64")


def cohens_kappa(a: pd.Series, b: pd.Series) -> dict:
    """Cohen's kappa for two binary raters on their commonly labelled items."""
    both = a.notna() & b.notna()
    x, y = a[both].astype(int).to_numpy(), b[both].astype(int).to_numpy()
    n = len(x)
    if n == 0:
        return {"n": 0, "kappa": None, "percent_agreement": None}

    observed = float((x == y).mean())
    # Expected agreement under independence of the two raters' marginals.
    px1, py1 = x.mean(), y.mean()
    expected = px1 * py1 + (1 - px1) * (1 - py1)
    if expected >= 1.0:
        kappa = None  # Undefined: both raters used a single class throughout.
    else:
        kappa = (observed - expected) / (1 - expected)

    return {
        "n": int(n),
        "percent_agreement": round(observed, 4),
        "expected_agreement": round(float(expected), 4),
        "kappa": round(float(kappa), 4) if kappa is not None else None,
        "disagreements": int((x != y).sum()),
        "positive_rate_annotator_1": round(float(px1), 4),
        "positive_rate_annotator_2": round(float(py1), 4),
    }


def fleiss_kappa(labels: pd.DataFrame) -> dict:
    """Fleiss' kappa over items rated by every annotator.

    Applicable here because the columns are independent raters judging the same
    items. The earlier ViHoRec agreement study computed this statistic over
    three deterministic *criteria*, which is why it was uninterpretable and was
    withdrawn; see ``annotation/superseded/README.md``.
    """
    complete = labels.dropna()
    n_items, n_raters = complete.shape
    if n_items == 0 or n_raters < 2:
        return {"n_items": int(n_items), "n_raters": int(n_raters), "kappa": None}

    counts = np.stack(
        [(complete == c).sum(axis=1).to_numpy(dtype=float) for c in (0, 1)], axis=1
    )
    # Agreement within each item, averaged over items.
    p_i = (np.square(counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.square(p_j).sum())

    kappa = None if p_e >= 1.0 else (p_bar - p_e) / (1 - p_e)
    return {
        "n_items": int(n_items),
        "n_raters": int(n_raters),
        "mean_item_agreement": round(p_bar, 4),
        "expected_agreement": round(p_e, 4),
        "kappa": round(float(kappa), 4) if kappa is not None else None,
    }


def agreement_report(df: pd.DataFrame) -> dict:
    """Pairwise Cohen's kappa, plus Fleiss' kappa once there are three raters."""
    cols = annotator_columns(df)
    labels = pd.DataFrame({c: _clean_labels(df[c]) for c in cols})
    if not cols:
        return {"n_annotators": 0, "n": 0, "kappa": None, "percent_agreement": None}

    complete = labels.dropna()
    unanimous = (
        int(complete.nunique(axis=1).eq(1).sum()) if len(complete) else 0
    )
    pairwise = {
        f"{a} vs {b}": cohens_kappa(labels[a], labels[b])
        for a, b in itertools.combinations(cols, 2)
    }
    n_labelled = int(labels.notna().any(axis=1).sum())
    needs_adjudication = int(len(complete) - unanimous)

    report = {
        "n_annotators": len(cols),
        "annotators": cols,
        "n_items_with_any_label": n_labelled,
        "n_items_rated_by_all": int(len(complete)),
        "unanimous_items": unanimous,
        "unanimous_rate": round(unanimous / len(complete), 4) if len(complete) else None,
        "items_needing_adjudication": needs_adjudication,
        "pairwise_cohens_kappa": pairwise,
    }
    if len(cols) >= 3:
        report["fleiss_kappa"] = fleiss_kappa(labels)

    # Flat aliases so single-number summaries keep working with any panel size.
    if len(cols) == 2:
        only = next(iter(pairwise.values()))
        report.update(
            n=only["n"],
            percent_agreement=only["percent_agreement"],
            kappa=only["kappa"],
            disagreements=only["disagreements"],
        )
    else:
        report.update(
            n=int(len(complete)),
            percent_agreement=report["unanimous_rate"],
            kappa=report.get("fleiss_kappa", {}).get("kappa"),
            disagreements=needs_adjudication,
        )
    return report


def resolve_gold(df: pd.DataFrame, min_raters: int = MIN_RATERS_FOR_GOLD) -> pd.Series:
    """Final label per item: adjudication where present, else a majority vote.

    A tie yields no label, so a two-annotator panel still falls through to
    adjudication exactly as before. An odd panel always produces a majority,
    but non-unanimous items are still surfaced for a human to check, because a
    2-1 split is where correlated annotator error tends to sit.
    """
    cols = annotator_columns(df)
    labels = pd.DataFrame({c: _clean_labels(df[c]) for c in cols})

    n_rated = labels.notna().sum(axis=1)
    n_ones = labels.eq(1).sum(axis=1)
    n_zeros = labels.eq(0).sum(axis=1)

    majority = pd.Series(pd.NA, index=df.index, dtype="Int64")
    enough = n_rated >= min_raters
    majority[enough & (n_ones > n_zeros)] = 1
    majority[enough & (n_zeros > n_ones)] = 0

    adj = (
        _clean_labels(df[ADJUDICATED])
        if ADJUDICATED in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    return adj.where(adj.notna(), majority)


def _packet_human_labels(*filenames: str) -> pd.Series:
    """First existing packet whose ``label`` column is actually filled."""
    for name in filenames:
        path = C.OUT_ANNOTATION / "packets" / name
        if not path.exists():
            continue
        pkt = pd.read_csv(path)
        if "label" not in pkt.columns or "pair_id" not in pkt.columns:
            continue
        labels = _clean_labels(pkt.set_index("pair_id")["label"])
        if labels.notna().any():
            return labels
    return pd.Series(dtype="Int64")


def _human_vs_panel(df: pd.DataFrame) -> dict | None:
    """Blind load-bearing verification vs the model-panel majority.

    Restricted to the verification packet so a later negative audit cannot
    inflate the agreement rate by adding easy unanimous zeros.
    """
    if ADJUDICATED not in df.columns:
        return None
    human = _clean_labels(df[ADJUDICATED])
    verify = _packet_human_labels("er_pairs_human_verification.csv")
    if verify.notna().any() and "pair_id" in df.columns:
        in_packet = df["pair_id"].astype(str).isin(verify.index.astype(str))
        human = human.where(in_packet)
    cols = annotator_columns(df)
    if not human.notna().any() or not cols:
        return None
    labels = pd.DataFrame({c: _clean_labels(df[c]) for c in cols})
    n_ones, n_zeros = labels.eq(1).sum(axis=1), labels.eq(0).sum(axis=1)
    panel = pd.Series(pd.NA, index=df.index, dtype="Int64")
    panel[n_ones > n_zeros] = 1
    panel[n_zeros > n_ones] = 0
    both = human.notna() & panel.notna()
    if not both.any():
        return None
    overturned = df.loc[both & (human != panel), "pair_id"].astype(str).tolist()
    return {
        "n_compared": int(both.sum()),
        "agreed": int((human == panel)[both].sum()),
        "agreement_rate": round(float((human == panel)[both].mean()), 4),
        "overturned_by_human": overturned,
    }


def _negative_audit(df: pd.DataFrame) -> dict | None:
    """Random audit of unanimous panel negatives vs the panel majority."""
    audit = _packet_human_labels(
        "er_pairs_negative_audited.csv",
        "er_pairs_negative_audit.csv",
    )
    if not audit.notna().any() or "pair_id" not in df.columns:
        return None
    mapped = df["pair_id"].astype(str).map(audit.rename(index=str))
    human = _clean_labels(mapped)
    cols = annotator_columns(df)
    if not cols:
        return None
    labels = pd.DataFrame({c: _clean_labels(df[c]) for c in cols})
    n_ones, n_zeros = labels.eq(1).sum(axis=1), labels.eq(0).sum(axis=1)
    panel = pd.Series(pd.NA, index=df.index, dtype="Int64")
    panel[n_ones > n_zeros] = 1
    panel[n_zeros > n_ones] = 0
    both = human.notna() & panel.notna()
    if not both.any():
        return None
    overturned = df.loc[both & (human != panel), "pair_id"].astype(str).tolist()
    n = int(both.sum())
    n_err = int((both & (human != panel)).sum())
    upper = round(3.0 / n, 4) if n_err == 0 and n else None
    return {
        "n_compared": n,
        "agreed": int((human == panel)[both].sum()),
        "n_human_positive": int((human == 1).sum()),
        "overturned_by_human": overturned,
        "rule_of_three_95_upper": upper,
    }


def label_provenance(df: pd.DataFrame, study: str = "er_pairs") -> dict:
    """Who produced each label column, and whether the result may be called gold.

    ``GUIDELINES.md`` §3.1 makes disclosure mandatory wherever the annotation is
    described, and this report is one of those places. Agreement among models
    measures task clarity, not label correctness, so a panel with no human
    verification is reported as **provisional** no matter how high its kappa is.
    """
    cols = annotator_columns(df)
    path = C.OUT_ANNOTATION / "llm_provenance.json"
    recorded = {}
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8")).get(study, {})

    slots = []
    for col in cols:
        info = recorded.get(col, {})
        kind = info.get("kind", "human")
        slots.append(
            {
                "slot": col,
                "kind": kind,
                "model": info.get("model"),
                "params": info.get("params"),
                "run_at": info.get("run_at"),
                "labelled": int(_clean_labels(df[col]).notna().sum()),
            }
        )

    n_llm = sum(1 for s in slots if s["kind"] == "llm")
    # Human labels arrive in `adjudicated`, from the tie-break sheet, the
    # blind verification packet, or the negative audit (see annotation_packets.py).
    n_human_labels = (
        int(_clean_labels(df[ADJUDICATED]).notna().sum())
        if ADJUDICATED in df.columns
        else 0
    )
    n_verified = int(_packet_human_labels("er_pairs_human_verification.csv").notna().sum())
    n_audited = int(
        _packet_human_labels(
            "er_pairs_negative_audited.csv",
            "er_pairs_negative_audit.csv",
        ).notna().sum()
    )

    if n_llm == 0:
        status, verdict = "manual", "Manual annotation."
    elif n_human_labels == 0:
        status, verdict = (
            "provisional",
            "Model-generated labels with NO human verification. Not a validated "
            "ground truth: do not cite these metrics as manual-annotation "
            "evidence. Run `annotation_packets.py verify`, have a person label "
            "the blind packet, then re-run this script.",
        )
    else:
        extra = ""
        if n_audited:
            extra = (
                f" plus a {n_audited}-row random audit of unanimous negatives"
            )
        status, verdict = (
            "llm_with_human_verification",
            f"LLM annotation with human verification of {n_verified or n_human_labels} "
            f"load-bearing rows{extra}. Describe it as such, never as manual annotation.",
        )

    return {
        "status": status,
        "verdict": verdict,
        "n_annotator_slots": len(cols),
        "n_llm_slots": n_llm,
        "n_human_resolved_labels": n_human_labels,
        "n_load_bearing_verified": n_verified,
        "n_negative_audited": n_audited,
        "slots": slots,
    }


def load_annotated(sheet: str | None) -> tuple[pd.DataFrame, dict]:
    sheet_path = C.OUT_ANNOTATION / "er_pairs_to_annotate.csv" if sheet is None else sheet
    key_path = C.OUT_ANNOTATION / "er_pairs_key.csv"
    summary_path = C.OUT_ANNOTATION / "er_pairs_summary.json"
    for p in (sheet_path, key_path, summary_path):
        if not pd.io.common.file_exists(str(p)):
            raise SystemExit(f"Missing {p}. Run er_candidate_pairs.py first.")

    sheet_df = pd.read_csv(sheet_path)
    key_df = pd.read_csv(key_path)
    pools = json.loads(open(summary_path, encoding="utf-8").read())["strata_pools"]
    return sheet_df.merge(key_df, on="pair_id", validate="one_to_one"), pools


def _f1(p, r):
    return round(2 * p * r / (p + r), 4) if p and r and (p + r) else None


def _score_matcher(labelled: pd.DataFrame, pred_col: str) -> dict:
    """Confusion matrix and metrics for one matcher against the gold labels."""
    pred = labelled[pred_col].astype(bool)
    tp = labelled[pred & (labelled["gold"] == 1)]
    fp = labelled[pred & (labelled["gold"] == 0)]
    fn = labelled[~pred & (labelled["gold"] == 1)]
    tn = labelled[~pred & (labelled["gold"] == 0)]

    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else None
    sample_recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else None

    # Population-level recall: weight false negatives back up to the full space.
    w_tp, w_fn = float(tp["weight"].sum()), float(fn["weight"].sum())
    est_recall = w_tp / (w_tp + w_fn) if (w_tp + w_fn) else None

    # The estimate is only as stable as the sampled strata it extrapolates from.
    # A single missed duplicate found in a 24x-weighted stratum moves recall far
    # more than one found in an exhaustive stratum, so the composition of the
    # weighted false-negative mass is reported alongside the point estimate.
    fn_mass = (
        fn.groupby("stratum")
        .agg(n_items=("weight", "size"), weighted_fn=("weight", "sum"))
        .assign(share=lambda d: d["weighted_fn"] / w_fn if w_fn else 0.0)
        .round({"weighted_fn": 2, "share": 4})
        .reset_index()
        .to_dict("records")
    )
    extrapolated = [r for r in fn_mass if r["share"] > 0.5 and r["n_items"] <= 3]

    return {
        "confusion": {
            "true_positives": n_tp,
            "false_positives": n_fp,
            "false_negatives": n_fn,
            "true_negatives": len(tn),
        },
        "weighted_false_negatives": round(w_fn, 2),
        "weighted_fn_by_stratum": fn_mass,
        "recall_estimate_fragile": bool(extrapolated),
        "recall_driver": extrapolated[0]["stratum"] if extrapolated else None,
        "precision": round(precision, 4) if precision is not None else None,
        "sample_recall": round(sample_recall, 4) if sample_recall is not None else None,
        "estimated_recall": round(est_recall, 4) if est_recall is not None else None,
        "f1_sample": _f1(precision, sample_recall),
        "f1_estimated": _f1(precision, est_recall),
        "false_positive_examples": fp[
            ["pair_id", "name_a", "city_a", "name_b", "city_b"]
        ].to_dict("records"),
        "false_negative_examples": fn[
            ["pair_id", "name_a", "city_a", "name_b", "city_b", "jaccard"]
        ].to_dict("records"),
    }


MATCHERS = {
    "name_only": ("merge_nameonly", "Name-only canonical key (submitted version)"),
    "resolved": ("merge_resolved", "City- and type-aware resolver (revised)"),
}


def evaluate(df: pd.DataFrame, pools: dict) -> dict:
    gold = resolve_gold(df)
    df = df.assign(gold=gold)
    labelled = df[df["gold"].notna()].copy()
    labelled["gold"] = labelled["gold"].astype(int)

    # Sampling weight: how many population pairs each annotated pair stands for.
    weights = {
        s: (pools[s]["pool_size"] / pools[s]["n_sampled"]) if pools[s]["n_sampled"] else 0.0
        for s in pools
    }
    labelled["weight"] = labelled["stratum"].map(weights).astype(float)

    provenance = label_provenance(df)
    matchers = {
        name: {"label": label, **_score_matcher(labelled, col)}
        for name, (col, label) in MATCHERS.items()
        if col in labelled.columns
    }

    # Pairs where the two matchers disagree carry the whole ablation.
    disputed = labelled[labelled["merge_nameonly"] != labelled["merge_resolved"]]

    # gold is 0/1, so sum counts the duplicates and mean gives their rate.
    per_stratum = (
        labelled.groupby("stratum")["gold"]
        .agg(n_labelled="size", n_true_duplicates="sum", duplicate_rate="mean")
        .round({"duplicate_rate": 4})
        .reset_index()
        .to_dict("records")
    )

    return {
        "human_vs_panel": _human_vs_panel(df),
        "negative_audit": _negative_audit(df),
        "coverage": {
            "n_pairs_in_sheet": int(len(df)),
            "n_labelled": int(len(labelled)),
            "n_unlabelled": int(len(df) - len(labelled)),
            "algo_merge_pairs_labelled": int(
                (labelled["stratum"] == "algo_merge").sum()
            ),
            "algo_merge_pairs_total": int(pools["algo_merge"]["pool_size"]),
        },
        "label_provenance": provenance,
        "agreement": agreement_report(df),
        "matchers": matchers,
        "stratum_weights": {k: round(v, 2) for k, v in weights.items()},
        "per_stratum": per_stratum,
        "disputed_pairs": disputed[
            [
                "pair_id", "name_a", "city_a", "name_b", "city_b",
                "merge_nameonly", "merge_resolved", "gold",
            ]
        ].to_dict("records"),
    }


_STATUS_BANNER = {
    "provisional": "PROVISIONAL — NOT A VALIDATED GROUND TRUTH",
    "llm_with_human_verification": "LLM annotation with human verification",
    "manual": "Manual annotation",
}


def to_markdown(rep: dict) -> str:
    cov, ag = rep["coverage"], rep["agreement"]
    prov = rep["label_provenance"]
    llm_panel = prov["n_llm_slots"] > 0
    agreement_heading = (
        "Agreement among annotators (model panel)"
        if llm_panel
        else "Inter-annotator agreement"
    )
    lines = [
        "# ViHoRec — Entity-Resolution Validation (Issue I2)",
        "",
        f"## Label provenance — {_STATUS_BANNER[prov['status']]}",
        "",
        prov["verdict"],
        "",
        "| Slot | Kind | Model | Labels |",
        "|---|---|---|---|",
    ]
    for s in prov["slots"]:
        lines.append(
            f"| `{s['slot']}` | {s['kind']} | {s['model'] or '—'} | {s['labelled']:,} |"
        )
    lines += [
        "",
        f"Human-resolved labels (adjudication or blind verification): "
        f"**{prov['n_human_resolved_labels']:,}**",
        "",
    ]
    hvp = rep.get("human_vs_panel")
    if hvp:
        lines += [
            "## Human vs model panel (blind verification)",
            "",
            "The human labelled the load-bearing packet without seeing model",
            "labels. This is the validity number; model κ is only task clarity.",
            "",
            f"- Compared: **{hvp['n_compared']}**",
            f"- Agreed: **{hvp['agreed']}** ({hvp['agreement_rate']})",
            f"- Overturned by human: {', '.join(hvp['overturned_by_human']) or 'none'}",
            "",
        ]
    audit = rep.get("negative_audit")
    if audit:
        lines += [
            "## Random audit of unanimous negatives",
            "",
            "Uniform sample of panel-unanimous different-hotel pairs outside the",
            "load-bearing packet. The annotator did not see model votes.",
            "",
            f"- Audited: **{audit['n_compared']}**",
            f"- Agreed with panel: **{audit['agreed']}**",
            f"- Human labelled as same-hotel: **{audit['n_human_positive']}**",
            f"- Overturned by human: {', '.join(audit['overturned_by_human']) or 'none'}",
        ]
        if audit.get("rule_of_three_95_upper") is not None:
            lines.append(
                f"- Rule-of-three 95% upper bound on residual panel error: "
                f"**{audit['rule_of_three_95_upper']}** "
                f"({audit['n_compared']} clean trials)"
            )
        lines.append("")
    lines += [
        "## Annotation coverage",
        "",
        f"- Candidate pairs in sheet: **{cov['n_pairs_in_sheet']:,}**",
        f"- Pairs with a resolved label: **{cov['n_labelled']:,}** "
        f"(unlabelled: {cov['n_unlabelled']:,})",
        f"- Algorithm merges annotated: **{cov['algo_merge_pairs_labelled']}** / "
        f"{cov['algo_merge_pairs_total']} (exhaustive => precision is unbiased)",
        "",
        f"## {agreement_heading}",
        "",
    ]
    if llm_panel:
        lines += [
            "These figures measure **agreement between models**, i.e. how well the",
            "task is specified. They are not evidence that the labels are correct,",
            "and must never be reported as inter-annotator agreement among people.",
            "",
        ]
    lines += [
        f"- Annotators: **{ag['n_annotators']}** ({', '.join(ag.get('annotators', []))})",
        f"- Pairs rated by all annotators: **{ag['n_items_rated_by_all']:,}**",
        f"- Unanimous: **{ag['unanimous_items']:,}** ({ag['unanimous_rate']})",
        f"- Sent to adjudication: **{ag['items_needing_adjudication']:,}**",
    ]
    if "fleiss_kappa" in ag:
        fk = ag["fleiss_kappa"]
        lines.append(
            f"- Fleiss' kappa ({fk['n_raters']} raters): **{fk['kappa']}**"
        )
    if ag.get("pairwise_cohens_kappa"):
        lines += [
            "",
            "| Annotator pair | Items | Percent agreement | Cohen's kappa |",
            "|---|---|---|---|",
        ]
        for pair, k in ag["pairwise_cohens_kappa"].items():
            lines.append(
                f"| {pair} | {k['n']:,} | {k['percent_agreement']} | {k['kappa']} |"
            )
    lines += [
        "",
        "## Matcher quality",
        "",
        "| Matcher | Precision | Recall (sample) | Recall (population) | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in rep["matchers"].values():
        c = m["confusion"]
        lines.append(
            f"| {m['label']} | **{m['precision']}** | {m['sample_recall']} | "
            f"**{m['estimated_recall']}** | **{m['f1_estimated']}** | "
            f"{c['true_positives']} | {c['false_positives']} | {c['false_negatives']} |"
        )
    lines += [
        "",
        "Stratum weights (population pairs per annotated pair): "
        + ", ".join(f"{k}={v}" for k, v in rep["stratum_weights"].items()),
    ]

    fragile = {n: m for n, m in rep["matchers"].items() if m["recall_estimate_fragile"]}
    if fragile:
        lines += [
            "",
            "### Stability of the population recall estimate",
            "",
            "Precision is exact: every merge is annotated. The **population recall**",
            "column is not — it extrapolates the sampled strata to all 168k pairs, so",
            "a single missed duplicate found in a heavily weighted stratum dominates",
            "it. Where that happens, quote the sample recall as the primary figure and",
            "the population recall only as an indicative lower bound.",
            "",
            "| Matcher | Stratum | Missed items found | Weighted FN | Share of FN mass |",
            "|---|---|---|---|---|",
        ]
        for m in rep["matchers"].values():
            for r in m["weighted_fn_by_stratum"]:
                lines.append(
                    f"| {m['label']} | {r['stratum']} | {int(r['n_items'])} | "
                    f"{r['weighted_fn']} | {r['share']} |"
                )

    lines += [
        "",
        "## Duplicate rate per stratum",
        "",
        "| Stratum | Labelled | True duplicates | Rate |",
        "|---|---|---|---|",
    ]
    for r in rep["per_stratum"]:
        lines.append(
            f"| {r['stratum']} | {int(r['n_labelled'])} | "
            f"{int(r['n_true_duplicates'])} | {r['duplicate_rate']} |"
        )

    if rep["disputed_pairs"]:
        lines += [
            "",
            "## Pairs where the two matchers disagree",
            "",
            "These decide the ablation: `gold = 1` favours whichever matcher merged.",
            "",
            "| Pair | Name A | City A | Name B | City B | Name-only | Resolved | Gold |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rep["disputed_pairs"]:
            lines.append(
                f"| {r['pair_id']} | {r['name_a']} | {r['city_a']} | {r['name_b']} | "
                f"{r['city_b']} | {int(bool(r['merge_nameonly']))} | "
                f"{int(bool(r['merge_resolved']))} | {int(r['gold'])} |"
            )

    for name, m in rep["matchers"].items():
        if m["false_positive_examples"]:
            lines += [
                "",
                f"## False merges — {m['label']}",
                "",
                "| Pair | Name A | City A | Name B | City B |",
                "|---|---|---|---|---|",
            ]
            for r in m["false_positive_examples"]:
                lines.append(
                    f"| {r['pair_id']} | {r['name_a']} | {r['city_a']} | "
                    f"{r['name_b']} | {r['city_b']} |"
                )
        if m["false_negative_examples"]:
            lines += [
                "",
                f"## Missed duplicates — {m['label']}",
                "",
                "| Pair | Name A | City A | Name B | City B | Jaccard |",
                "|---|---|---|---|---|---|",
            ]
            for r in m["false_negative_examples"][:25]:
                lines.append(
                    f"| {r['pair_id']} | {r['name_a']} | {r['city_a']} | "
                    f"{r['name_b']} | {r['city_b']} | {r['jaccard']} |"
                )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", default=None, help="Path to the annotated sheet CSV")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df, pools = load_annotated(args.sheet)

    gold = resolve_gold(df)
    if int(gold.notna().sum()) == 0:
        cols = annotator_columns(df) or ["annotator_1", "annotator_2"]
        print(
            "No labels found yet.\n"
            f"  Sheet: {C.OUT_ANNOTATION / 'er_pairs_to_annotate.csv'}\n"
            f"  Pairs awaiting annotation: {len(df)}\n\n"
            f"Fill {', '.join(repr(c) for c in cols)} with 1 (same hotel) or 0 "
            "(different hotel). The gold label is the majority vote; use "
            "'adjudicated' to override it on non-unanimous rows.\n"
            "Protocol: annotation/GUIDELINES.md, section 2."
        )
        raise SystemExit(0)

    rep = evaluate(df, pools)
    (C.OUT_REPORTS / "er_validation.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = to_markdown(rep)
    (C.OUT_REPORTS / "er_validation.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
