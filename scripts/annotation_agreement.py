"""Record-level validation of the ViHoRec release: automated checks + human labels.

This script replaces an earlier version whose three ``rater_*`` columns were not
raters at all: they were three deterministic checklist criteria applied by code
to the same record. Treating them as independent annotators made Fleiss' kappa
meaningless (two criteria pass every cleaned record by construction, so the
expected agreement is ~1). The two kinds of evidence are now separated.

**Automated criteria** are reproducible boolean checks. They belong in a
pass-rate table, not in an agreement statistic:

* ``C1_field_validity`` -- rating in range, date parseable and in window,
  identifiers non-empty.
* ``C2_release_consistency`` -- the record appears in the published tables with
  matching values.
* ``C3_identity_plausibility`` -- the underlying display name is neither a
  crawl-time placeholder nor too short to identify a person (cross-checked
  against the private mapping, which is never published).

**Human annotation** covers what code cannot decide: whether a catalogue entry
denotes one real, correctly located hotel. Two annotators label independently,
a third adjudicates disagreements, and Cohen's kappa is reported over the
doubly annotated items.

Usage:
    python annotation_agreement.py sample --n 250   # build the human sheet
    python annotation_agreement.py checklist        # run automated criteria
    python annotation_agreement.py score            # pass rates + kappa
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C
from er_evaluate import _clean_labels, agreement_report, annotator_columns, resolve_gold
from textnorm import is_placeholder_name

SHEET = "qc_sample_to_annotate.csv"
CRITERIA = "qc_criteria.csv"
REPORT = "qc_validation"
SHORT_NAME_CHARS = 3
DATE_LO, DATE_HI = pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")


def _release():
    inter = pd.read_csv(C.OUT_RELEASE / "interactions.csv")
    hotels = pd.read_csv(C.OUT_RELEASE / "hotels.csv")
    return inter, hotels


def _private_names() -> dict[str, str]:
    """user_id -> raw display name. Local only; never part of the release."""
    path = C.OUT_REPORTS / "_private_mapping.csv"
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    return dict(zip(m["user_id"].astype(str), m["CustomerName"].astype(str)))


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------

def cmd_sample(n: int, seed: int) -> dict:
    """Stratified sample: interactions balanced across platforms, plus hotels."""
    inter, hotels = _release()
    rng = np.random.default_rng(seed)

    n_hotels = max(1, n // 4)
    n_inter = n - n_hotels
    per_source = max(1, n_inter // inter["source"].nunique())

    picks = [
        g.sample(min(per_source, len(g)), random_state=seed)
        for _, g in inter.groupby("source")
    ]
    inter_s = pd.concat(picks, ignore_index=True)
    inter_s["record_type"] = "interaction"

    hotels_s = hotels.sample(min(n_hotels, len(hotels)), random_state=seed).copy()
    hotels_s["record_type"] = "hotel"

    sheet = pd.concat([inter_s, hotels_s], ignore_index=True)
    sheet = sheet.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "record_id", [f"R{i:04d}" for i in range(len(sheet))])

    # Only hotel rows carry a human question; interaction rows are machine-checked.
    sheet["human_task"] = np.where(
        sheet["record_type"] == "hotel",
        "Does this entry denote exactly one real hotel at this location?",
        "",
    )
    ann_cols = [f"annotator_{i}" for i in range(1, C.N_ANNOTATORS + 1)]
    for col in (*ann_cols, "adjudicated", "notes"):
        sheet[col] = ""

    out = C.OUT_ANNOTATION / SHEET
    sheet.to_csv(out, index=False, encoding="utf-8")
    return {
        "sheet": str(out),
        "n_records": int(len(sheet)),
        "n_interactions": int((sheet["record_type"] == "interaction").sum()),
        "n_hotels": int((sheet["record_type"] == "hotel").sum()),
        "human_labelled_records": int((sheet["human_task"] != "").sum()),
        "seed": seed,
    }


# --------------------------------------------------------------------------
# checklist (automated, deterministic)
# --------------------------------------------------------------------------

def cmd_checklist() -> dict:
    src = C.OUT_ANNOTATION / SHEET
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run: annotation_agreement.py sample --n 250")
    sheet = pd.read_csv(src)
    inter, hotels = _release()
    names = _private_names()

    inter_index = set(
        zip(inter["user_id"], inter["hotel_id"], inter["rating"].round(4), inter["date"])
    )
    hotel_ids = set(hotels["hotel_id"])

    rows = []
    for r in sheet.itertuples():
        rtype = r.record_type
        if rtype == "interaction":
            rating = pd.to_numeric(r.rating, errors="coerce")
            date = pd.to_datetime(r.date, errors="coerce")
            c1 = bool(
                pd.notna(rating)
                and C.RATING_MIN <= rating <= C.RATING_MAX
                and pd.notna(date)
                and DATE_LO <= date <= DATE_HI
                and str(r.user_id).strip() != ""
                and str(r.hotel_id).strip() != ""
            )
            c2 = (
                str(r.user_id),
                str(r.hotel_id),
                round(float(rating), 4) if pd.notna(rating) else None,
                str(r.date),
            ) in inter_index
            raw = names.get(str(r.user_id), "")
            c3 = bool(
                raw
                and not is_placeholder_name(raw)
                and len(raw.strip()) > SHORT_NAME_CHARS
            )
        else:
            c1 = bool(str(r.name).strip() and str(r.location).strip())
            c2 = str(r.hotel_id) in hotel_ids
            c3 = bool((inter["hotel_id"] == r.hotel_id).any())
        rows.append(
            {
                "record_id": r.record_id,
                "record_type": rtype,
                "C1_field_validity": int(c1),
                "C2_release_consistency": int(c2),
                "C3_identity_plausibility": int(c3),
            }
        )

    out = pd.DataFrame(rows)
    path = C.OUT_ANNOTATION / CRITERIA
    out.to_csv(path, index=False, encoding="utf-8")
    return {
        "criteria_file": str(path),
        "n_records": int(len(out)),
        "note": "Deterministic checks; not an inter-annotator agreement study.",
    }


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def cmd_score() -> dict:
    sheet_path, crit_path = C.OUT_ANNOTATION / SHEET, C.OUT_ANNOTATION / CRITERIA
    if not crit_path.exists():
        raise SystemExit(f"Missing {crit_path}. Run: annotation_agreement.py checklist")
    sheet, crit = pd.read_csv(sheet_path), pd.read_csv(crit_path)
    merged = sheet.merge(crit, on=["record_id", "record_type"], validate="one_to_one")

    crit_cols = ["C1_field_validity", "C2_release_consistency", "C3_identity_plausibility"]
    automated = {
        col: {
            "pass_rate": round(float(merged[col].mean()), 4),
            "failures": int((merged[col] == 0).sum()),
            "by_type": {
                str(t): round(float(g[col].mean()), 4)
                for t, g in merged.groupby("record_type")
            },
        }
        for col in crit_cols
    }
    all_pass = merged[crit_cols].all(axis=1)

    human_rows = merged[merged["human_task"].fillna("") != ""]
    agreement = agreement_report(human_rows)
    ann_cols = annotator_columns(human_rows)
    labels = pd.DataFrame({c: _clean_labels(human_rows[c]) for c in ann_cols})
    n_human = int(labels.notna().any(axis=1).sum()) if ann_cols else 0
    gold = resolve_gold(human_rows) if n_human else pd.Series(dtype="Int64")
    n_valid = int((gold == 1).sum()) if n_human else 0
    n_invalid = int((gold == 0).sum()) if n_human else 0

    return {
        "n_records": int(len(merged)),
        "automated_criteria": automated,
        "records_passing_all_criteria": {
            "count": int(all_pass.sum()),
            "rate": round(float(all_pass.mean()), 4),
        },
        "human_annotation": {
            "records_in_scope": int(len(human_rows)),
            "records_labelled": n_human,
            "gold_valid_hotels": n_valid,
            "gold_invalid_hotels": n_invalid,
            "agreement": agreement,
            "status": "pending" if n_human == 0 else "labelled",
        },
        "methodology_note": (
            "Automated criteria are reported as pass rates. Inter-annotator "
            "agreement is computed only over human-labelled hotel entries; the "
            "criteria are not treated as raters."
        ),
    }


def to_markdown(rep: dict) -> str:
    lines = [
        "# ViHoRec — Record-Level Validation (Issue I8)",
        "",
        f"Sampled records: **{rep['n_records']:,}**",
        "",
        "## Automated checklist criteria (deterministic)",
        "",
        "| Criterion | Pass rate | Failures | Interactions | Hotels |",
        "|---|---|---|---|---|",
    ]
    for name, d in rep["automated_criteria"].items():
        bt = d["by_type"]
        lines.append(
            f"| `{name}` | {d['pass_rate']} | {d['failures']} | "
            f"{bt.get('interaction', '--')} | {bt.get('hotel', '--')} |"
        )
    ap = rep["records_passing_all_criteria"]
    lines += [
        "",
        f"Records passing all three criteria: **{ap['count']:,}** ({ap['rate']}).",
        "",
        "## Human annotation (hotel catalogue entries)",
        "",
    ]
    h = rep["human_annotation"]
    if h["status"] == "pending":
        lines += [
            f"- Records awaiting labels: **{h['records_in_scope']:,}**",
            "- Annotators label independently; non-unanimous items are adjudicated.",
            "- Protocol: `annotation/GUIDELINES.md`, section 1.",
        ]
    else:
        ag = h["agreement"]
        lines += [
            f"- Records labelled: **{h['records_labelled']:,}**",
            f"- Gold: **{h.get('gold_valid_hotels', '—')}** valid hotels, "
            f"**{h.get('gold_invalid_hotels', '—')}** invalid",
            f"- Annotators: **{ag['n_annotators']}**; rated by all: "
            f"**{ag['n_items_rated_by_all']:,}**",
            f"- Unanimous: **{ag['unanimous_items']:,}** ({ag['unanimous_rate']})",
            f"- Percent agreement: **{ag.get('percent_agreement')}**",
        ]
        if "fleiss_kappa" in ag:
            lines.append(f"- Fleiss' kappa: **{ag['fleiss_kappa']['kappa']}**")
        for pair, k in ag.get("pairwise_cohens_kappa", {}).items():
            lines.append(f"- Cohen's kappa, {pair}: **{k['kappa']}** (n={k['n']:,})")
        lines.append(f"- Sent to adjudication: **{ag['items_needing_adjudication']}**")
        if ag.get("percent_agreement") and ag.get("kappa") is not None and ag["kappa"] <= 0:
            lines += [
                "",
                "Kappa is not informative here: nearly every hotel was labelled valid,",
                "so chance agreement is already ~1 and a single disagreement drives",
                "kappa to 0 or below. Report **percent agreement** and the one",
                "adjudicated row, not kappa, as the IAA figure.",
            ]
    lines += ["", f"_{rep['methodology_note']}_"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sample = sub.add_parser("sample", help="build the validation sheet")
    p_sample.add_argument("--n", type=int, default=250)
    p_sample.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    sub.add_parser("checklist", help="apply deterministic criteria")
    sub.add_parser("score", help="pass rates + inter-annotator agreement")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.cmd == "sample":
        print(json.dumps(cmd_sample(args.n, args.seed), ensure_ascii=False, indent=2))
    elif args.cmd == "checklist":
        print(json.dumps(cmd_checklist(), ensure_ascii=False, indent=2))
    else:
        rep = cmd_score()
        (C.OUT_REPORTS / f"{REPORT}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md = to_markdown(rep)
        (C.OUT_REPORTS / f"{REPORT}.md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
