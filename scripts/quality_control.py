"""Quantitative quality-control pipeline for the ViHoRec hotel dataset.

Produces the numbers required by a data-descriptor paper:
  1. Per-site record counts and column completeness (missing rates).
  2. Exact and near-duplicate interaction rates (and how many were removed).
  3. Cross-site hotel entity matching (raw names -> canonical hotels, overlap).
  4. Consistency checks (rating range, parseable/future dates, one-hotel-many-locations).
  5. A cleaned, de-duplicated interaction table for downstream anonymisation.

Run:  python quality_control.py
Outputs: dataset_release/reports/quality_report.json  (+ .md, + cleaned csv)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import config as C
from textnorm import (
    canonical_hotel_key,
    canonical_location,
    is_placeholder_name,
    resolve_hotel_entities,
)


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def load_sites() -> pd.DataFrame:
    """Load the three raw per-site files and stack them with a ``source`` tag."""
    frames = []
    for site, path in C.SITE_FILES.items():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        # Sites use "Address"; the merged file renames it to "Location".
        if "Address" in df.columns and "Location" not in df.columns:
            df = df.rename(columns={"Address": "Location"})
        if "Ratting" in df.columns:
            df = df.rename(columns={"Ratting": "Rating"})
        df["source"] = site
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def missing_report(df: pd.DataFrame, cols: list[str]) -> dict:
    """Field completeness, counting crawl-time placeholders as missing.

    Placeholder detection is accent-insensitive (``is_placeholder_name``): an
    earlier exact comparison against ``"khong ten"`` never matched the accented
    ``"Không tên"`` actually present in the crawl, so the name field was
    reported as 100% complete while 342 rows carried an imputed value.
    """
    n = len(df)
    out = {}
    for col in cols:
        if col not in df.columns:
            out[col] = {"present": False}
            continue
        s = df[col].astype("string")
        blank = s.isna() | (s.str.strip() == "") | s.map(is_placeholder_name)
        out[col] = {
            "present": True,
            "missing_count": int(blank.sum()),
            "missing_pct": _pct(int(blank.sum()), n),
        }
    return out


def clean_rating(series: pd.Series) -> tuple[pd.Series, int]:
    """Coerce ratings to float; count values that could not be parsed/were invalid.

    Handles the known dirty token ``'8..5'`` and any non-numeric strings.
    """
    fixed = series.astype("string").str.replace("..", ".", regex=False)
    numeric = pd.to_numeric(fixed, errors="coerce")
    invalid = numeric.isna() | (numeric < C.RATING_MIN) | (numeric > C.RATING_MAX)
    return numeric, int(invalid.sum())


def run() -> dict:
    df = load_sites()
    n_raw = len(df)
    report: dict = {"n_raw_interactions": n_raw, "per_site": {}}

    # (1) Per-site counts.
    for site, cnt in df["source"].value_counts().items():
        report["per_site"][site] = int(cnt)

    # (2) Missing / completeness.
    report["completeness"] = missing_report(
        df, ["CustomerName", "Location", "NameHotel", "Rating", "Date"]
    )

    # (3) Consistency: ratings.
    df["Rating_clean"], n_bad_rating = clean_rating(df["Rating"])
    report["rating_consistency"] = {
        "invalid_or_out_of_range": n_bad_rating,
        "invalid_pct": _pct(n_bad_rating, n_raw),
        "min": float(np.nanmin(df["Rating_clean"])),
        "max": float(np.nanmax(df["Rating_clean"])),
        "mean": round(float(np.nanmean(df["Rating_clean"])), 4),
    }

    # (3b) Consistency: dates.
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    unparsable = int(df["Date_parsed"].isna().sum())
    lo, hi = pd.Timestamp(C.DATE_MIN), pd.Timestamp(C.DATE_MAX)
    out_of_range = int(((df["Date_parsed"] < lo) | (df["Date_parsed"] > hi)).sum())
    report["date_consistency"] = {
        "unparsable": unparsable,
        "unparsable_pct": _pct(unparsable, n_raw),
        "out_of_range": out_of_range,
        "range_observed": [
            str(df["Date_parsed"].min().date()),
            str(df["Date_parsed"].max().date()),
        ],
    }

    # (4) Duplicate analysis.
    df["loc_key"] = df["Location"].map(canonical_location)
    # City- and type-aware entity key (see textnorm.resolve_hotel_entities).
    # The name-only key is retained under a separate column so the entity-
    # resolution ablation can compare the two matchers on identical data.
    resolved = resolve_hotel_entities(df["NameHotel"], df["Location"])
    df["hotel_key"] = resolved["entity_key"]
    df["property_type"] = resolved["resolved_type"]
    df["hotel_key_nameonly"] = df["NameHotel"].map(canonical_hotel_key)

    exact_dupe_mask = df.duplicated(
        subset=["CustomerName", "NameHotel", "Rating", "Date"], keep="first"
    )
    # Near-duplicate: same reviewer + canonical hotel + date, ignoring rating noise
    # and cross-site re-posting.
    near_dupe_mask = df.duplicated(
        subset=["CustomerName", "hotel_key", "Date"], keep="first"
    )
    report["duplicates"] = {
        "exact_duplicate_rows": int(exact_dupe_mask.sum()),
        "exact_duplicate_pct": _pct(int(exact_dupe_mask.sum()), n_raw),
        "near_duplicate_rows": int(near_dupe_mask.sum()),
        "near_duplicate_pct": _pct(int(near_dupe_mask.sum()), n_raw),
    }

    # (5) Cross-site hotel entity matching.
    raw_names = df["NameHotel"].nunique()
    canonical_hotels = df["hotel_key"].nunique()
    # How many canonical hotels appear on >1 site (true cross-site matches)?
    per_hotel_sites = df.groupby("hotel_key")["source"].nunique()
    multi_site = int((per_hotel_sites > 1).sum())
    # How many raw spellings collapsed into a shared canonical key?
    names_per_key = df.groupby("hotel_key")["NameHotel"].nunique()
    merged_spelling_variants = int((names_per_key > 1).sum())
    nameonly_hotels = df["hotel_key_nameonly"].nunique()
    report["entity_matching"] = {
        "matcher": "city + property-type aware (resolve_hotel_entities)",
        "raw_distinct_hotel_names": int(raw_names),
        "canonical_hotels": int(canonical_hotels),
        "name_variants_collapsed": int(raw_names - canonical_hotels),
        "collapse_rate_pct": _pct(int(raw_names - canonical_hotels), int(raw_names)),
        "hotels_on_multiple_sites": multi_site,
        "canonical_keys_with_variant_spellings": merged_spelling_variants,
        "name_only_matcher_hotels": int(nameonly_hotels),
        "entities_split_from_name_only_matcher": int(
            df.groupby("hotel_key_nameonly")["hotel_key"].nunique().gt(1).sum()
        ),
        "entities_merged_beyond_name_only_matcher": int(
            df.groupby("hotel_key")["hotel_key_nameonly"].nunique().gt(1).sum()
        ),
    }

    # (5b) Consistency: one canonical hotel mapped to multiple locations.
    loc_per_hotel = df.groupby("hotel_key")["loc_key"].nunique()
    report["location_consistency"] = {
        "hotels_with_conflicting_location": int((loc_per_hotel > 1).sum()),
    }

    # (5c) Reviewer-identity validity: placeholder names are not identities and
    # must never be collapsed into a single pseudonym (see user_identity_audit).
    placeholder_mask = df["CustomerName"].map(is_placeholder_name)
    report["identity_validity"] = {
        "placeholder_name_rows": int(placeholder_mask.sum()),
        "placeholder_name_pct": _pct(int(placeholder_mask.sum()), n_raw),
        "placeholder_variants": sorted(
            df.loc[placeholder_mask, "CustomerName"].astype(str).unique().tolist()
        ),
        "note": (
            "These rows carry an imputed display name. Collapsing them by name "
            "creates one artificial high-activity user; they are excluded from "
            "user-level statistics and from the benchmark split."
        ),
    }

    # Build the cleaned interaction table: drop exact duplicates, invalid
    # ratings/dates, and rows whose reviewer name is an imputed placeholder.
    # Placeholder rows cannot be attributed to a person, and keeping them
    # collapses hundreds of unrelated reviews into one synthetic user.
    clean = df.loc[~exact_dupe_mask].copy()
    clean = clean[clean["Rating_clean"].between(C.RATING_MIN, C.RATING_MAX)]
    clean = clean[clean["Date_parsed"].notna()]
    n_before_placeholder = len(clean)
    clean = clean[~clean["CustomerName"].map(is_placeholder_name)]
    report["cleaned"] = {
        "n_after_cleaning": int(len(clean)),
        "removed_total": int(n_raw - len(clean)),
        "removed_pct": _pct(int(n_raw - len(clean)), n_raw),
        "removed_placeholder_identity": int(n_before_placeholder - len(clean)),
        "distinct_users_by_name": int(clean["CustomerName"].nunique()),
        "distinct_canonical_hotels": int(clean["hotel_key"].nunique()),
    }

    # Persist cleaned interactions for the anonymisation step.
    keep_cols = [
        "CustomerName", "NameHotel", "hotel_key", "hotel_key_nameonly",
        "property_type", "Location", "loc_key",
        "Rating_clean", "Date_parsed", "source",
    ]
    clean[keep_cols].to_csv(
        C.OUT_REPORTS / "interactions_clean.csv", index=False, encoding="utf-8"
    )
    return report


def to_markdown(rep: dict) -> str:
    d = rep
    lines = [
        "# ViHoRec — Data Quality-Control Report",
        "",
        f"- Raw interactions crawled: **{d['n_raw_interactions']:,}**",
        "- Per-site counts: "
        + ", ".join(f"{k}: {v:,}" for k, v in d["per_site"].items()),
        "",
        "## Completeness (missing rate per field)",
        "| Field | Missing | % |",
        "|---|---|---|",
    ]
    for col, info in d["completeness"].items():
        if info.get("present"):
            lines.append(f"| {col} | {info['missing_count']:,} | {info['missing_pct']}% |")
    em = d["entity_matching"]
    dup = d["duplicates"]
    cl = d["cleaned"]
    lines += [
        "",
        "## Duplicates",
        f"- Exact duplicate rows: **{dup['exact_duplicate_rows']:,}** "
        f"({dup['exact_duplicate_pct']}%)",
        f"- Near-duplicate (same reviewer + canonical hotel + date): "
        f"**{dup['near_duplicate_rows']:,}** ({dup['near_duplicate_pct']}%)",
        "",
        "## Cross-site entity matching (hotels)",
        f"- Raw distinct hotel names: **{em['raw_distinct_hotel_names']:,}**",
        f"- Canonical hotels after resolution: **{em['canonical_hotels']:,}**",
        f"- Name variants collapsed: **{em['name_variants_collapsed']:,}** "
        f"({em['collapse_rate_pct']}%)",
        f"- Hotels appearing on >1 site: **{em['hotels_on_multiple_sites']:,}**",
        "",
        "## Consistency",
        f"- Invalid/out-of-range ratings: **{d['rating_consistency']['invalid_or_out_of_range']:,}** "
        f"({d['rating_consistency']['invalid_pct']}%); observed range "
        f"[{d['rating_consistency']['min']}, {d['rating_consistency']['max']}]",
        f"- Unparsable dates: **{d['date_consistency']['unparsable']:,}**; "
        f"observed span {d['date_consistency']['range_observed']}",
        f"- Hotels with conflicting location: "
        f"**{d['location_consistency']['hotels_with_conflicting_location']:,}**",
        f"- Rows with a placeholder reviewer name: "
        f"**{d['identity_validity']['placeholder_name_rows']:,}** "
        f"({d['identity_validity']['placeholder_name_pct']}%) — "
        f"variants: {', '.join(d['identity_validity']['placeholder_variants']) or 'none'}",
        "",
        "## Cleaned dataset",
        f"- Interactions after cleaning: **{cl['n_after_cleaning']:,}** "
        f"(removed {cl['removed_total']:,}, {cl['removed_pct']}%; of which "
        f"{cl['removed_placeholder_identity']:,} had a placeholder reviewer name)",
        f"- Distinct users (by name key): **{cl['distinct_users_by_name']:,}**",
        f"- Distinct canonical hotels: **{cl['distinct_canonical_hotels']:,}**",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run()
    (C.OUT_REPORTS / "quality_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (C.OUT_REPORTS / "quality_report.md").write_text(to_markdown(rep), encoding="utf-8")
    # Vietnamese placeholder names break the default cp1252 Windows console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(to_markdown(rep))
