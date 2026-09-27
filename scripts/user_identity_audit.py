"""Reliability audit of the display-name-derived reviewer identifiers (Issue I1).

The public ``user_id`` is ``HMAC(salt, CustomerName)``: two reviews collapse into
one identifier whenever the crawled display names are byte-identical. The three
platforms expose only short, often single-token Vietnamese display names
("Đặng", "Anh", "Nguyen H. Y. N."), so the identifier is a *reviewer name key*
rather than a person. This script quantifies how far that key deviates from a
true user identity instead of asserting the gap in prose.

Three questions are answered with released data only:

1. **How ambiguous is the name space?** Character length, token count, and the
   share of single-token names, which carry almost no discriminative signal.
2. **How many identities provably merge distinct people?** A single traveller
   cannot review hotels in two different cities on the same calendar day. For
   every name we take the maximum number of distinct cities observed on one
   date; that value is a *lower bound* on how many individuals share the name.
   Summing over names lower-bounds the true population behind 6,832 ids.
3. **What survives a stricter identity filter?** User counts, history lengths,
   and benchmark eligibility (min-k = 4) are recomputed under progressively
   stricter name filters, so the paper can report a sensitivity range instead
   of a single optimistic number.

Run:  python user_identity_audit.py
Outputs: reports/user_identity_audit.{json,md}, Image/UserIdentity.png
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

import config as C
from textnorm import is_placeholder_name

MIN_INTERACTIONS = 4  # must match make_benchmark_split.py
SHORT_NAME_CHARS = 3  # threshold already used by the rater_3 QC criterion
TOP_N_NAMES = 15


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def load_named_interactions() -> pd.DataFrame:
    """Cleaned interactions that still carry the raw display name."""
    src = C.OUT_REPORTS / "interactions_clean.csv"
    if not src.exists():
        raise SystemExit("Run quality_control.py first to produce interactions_clean.csv")
    df = pd.read_csv(src)
    df["CustomerName"] = df["CustomerName"].astype("string").str.strip()
    df = df[df["CustomerName"].notna() & (df["CustomerName"] != "")].copy()
    df["date"] = pd.to_datetime(df["Date_parsed"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# (1) Name-space ambiguity
# --------------------------------------------------------------------------

def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"\s+", str(name).strip()) if t]


def name_space_report(df: pd.DataFrame) -> dict:
    names = df["CustomerName"]
    per_name = df.groupby("CustomerName").size()
    uniq = pd.Series(per_name.index, dtype="string")

    char_len = uniq.str.len().astype("int64")
    tok_count = uniq.map(lambda s: len(_tokens(s))).astype("int64")
    # "Nguyen H. Y. N." style: a surname followed by initials only.
    initials_only = uniq.map(
        lambda s: bool(re.fullmatch(r"[^\s]+(\s+[A-ZĐ]\.)+", str(s).strip()))
    )

    single_token = tok_count == 1
    short_name = char_len <= SHORT_NAME_CHARS

    # Interaction mass carried by ambiguous names (what actually pollutes the split).
    inter_single = int(per_name[single_token.to_numpy()].sum())
    inter_short = int(per_name[short_name.to_numpy()].sum())

    top = (
        per_name.sort_values(ascending=False)
        .head(TOP_N_NAMES)
        .rename("n_interactions")
        .reset_index()
    )
    top["n_tokens"] = top["CustomerName"].map(lambda s: len(_tokens(s)))
    top["n_cities"] = top["CustomerName"].map(
        df.groupby("CustomerName")["loc_key"].nunique()
    )

    return {
        "n_interactions": int(len(df)),
        "n_distinct_names": int(len(per_name)),
        "name_char_length": {
            "mean": round(float(char_len.mean()), 2),
            "median": int(char_len.median()),
            "p10": int(char_len.quantile(0.10)),
            "p90": int(char_len.quantile(0.90)),
            "max": int(char_len.max()),
        },
        "token_count_distribution": {
            str(k): int(v) for k, v in sorted(Counter(tok_count.tolist()).items())
        },
        "single_token_names": int(single_token.sum()),
        "single_token_names_pct": _pct(int(single_token.sum()), len(per_name)),
        "single_token_interactions": inter_single,
        "single_token_interactions_pct": _pct(inter_single, len(df)),
        "short_names_le3_chars": int(short_name.sum()),
        "short_names_le3_chars_pct": _pct(int(short_name.sum()), len(per_name)),
        "short_name_interactions": inter_short,
        "short_name_interactions_pct": _pct(inter_short, len(df)),
        "initials_only_names": int(initials_only.sum()),
        "initials_only_names_pct": _pct(int(initials_only.sum()), len(per_name)),
        "top_names": top.to_dict("records"),
    }


# --------------------------------------------------------------------------
# (1b) Placeholder names: imputed values that are not identities at all
# --------------------------------------------------------------------------

def placeholder_report(df: pd.DataFrame) -> dict:
    """Rows whose display name was imputed at crawl time.

    These are the most severe identity defect in the release: every such row
    shares one display name, so pseudonymisation maps them to a single
    ``user_id`` that becomes the most active "user" in the corpus.
    """
    mask = df["CustomerName"].map(is_placeholder_name)
    ph = df[mask]
    per_variant = (
        ph.groupby("CustomerName")
        .agg(
            n_interactions=("CustomerName", "size"),
            n_hotels=("hotel_key", "nunique"),
            n_cities=("loc_key", "nunique"),
        )
        .sort_values("n_interactions", ascending=False)
        .reset_index()
    )
    real = df[~mask]
    real_per_user = real.groupby("CustomerName").size()
    return {
        "placeholder_rows": int(mask.sum()),
        "placeholder_rows_pct": _pct(int(mask.sum()), len(df)),
        "placeholder_user_ids_created": int(ph["CustomerName"].nunique()),
        "largest_placeholder_bucket": int(per_variant["n_interactions"].max())
        if len(per_variant)
        else 0,
        "variants": per_variant.to_dict("records"),
        "after_removal": {
            "n_interactions": int(len(real)),
            "n_users": int(real_per_user.size),
            "max_interactions_per_user": int(real_per_user.max()) if real_per_user.size else 0,
            "benchmark_eligible_users": int((real_per_user >= MIN_INTERACTIONS).sum()),
        },
        "impact": (
            "The top entry of users.csv is a placeholder bucket, not a person; "
            "it must be removed or split before any user-level claim."
        ),
    }


# --------------------------------------------------------------------------
# (2) Provable collisions: same name, same day, different cities
# --------------------------------------------------------------------------

def collision_report(df: pd.DataFrame) -> dict:
    """Lower-bound the number of individuals hidden behind each name key.

    Assumption (stated in the paper): one traveller does not post reviews for
    hotels in two different cities on the same calendar day. For a name, the
    maximum number of distinct cities seen on any single date is therefore a
    lower bound on how many distinct people share that name.
    """
    # Placeholder buckets are trivially "colliding"; excluding them keeps this
    # a conservative statement about genuine display names.
    df = df[~df["CustomerName"].map(is_placeholder_name)]
    day_city = (
        df.groupby(["CustomerName", "date"])["loc_key"]
        .nunique()
        .rename("cities_same_day")
        .reset_index()
    )
    per_name_bound = (
        day_city.groupby("CustomerName")["cities_same_day"].max().rename("min_people")
    )

    colliding = per_name_bound[per_name_bound > 1]
    n_names = int(df["CustomerName"].nunique())
    per_name_counts = df.groupby("CustomerName").size()

    # Interactions attributable to names with proven collisions.
    inter_colliding = int(per_name_counts[colliding.index].sum())

    # Population lower bound: every non-colliding name contributes >= 1 person.
    lower_bound_people = int(per_name_bound.sum())

    # Same-day-different-city event count (the raw evidence).
    n_events = int((day_city["cities_same_day"] > 1).sum())

    worst = (
        colliding.sort_values(ascending=False)
        .head(TOP_N_NAMES)
        .rename("min_distinct_people")
        .reset_index()
    )
    worst["n_interactions"] = worst["CustomerName"].map(per_name_counts)
    worst["n_cities_total"] = worst["CustomerName"].map(
        df.groupby("CustomerName")["loc_key"].nunique()
    )

    # Cross-platform check: a name active on all three sites is weak evidence of
    # merging, reported separately because re-posting is a legitimate behaviour.
    sites_per_name = df.groupby("CustomerName")["source"].nunique()

    return {
        "assumption": (
            "one reviewer cannot post reviews for hotels in two different cities "
            "on the same calendar day"
        ),
        "same_day_multi_city_events": n_events,
        "names_with_proven_collision": int(len(colliding)),
        "names_with_proven_collision_pct": _pct(int(len(colliding)), n_names),
        "interactions_under_colliding_names": inter_colliding,
        "interactions_under_colliding_names_pct": _pct(inter_colliding, len(df)),
        "released_user_ids": n_names,
        "lower_bound_distinct_people": lower_bound_people,
        "identity_inflation_factor": round(lower_bound_people / n_names, 3)
        if n_names
        else 0.0,
        "names_on_all_three_platforms": int((sites_per_name == 3).sum()),
        "worst_offenders": worst.to_dict("records"),
    }


# --------------------------------------------------------------------------
# (3) Sensitivity of the corpus and benchmark to stricter identity filters
# --------------------------------------------------------------------------

def _filter_mask(df: pd.DataFrame, policy: str) -> pd.Series:
    name = df["CustomerName"]
    not_placeholder = ~name.map(is_placeholder_name)
    if policy == "all_names":
        return pd.Series(True, index=df.index)
    if policy == "drop_placeholders":
        return not_placeholder
    if policy == "drop_short_le3":
        return not_placeholder & (name.str.len() > SHORT_NAME_CHARS)
    if policy == "drop_single_token":
        return not_placeholder & name.map(lambda s: len(_tokens(s)) >= 2)
    if policy == "min_two_tokens_and_no_collision":
        multi = name.map(lambda s: len(_tokens(s)) >= 2)
        day_city = df.groupby(["CustomerName", "date"])["loc_key"].nunique()
        bad = {n for (n, _), v in day_city.items() if v > 1}
        return not_placeholder & multi & ~name.isin(bad)
    raise ValueError(policy)


POLICIES = [
    ("all_names", "All display names (current release)"),
    ("drop_placeholders", "Drop imputed placeholder names"),
    ("drop_short_le3", "... and names with <= 3 characters"),
    ("drop_single_token", "... and single-token names"),
    ("min_two_tokens_and_no_collision", ">= 2 tokens, no proven collision"),
]


def sensitivity_report(df: pd.DataFrame) -> list[dict]:
    rows = []
    for policy, label in POLICIES:
        sub = df[_filter_mask(df, policy)]
        if sub.empty:
            continue
        per_user = sub.groupby("CustomerName").size()
        eligible = per_user[per_user >= MIN_INTERACTIONS]
        n_users, n_items = len(per_user), sub["hotel_key"].nunique()
        rows.append(
            {
                "policy": policy,
                "label": label,
                "n_interactions": int(len(sub)),
                "retained_pct": _pct(len(sub), len(df)),
                "n_users": int(n_users),
                "n_hotels": int(n_items),
                "mean_interactions_per_user": round(float(per_user.mean()), 2),
                "max_interactions_per_user": int(per_user.max()),
                "single_interaction_users_pct": _pct(int((per_user == 1).sum()), n_users),
                "benchmark_eligible_users": int(len(eligible)),
                "sparsity_pct": round(
                    100.0 * (1 - len(sub) / (n_users * n_items)), 2
                )
                if n_users and n_items
                else 0.0,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Figure + reporting
# --------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import plot_style as ps
    except Exception:
        return None

    ps.apply_acl_style()
    uniq = pd.Series(df["CustomerName"].unique(), dtype="string")
    tok = uniq.map(lambda s: min(len(_tokens(s)), 5))
    counts = tok.value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(ps.PAGE_WIDTH * 0.78, 2.2))
    ax = axes[0]
    ax.bar([str(i) if i < 5 else "5+" for i in counts.index], counts.to_numpy(),
           color=ps.VIH_BLUE, width=0.65)
    ax.set_xlabel("Tokens in display name")
    ax.set_ylabel("Distinct names")
    ax.set_title("Display-name granularity", fontsize=8)

    ax = axes[1]
    day_city = df.groupby(["CustomerName", "date"])["loc_key"].nunique()
    bound = day_city.groupby(level=0).max()
    b = bound.value_counts().sort_index()
    b = b[b.index <= 6]
    ax.bar([str(i) for i in b.index], b.to_numpy(), color=ps.VIH_ACCENT, width=0.65)
    ax.set_xlabel("Lower bound on people per name key")
    ax.set_ylabel("Name keys (log)")
    ax.set_yscale("log")
    ax.set_title("Same-day multi-city evidence", fontsize=8)

    out = C.PAPER_IMG_DIR / "UserIdentity.png"
    ps.save_fig(fig, out)
    plt.close(fig)
    return str(out)


def to_markdown(rep: dict) -> str:
    ns, col = rep["name_space"], rep["collisions"]
    lines = [
        "# ViHoRec — Reviewer-Identity Reliability Audit (Issue I1)",
        "",
        "Reviewer identifiers are `HMAC(salt, display_name)`. This report measures",
        "how far that key is from a true user identity.",
        "",
        "## 1. Name-space ambiguity",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Interactions | {ns['n_interactions']:,} |",
        f"| Distinct display names (= released user ids) | {ns['n_distinct_names']:,} |",
        f"| Mean / median name length (chars) | {ns['name_char_length']['mean']} / {ns['name_char_length']['median']} |",
        f"| Single-token names | {ns['single_token_names']:,} ({ns['single_token_names_pct']}%) |",
        f"| Interactions under single-token names | {ns['single_token_interactions']:,} ({ns['single_token_interactions_pct']}%) |",
        f"| Names <= {SHORT_NAME_CHARS} characters | {ns['short_names_le3_chars']:,} ({ns['short_names_le3_chars_pct']}%) |",
        f"| Surname + initials only (e.g. `Nguyen H. Y. N.`) | {ns['initials_only_names']:,} ({ns['initials_only_names_pct']}%) |",
        "",
        "### Most frequent name keys",
        "",
        "| Display name | Interactions | Tokens | Distinct cities |",
        "|---|---|---|---|",
    ]
    for r in ns["top_names"]:
        lines.append(
            f"| {r['CustomerName']} | {r['n_interactions']:,} | {r['n_tokens']} | {r['n_cities']} |"
        )

    ph = rep["placeholders"]
    lines += [
        "",
        "## 1b. Imputed placeholder names (most severe defect)",
        "",
        f"**{ph['placeholder_rows']:,} interactions ({ph['placeholder_rows_pct']}%)** carry a",
        "crawl-time placeholder instead of a display name. Pseudonymisation maps each",
        f"placeholder string to one `user_id`, creating a bucket of up to",
        f"**{ph['largest_placeholder_bucket']:,} interactions** that appears in `users.csv`",
        "as the single most active user in the corpus.",
        "",
        "| Placeholder | Interactions | Hotels | Cities |",
        "|---|---|---|---|",
    ]
    for r in ph["variants"]:
        lines.append(
            f"| `{r['CustomerName']}` | {r['n_interactions']:,} | "
            f"{r['n_hotels']:,} | {r['n_cities']} |"
        )
    ar = ph["after_removal"]
    lines += [
        "",
        f"After removal: {ar['n_interactions']:,} interactions, {ar['n_users']:,} users, "
        f"longest history {ar['max_interactions_per_user']:,} "
        f"(was {ph['largest_placeholder_bucket']:,}), "
        f"{ar['benchmark_eligible_users']:,} users eligible for the min-k={MIN_INTERACTIONS} split.",
        "",
        "## 2. Provable identity collisions",
        "",
        "_Placeholder names are excluded here, so these figures describe genuine display names only._",
        "",
        f"Assumption: {col['assumption']}.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Same-day multi-city events | {col['same_day_multi_city_events']:,} |",
        f"| Name keys with a proven collision | {col['names_with_proven_collision']:,} ({col['names_with_proven_collision_pct']}%) |",
        f"| Interactions under colliding keys | {col['interactions_under_colliding_names']:,} ({col['interactions_under_colliding_names_pct']}%) |",
        f"| Released user ids | {col['released_user_ids']:,} |",
        f"| Lower bound on distinct individuals | {col['lower_bound_distinct_people']:,} |",
        f"| Identity inflation factor (lower bound) | {col['identity_inflation_factor']}x |",
        "",
        "### Name keys hiding the most people",
        "",
        "| Display name | Min. distinct people | Interactions | Cities |",
        "|---|---|---|---|",
    ]
    for r in col["worst_offenders"]:
        lines.append(
            f"| {r['CustomerName']} | {r['min_distinct_people']} | "
            f"{r['n_interactions']:,} | {r['n_cities_total']} |"
        )

    lines += [
        "",
        "## 3. Sensitivity to stricter identity filters",
        "",
        "| Identity policy | Interactions | Users | Mean hist. | Max hist. | Eligible (>=4) | Sparsity |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rep["sensitivity"]:
        lines.append(
            f"| {r['label']} | {r['n_interactions']:,} ({r['retained_pct']}%) | "
            f"{r['n_users']:,} | {r['mean_interactions_per_user']} | "
            f"{r['max_interactions_per_user']:,} | {r['benchmark_eligible_users']:,} | "
            f"{r['sparsity_pct']}% |"
        )

    lines += [
        "",
        "## Reading for the manuscript",
        "",
        "- `user_id` must be described as a **pseudonymous reviewer name key**, not a user identity.",
        "- The distinct-user count is an **upper bound on identifiers** and a **lower bound on people**.",
        "- Per-user history lengths are inflated wherever a common name absorbs several travellers.",
        "- Placeholder rows must be removed before any user-level statistic is reported.",
        "- Reported collisions are a **floor**, not an estimate: the same-day-multi-city test can only",
        "  fire for names that already have multiple interactions, so names appearing once are",
        "  unverifiable by construction.",
    ]
    return "\n".join(lines)


def run() -> dict:
    df = load_named_interactions()
    rep = {
        "name_space": name_space_report(df),
        "placeholders": placeholder_report(df),
        "collisions": collision_report(df),
        "sensitivity": sensitivity_report(df),
        "config": {
            "min_interactions_benchmark": MIN_INTERACTIONS,
            "short_name_chars": SHORT_NAME_CHARS,
        },
    }
    fig = make_figure(df)
    if fig:
        rep["figure"] = fig
    return rep


if __name__ == "__main__":
    report = run()
    (C.OUT_REPORTS / "user_identity_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = to_markdown(report)
    out_md = C.OUT_REPORTS / "user_identity_audit.md"
    out_md.write_text(md, encoding="utf-8")
    # Vietnamese names break the default cp1252 Windows console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(md)
    print(f"\n[written] {out_md}")
