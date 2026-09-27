"""Anonymisation / pseudonymisation for the public ViHoRec release.

Rationale
---------
The original pipeline stored the reviewer display name (``CustomerName``) in
plaintext and derived ``IDuser`` directly from it. For a public dataset we:
  * DROP every direct identifier (the display name never leaves this machine);
  * replace it with a salted **HMAC-SHA256** pseudonym so the same reviewer maps
    to the same opaque token across releases, but the token cannot be trivially
    reversed without the secret salt;
  * assign stable canonical hotel ids from the entity-resolution key so that
    cross-site spelling variants share one id.

The reviewer-name -> pseudonym lookup is written to ``_private_mapping.csv``,
which MUST NOT be published (see DATASHEET.md). Only ``release/`` is public.

Caveat: HMAC over a low-cardinality name space is still vulnerable to a
dictionary attack if the salt leaks; therefore the salt is secret AND the name
is dropped. Downstream users only ever see opaque ids.

Run:  python anonymize.py      (reads reports/interactions_clean.csv)
"""

from __future__ import annotations

import hashlib
import hmac

import pandas as pd

import config as C
from textnorm import resolve_hotel_entities


def pseudonym(value: str, salt: str = C.PSEUDONYM_SALT, prefix: str = "U") -> str:
    """Deterministic, salted, non-reversible pseudonym for a raw identifier."""
    digest = hmac.new(salt.encode("utf-8"), str(value).encode("utf-8"), hashlib.sha256)
    return f"{prefix}{digest.hexdigest()[:12]}"


def run() -> dict:
    src = C.OUT_REPORTS / "interactions_clean.csv"
    if not src.exists():
        raise SystemExit("Run quality_control.py first to produce interactions_clean.csv")
    df = pd.read_csv(src)

    # --- Pseudonymous user ids (drop the name entirely from the release) ---
    df["user_id"] = df["CustomerName"].map(pseudonym)

    # --- Canonical, stable hotel ids from the entity-resolution key ---
    # quality_control.py resolves entities with city and property type; recover
    # anything unresolved rather than silently dropping it.
    missing = df["hotel_key"].isna()
    if missing.any():
        df.loc[missing, "hotel_key"] = resolve_hotel_entities(
            df.loc[missing, "NameHotel"], df.loc[missing, "Location"]
        )["entity_key"]
    hotel_keys = sorted(df["hotel_key"].unique())
    key_to_id = {k: f"H{ i:04d}" for i, k in enumerate(hotel_keys)}
    df["hotel_id"] = df["hotel_key"].map(key_to_id)

    # Representative display name per canonical hotel: the most frequent spelling.
    rep_name = (
        df.groupby("hotel_id")["NameHotel"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    rep_loc = (
        df.groupby("hotel_id")["Location"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    # --- Public interaction table (no identifiers) ---
    interactions = (
        df[["user_id", "hotel_id", "Rating_clean", "Date_parsed", "source"]]
        .rename(columns={"Rating_clean": "rating", "Date_parsed": "date"})
        .sort_values(["date", "user_id"])
        .reset_index(drop=True)
    )
    interactions["date"] = pd.to_datetime(interactions["date"]).dt.strftime("%Y-%m-%d")
    interactions.to_csv(C.OUT_RELEASE / "interactions.csv", index=False, encoding="utf-8")

    # --- Public user table (opaque id + activity count only) ---
    users = (
        interactions.groupby("user_id")
        .size()
        .rename("n_interactions")
        .reset_index()
        .sort_values("n_interactions", ascending=False)
    )
    users.to_csv(C.OUT_RELEASE / "users.csv", index=False, encoding="utf-8")

    # --- Public hotel table ---
    hotels = pd.DataFrame(
        {"hotel_id": rep_name.index, "name": rep_name.values, "location": rep_loc.values}
    ).sort_values("hotel_id")
    hotels.to_csv(C.OUT_RELEASE / "hotels.csv", index=False, encoding="utf-8")

    # --- PRIVATE mapping (never publish) ---
    private = (
        df[["CustomerName", "user_id"]]
        .drop_duplicates()
        .sort_values("user_id")
    )
    private.to_csv(
        C.OUT_REPORTS / "_private_mapping.csv", index=False, encoding="utf-8"
    )

    summary = {
        "n_interactions": int(len(interactions)),
        "n_users": int(users["user_id"].nunique()),
        "n_hotels": int(hotels["hotel_id"].nunique()),
        "identifiers_dropped": ["CustomerName"],
        "pseudonym_scheme": "HMAC-SHA256(salt, name)[:12]",
        "salt_is_default_demo": C.PSEUDONYM_SALT == C.DEFAULT_SALT,
    }
    print("Anonymised release written to", C.OUT_RELEASE)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary["salt_is_default_demo"]:
        print("  WARNING: using the public demo salt. Set VIHOREC_SALT for a real release.")
    return summary


if __name__ == "__main__":
    run()
