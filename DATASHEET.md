# Datasheet for the ViHoRec Dataset

Following the *Datasheets for Datasets* framework (Gebru et al., 2021). All
statistics below are produced automatically by `scripts/quality_control.py`,
`scripts/anonymize.py`, and `scripts/make_benchmark_split.py`.

## 1. Motivation
- **Purpose.** There is no publicly documented Vietnamese hotel recommendation
  dataset. ViHoRec fills this gap for research on collaborative filtering,
  content-based, and hybrid recommendation, and on short-history ranking.
  The full corpus is cold-start dominated (most reviewer keys have one
  interaction); the public protocol is short-history, not strict cold-start.
- **Created by.** The authors (University of Information Technology, VNU-HCM).
- **Not for.** Commercial use (see LICENSE) or re-identification of individuals.

## 2. Composition
Three released tables (`release/`):

| File | Rows | Columns |
|---|---|---|
| `interactions.csv` | 17,911 | user_id, hotel_id, rating, date, source |
| `users.csv` | 6,822 | user_id, n_interactions |
| `hotels.csv` | 560 | hotel_id, name, location |
| content metadata (`data_content_based_raw.csv`) | 309 | 11 attributes (facilities, surroundings, vicinity, price, distance, ...) |

- **Instances.** A row in `interactions.csv` is one user–hotel rating (0–10)
  with a timestamp and its originating site.
- **Sources.** Booking.com (7,597), Traveloka (6,273), Ivivu (4,404).
- **Ratings** span 1.0–10.0; **dates** span 2011-10-15 to 2023-12-09.
- **Sensitive data.** Direct identifiers (reviewer display names) are **removed**
  before release; user ids are salted-HMAC pseudonyms (see §6).

## 3. Collection Process
- **How.** Automated crawling with `requests`/BeautifulSoup and JSON review
  APIs where available; manual collection for content metadata (no public API).
- **Sampling.** Sites and hotels selected by credibility and user volume; hotels
  concentrated in Vietnamese destinations (Đà Lạt, Đà Nẵng, Nha Trang, Vũng Tàu,
  Phú Quốc, Phan Thiết, ...).
- **Timeframe.** Reviews were posted 2011–2023; crawling performed in 2023.

## 4. Preprocessing / Cleaning / Quality Control
Reproduced by `scripts/quality_control.py`. Reported measures:

| Check | Result |
|---|---|
| Field completeness | CustomerName missing/placeholder 1.95% (357 rows); other fields 0% |
| Exact duplicate interactions | 7 (0.038%) removed |
| Near-duplicates (reviewer + canonical hotel + date) | 11 (0.060%) |
| Invalid / out-of-range ratings | 0 (dirty token `8..5` repaired) |
| Unparsable dates | 0 |
| Placeholder reviewer names dropped | 356 rows (`Không tên` and variants) |
| Raw hotel names → canonical hotels | 581 → 560 (city- and type-aware resolver) |
| Hotels appearing on ≥2 sites | 81 |
| Hotels with conflicting location | 0 |

- **Entity resolution.** Listings merge when they share a city and the same
  discriminative name tokens, and their stated property types are compatible
  (`textnorm.resolve_hotel_entities`). The submitted name-only key is retained
  for ablation. Pair-level validation is documented in `reports/er_validation.md`.
- **Public split.** Three-way temporal leave-one-out on users with ≥4
  interactions: last = test, second-last = val, rest = train
  (798 users, 8,645 / 798 / 798). See `release/benchmark/PROTOCOL.md`.
  Relevance is implicit next-item; all ratings are positives.
- **Manual validation.** `annotation_agreement.py` draws a stratified sample
  (default n≈250: interactions + hotels) for an annotator panel (default 3) and
  reports percent agreement, pairwise Cohen's κ and Fleiss' κ over the panel,
  plus an estimated record-accuracy rate. Agreement is computed only over
  human-judged entries; the deterministic criteria are not treated as raters.

## 5. Uses
- Recommended: benchmarking CF/CB/hybrid recommenders, short-history studies,
  Vietnamese-language RecSys, low-resource / sparse-data research.
  Users with a single interaction are a corpus property, not part of the
  public ranking protocol.
- **Known limitations.**
  - Small scale (17,911 interactions) vs. MovieLens-100k / Amazon; sparse
    (97.60% sparsity in the benchmark split).
  - Reviewer display names were partially imputed/normalised at crawl time
    (missing names were replaced), so `n_interactions` per user and the
    number of distinct users are approximate — user identity is derived from
    a low-cardinality name string and may merge distinct individuals.
  - Ratings are aggregate scores, not multi-criteria.

## 6. Ethics, Terms of Service & Legal
- **Terms of Service.** Booking.com, Traveloka, and Ivivu restrict automated
  scraping and commercial reuse in their ToS. To stay within a defensible
  research-use position we: (a) collected only publicly visible review text and
  ratings, no private/account data; (b) do **not** redistribute raw HTML or
  full review text, only derived numeric ratings and hotel metadata;
  (c) release under **CC BY-NC 4.0** (non-commercial); (d) provide takedown on
  request. Users of this dataset must comply with the source platforms' ToS.
- **Personal data / pseudonymization.** No emails, account ids, or full names are
  released. `anonymize.py` drops the display name entirely and assigns a
  salted `HMAC-SHA256(secret_salt, name)[:12]` pseudonym; the secret salt is
  kept off-repo (`VIHOREC_SALT`) and the name-to-id lookup
  (`reports/_private_mapping.csv`) is **never** published. HMAC of a display
  name is not anonymization: a holder of the salt could re-identify keys, and
  common names still collide (see `reports/user_identity_audit.md`).
- **Risk.** Residual re-identification risk is non-zero. Mitigations: no free
  text, city-level location only, unpublished name map. Do not describe the
  release as anonymous.

## 7. Distribution & Maintenance
- Hosted with a versioned DOI (e.g., Zenodo); this repository is the canonical
  build pipeline. Report issues / request takedown to the corresponding author.
