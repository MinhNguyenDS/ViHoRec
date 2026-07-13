# Datasheet for the ViHoRec Dataset

Following the *Datasheets for Datasets* framework (Gebru et al., 2021). All
statistics below are produced automatically by `scripts/quality_control.py`,
`scripts/anonymize.py`, and `scripts/make_benchmark_split.py`.

## 1. Motivation
- **Purpose.** There is no publicly documented Vietnamese hotel recommendation
  dataset. ViHoRec fills this gap for research on collaborative filtering,
  content-based, and hybrid recommendation, and on cold-start handling.
- **Created by.** The authors (University of Information Technology, VNU-HCM).
- **Not for.** Commercial use (see LICENSE) or re-identification of individuals.

## 2. Composition
Three released tables (`release/`):

| File | Rows | Columns |
|---|---|---|
| `interactions.csv` | 18,267 | user_id, hotel_id, rating, date, source |
| `users.csv` | 6,832 | user_id, n_interactions |
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
| Field completeness | 0% missing after collection-time imputation (see limitation below) |
| Exact duplicate interactions | 7 (0.038%) removed |
| Near-duplicates (reviewer + canonical hotel + date) | 11 (0.060%) |
| Invalid / out-of-range ratings | 0 (dirty token `8..5` repaired) |
| Unparsable dates | 0 |
| Raw hotel names → canonical hotels | 581 → 560 (21 spelling variants merged, 3.6%) |
| Hotels appearing on ≥2 sites | 78 |
| Hotels with conflicting location | 1 (flagged) |

- **Entity resolution.** Cross-site hotel matching uses an accent-free,
  stopword-stripped, order-independent canonical key (`textnorm.py`), replacing
  the original naïve `LabelEncoder(NameHotel)` exact-string matching.
- **Manual validation.** `annotation_agreement.py` draws a stratified sample
  (default n≈250: interactions + hotels) for ≥2 annotators and reports percent
  agreement and Cohen's / Fleiss' κ, plus an estimated record-accuracy rate.

## 5. Uses
- Recommended: benchmarking CF/CB/hybrid recommenders, cold-start studies,
  Vietnamese-language RecSys, low-resource / sparse-data research.
- **Known limitations.**
  - Small scale (18k interactions) vs. MovieLens-100k / Amazon; sparse
    (97.5% sparsity in the benchmark split).
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
- **Personal data / anonymisation.** No emails, account ids, or full names are
  released. `anonymize.py` drops the display name entirely and assigns a
  salted `HMAC-SHA256(secret_salt, name)[:12]` pseudonym; the secret salt is
  kept off-repo (`VIHOREC_SALT`) and the name→id lookup
  (`reports/_private_mapping.csv`) is **never** published.
- **Risk.** Re-identification risk is low: no free-text, no geolocation beyond
  city, and pseudonymous ids.

## 7. Distribution & Maintenance
- Hosted with a versioned DOI (e.g., Zenodo); this repository is the canonical
  build pipeline. Report issues / request takedown to the corresponding author.
