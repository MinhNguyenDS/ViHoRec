# ViHoRec — Vietnamese Hotel Recommendation Dataset

A reproducible pipeline that turns the raw crawled hotel reviews into a
**quality-controlled, anonymised, benchmark-ready** dataset for recommender
systems research. This directory is the canonical build for the ViHoRec data
descriptor paper.

## Layout
```
dataset_release/
├── scripts/                 # reproducible pipeline (MIT licensed)
│   ├── config.py            # paths, seeds, secret-salt handling
│   ├── textnorm.py          # accent/stopword canonicalisation for entity matching
│   ├── quality_control.py   # dedup, missing, consistency, entity matching -> reports/
│   ├── user_identity_audit.py   # display-name collision / placeholder audit -> reports/
│   ├── er_candidate_pairs.py    # stratified hotel-pair sheet for annotation
│   ├── er_evaluate.py           # entity-resolution P/R/F1 vs human labels
│   ├── anonymize.py         # drop names, salted-HMAC pseudonyms -> release/
│   ├── make_benchmark_split.py  # three-way temporal short-history split
│   ├── run_baselines.py     # Random / MostPop / ItemKNN / UserKNN / BPR-MF / Content-TFIDF
│   ├── content_baseline.py  # TF-IDF content-based ranker (standalone)
│   ├── plot_style.py        # ACL-style matplotlib defaults
│   ├── data_analysis.py     # characterization (sparsity, Gini, long-tail) + ablations
│   ├── short_history_eval.py    # metrics stratified by train-history length
│   ├── run_cornac_benchmark.py  # SOTA models on the public split (Colab)
│   ├── make_figures.py      # statistics figures -> DS300/Image/
│   └── annotation_agreement.py  # record sampling, automated criteria, Cohen's kappa
├── release/                 # PUBLIC anonymised dataset (CC BY-NC 4.0)
│   ├── interactions.csv     # user_id, hotel_id, rating, date, source
│   ├── users.csv            # user_id, n_interactions
│   ├── hotels.csv           # hotel_id, name, location
│   └── benchmark/           # train.csv, val.csv, test.csv, PROTOCOL.md, maps, results
├── reports/                 # QC report (json + md); _private_mapping.csv is NOT public
├── annotation/             # validation sheet + agreement demo
├── DATASHEET.md             # Datasheets-for-Datasets documentation
└── LICENSE                  # CC BY-NC 4.0 (data) + MIT (code)
```

## Reproduce end-to-end
```bash
cd dataset_release/scripts
python quality_control.py        # -> reports/quality_report.{json,md}
python anonymize.py              # -> release/{interactions,users,hotels}.csv
python make_benchmark_split.py   # -> release/benchmark/{train,val,test}.csv
python run_baselines.py          # λ / Adaptive / EASE selected on val; test + CI + Wilcoxon
python run_phase3.py             # core table + LightGCN/NeuMF/GRU4Rec + short-history buckets
python data_analysis.py          # -> reports/analysis_report.json + Image/LongTail.png
python short_history_eval.py     # -> reports/short_history.json + Image/ShortHistory.png
python make_figures.py           # -> ViHoRec-SN/Image/*.png (ACL-style statistics)
```

### Dataset-credibility validation
```bash
python user_identity_audit.py    # -> reports/user_identity_audit.{json,md} + Image/UserIdentity.png

python annotation_agreement.py sample --n 250   # -> annotation/qc_sample_to_annotate.csv
python annotation_agreement.py checklist        # -> annotation/qc_criteria.csv (automated C1-C3)
python annotation_agreement.py score            # -> reports/qc_validation.{json,md}

python er_candidate_pairs.py     # -> annotation/er_pairs_to_annotate.csv (+ key, summary)
# ... a panel of 3 fills annotator_1..3; an adjudicator fills adjudicated
#     for non-unanimous rows. Packets: python annotation_packets.py split
#     Optional LLM panel:  scripts/llm_annotate/  (see annotation/GUIDELINES.md §3.1)
python er_evaluate.py            # -> reports/er_validation.{json,md}
```
Annotation protocols for both studies: `annotation/GUIDELINES.md`.
For a real (non-demo) release set a secret salt first:
```bash
export VIHOREC_SALT="your-secret"     # PowerShell: $env:VIHOREC_SALT="your-secret"
```

## Key statistics (auto-generated)
- Raw interactions: 18,274 (Booking 7,597 / Traveloka 6,273 / Ivivu 4,404)
- After cleaning: **17,911** interactions, **6,822** users, **560** hotels
  (7 exact duplicates and 356 rows with a placeholder reviewer name removed)
- Entity resolution is city- and property-type aware; 81 hotels on ≥2 sites,
  and **0** hotels left with a conflicting location
- Benchmark split (short-history, not strict cold-start): 798 users × 535 items,
  8,645 train / 798 val / 798 test, 97.60% sparse; shortest train history = 2
- Relevance: implicit next reviewed hotel; all ratings are positives.
  Sensitivity fold (`benchmark/relevance_ge_7/`): holdouts rated ≥ 7 only

## Known identity caveat
`user_id` is `HMAC(salt, display_name)`, so it is a **reviewer name key, not a
person**. `user_identity_audit.py` quantifies the gap: 39.8% of interactions
sit under single-token names such as `Nguyễn` or `Thanh`, and same-day
multi-city activity proves that some keys cover several travellers. Treat the
user count as an upper bound on identifiers and a lower bound on people.

## Entity-resolution caveat
The name-only canonical key used in the first release merged same-brand hotels
across cities: `RAON Hotel` (Quy Nhơn), `Raon Hotel` (Đà Nẵng) and `Raon Villa`
(Đà Lạt) shared one id carrying 139 interactions. `resolve_hotel_entities` now
keys on city and property type. Run `er_evaluate.py` after annotation to get
precision/recall for both matchers on the same gold labels.

## Requirements
Python ≥ 3.9 with `pandas`, `numpy` and `matplotlib` (see `requirements.txt`);
verified on Python 3.13.13 with pandas 3.0.3 and numpy 2.4.6. The content-based
baseline implements TF-IDF directly in numpy, so scikit-learn is **not** needed.
The full SOTA benchmark table additionally uses
[`cornac`](https://github.com/PreferredAI/cornac) and
[`recommenders`](https://github.com/recommenders-team/recommenders) with the
hyper-parameters listed in the paper.
