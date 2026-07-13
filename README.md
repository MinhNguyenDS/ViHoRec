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
│   ├── anonymize.py         # drop names, salted-HMAC pseudonyms -> release/
│   ├── make_benchmark_split.py  # leave-last-one-out temporal split -> release/benchmark/
│   ├── run_baselines.py     # Random / MostPop / ItemKNN / UserKNN / BPR-MF / Content-TFIDF
│   ├── content_baseline.py  # TF-IDF content-based ranker (standalone)
│   ├── plot_style.py        # ACL-style matplotlib defaults
│   ├── data_analysis.py     # characterization (sparsity, Gini, long-tail) + ablations
│   ├── cold_start_eval.py   # metrics stratified by user history length
│   ├── run_cornac_benchmark.py  # SOTA models on the public split (Colab)
│   ├── make_figures.py      # statistics figures -> DS300/Image/
│   └── annotation_agreement.py  # sampling + Cohen's / Fleiss' kappa
├── release/                 # PUBLIC anonymised dataset (CC BY-NC 4.0)
│   ├── interactions.csv     # user_id, hotel_id, rating, date, source
│   ├── users.csv            # user_id, n_interactions
│   ├── hotels.csv           # hotel_id, name, location
│   └── benchmark/           # train.csv, test.csv, *_map.csv, split_config.json, results
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
python make_benchmark_split.py   # -> release/benchmark/{train,test}.csv
python run_baselines.py          # -> release/benchmark/baseline_results.{csv,md} (Random/MostPop/ItemKNN/UserKNN/BPR-MF)
python data_analysis.py          # -> reports/analysis_report.json (characterization + ablations) + Image/LongTail.png
python cold_start_eval.py        # -> reports/cold_start.json (metrics by user history length) + Image/ColdStart.png
python make_figures.py           # -> ViHoRec/Image/*.png (ACL-style statistics)
python annotation_agreement.py sample --n 250   # build validation sheet
python annotation_agreement.py label            # apply 3-rater checklist -> annotated.csv
python annotation_agreement.py score annotation/annotated.csv
```
For a real (non-demo) release set a secret salt first:
```bash
export VIHOREC_SALT="your-secret"     # PowerShell: $env:VIHOREC_SALT="your-secret"
```

## Key statistics (auto-generated)
- Raw interactions: 18,274 (Booking 7,597 / Traveloka 6,273 / Ivivu 4,404)
- After cleaning: **18,267** interactions, **6,832** users, **560** hotels
- Entity matching merged 21 cross-site name variants; 78 hotels on ≥2 sites
- Benchmark split: 800 users × 535 items, 9,787 train / 800 test, 97.53% sparse

## Requirements
Python ≥ 3.9 with `pandas`, `numpy`, `scikit-learn` (see `requirements.txt`).
The full SOTA benchmark table additionally uses
[`cornac`](https://github.com/PreferredAI/cornac) and
[`recommenders`](https://github.com/recommenders-team/recommenders) with the
hyper-parameters listed in the paper.
