# ViHoRec — Entity-Resolution Validation (Issue I2)

## Label provenance — LLM annotation with human verification

LLM annotation with human verification of 30 load-bearing rows plus a 100-row random audit of unanimous negatives. Describe it as such, never as manual annotation.

| Slot | Kind | Model | Labels |
|---|---|---|---|
| `annotator_1` | llm | openai/gpt-5-mini | 426 |
| `annotator_2` | llm | openai/gpt-4.1-mini | 426 |
| `annotator_3` | llm | z-ai/glm-5.3-flash | 413 |

Human-resolved labels (adjudication or blind verification): **130**

## Human vs model panel (blind verification)

The human labelled the load-bearing packet without seeing model
labels. This is the validity number; model κ is only task clarity.

- Compared: **30**
- Agreed: **28** (0.9333)
- Overturned by human: P0123, P0185

## Random audit of unanimous negatives

Uniform sample of panel-unanimous different-hotel pairs outside the
load-bearing packet. The annotator did not see model votes.

- Audited: **100**
- Agreed with panel: **100**
- Human labelled as same-hotel: **0**
- Overturned by human: none
- Rule-of-three 95% upper bound on residual panel error: **0.03** (100 clean trials)

## Annotation coverage

- Candidate pairs in sheet: **426**
- Pairs with a resolved label: **426** (unlabelled: 0)
- Algorithm merges annotated: **26** / 26 (exhaustive => precision is unbiased)

## Agreement among annotators (model panel)

These figures measure **agreement between models**, i.e. how well the
task is specified. They are not evidence that the labels are correct,
and must never be reported as inter-annotator agreement among people.

- Annotators: **3** (annotator_1, annotator_2, annotator_3)
- Pairs rated by all annotators: **413**
- Unanimous: **409** (0.9903)
- Sent to adjudication: **4**
- Fleiss' kappa (3 raters): **0.9418**

| Annotator pair | Items | Percent agreement | Cohen's kappa |
|---|---|---|---|
| annotator_1 vs annotator_2 | 426 | 0.9953 | 0.9575 |
| annotator_1 vs annotator_3 | 413 | 0.9952 | 0.9558 |
| annotator_2 vs annotator_3 | 413 | 0.9903 | 0.9115 |

## Matcher quality

| Matcher | Precision | Recall (sample) | Recall (population) | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Name-only canonical key (submitted version) | **0.7727** | 0.7391 | **0.7391** | **0.7556** | 17 | 5 | 6 |
| City- and type-aware resolver (revised) | **1.0** | 0.913 | **0.913** | **0.9545** | 21 | 0 | 2 |

Stratum weights (population pairs per annotated pair): algo_merge=1.0, top_similar=1.0, hard_negative=23.94, random_negative=1098.49

### Stability of the population recall estimate

Precision is exact: every merge is annotated. The **population recall**
column is not — it extrapolates the sampled strata to all 168k pairs, so
a single missed duplicate found in a heavily weighted stratum dominates
it. Where that happens, quote the sample recall as the primary figure and
the population recall only as an indicative lower bound.

| Matcher | Stratum | Missed items found | Weighted FN | Share of FN mass |
|---|---|---|---|---|
| Name-only canonical key (submitted version) | algo_merge | 4 | 4.0 | 0.6667 |
| Name-only canonical key (submitted version) | top_similar | 2 | 2.0 | 0.3333 |
| City- and type-aware resolver (revised) | top_similar | 2 | 2.0 | 1.0 |

## Duplicate rate per stratum

| Stratum | Labelled | True duplicates | Rate |
|---|---|---|---|
| algo_merge | 26 | 21 | 0.8077 |
| hard_negative | 150 | 0 | 0.0 |
| random_negative | 150 | 0 | 0.0 |
| top_similar | 100 | 2 | 0.02 |

## Pairs where the two matchers disagree

These decide the ablation: `gold = 1` favours whichever matcher merged.

| Pair | Name A | City A | Name B | City B | Name-only | Resolved | Gold |
|---|---|---|---|---|---|---|---|
| P0050 | Khách sạn Rosaleen Boutique Huế | Huế | Rosaleen Boutique Hotel | Huế | 0 | 1 | 1 |
| P0076 | Raon Hotel - STAY 24H | Đà Nẵng | Raon Villa - STAY 24H | Đà Lạt | 1 | 0 | 0 |
| P0085 | Khách sạn Thanh Lịch Royal Boutique Huế | Huế | Thanh Lich Royal Boutique | Huế | 0 | 1 | 1 |
| P0098 | Khu nghỉ dưỡng Pilgrimage Village Boutique | Huế | Pilgrimage Village Resort & Spa Huế | Huế | 0 | 1 | 1 |
| P0104 | Khách sạn Sea Links Beach Phan Thiết | Phan Thiết | Sea Links Beach Villas Phan Thiết | Phan Thiết | 1 | 0 | 0 |
| P0123 | Khu nghỉ dưỡng FLC Luxury Quy Nhơn | Quy Nhơn | Khách sạn FLC Luxury Quy Nhơn | Quy Nhơn | 1 | 0 | 0 |
| P0193 | HAIAN Beach Hotel & Spa | Đà Nẵng | HAIAN Beach Hotel & Spa Đà Nẵng | Đà Nẵng | 0 | 1 | 1 |
| P0398 | RAON Hotel - STAY 24H | Quy Nhơn | Raon Villa - STAY 24H | Đà Lạt | 1 | 0 | 0 |
| P0419 | RAON Hotel - STAY 24H | Quy Nhơn | Raon Hotel - STAY 24H | Đà Nẵng | 1 | 0 | 0 |

## False merges — Name-only canonical key (submitted version)

| Pair | Name A | City A | Name B | City B |
|---|---|---|---|---|
| P0076 | Raon Hotel - STAY 24H | Đà Nẵng | Raon Villa - STAY 24H | Đà Lạt |
| P0104 | Khách sạn Sea Links Beach Phan Thiết | Phan Thiết | Sea Links Beach Villas Phan Thiết | Phan Thiết |
| P0123 | Khu nghỉ dưỡng FLC Luxury Quy Nhơn | Quy Nhơn | Khách sạn FLC Luxury Quy Nhơn | Quy Nhơn |
| P0398 | RAON Hotel - STAY 24H | Quy Nhơn | Raon Villa - STAY 24H | Đà Lạt |
| P0419 | RAON Hotel - STAY 24H | Quy Nhơn | Raon Hotel - STAY 24H | Đà Nẵng |

## Missed duplicates — Name-only canonical key (submitted version)

| Pair | Name A | City A | Name B | City B | Jaccard |
|---|---|---|---|---|---|
| P0050 | Khách sạn Rosaleen Boutique Huế | Huế | Rosaleen Boutique Hotel | Huế | 0.3333 |
| P0085 | Khách sạn Thanh Lịch Royal Boutique Huế | Huế | Thanh Lich Royal Boutique | Huế | 0.5714 |
| P0098 | Khu nghỉ dưỡng Pilgrimage Village Boutique | Huế | Pilgrimage Village Resort & Spa Huế | Huế | 0.2222 |
| P0122 | Khu nghỉ dưỡng Hải Dương Intourco Vũng Tàu | Vũng Tàu | Khu nghỉ dưỡng Intourco  Vũng Tàu | Vũng Tàu | 0.8571 |
| P0193 | HAIAN Beach Hotel & Spa | Đà Nẵng | HAIAN Beach Hotel & Spa Đà Nẵng | Đà Nẵng | 0.6667 |
| P0362 | Khách sạn TTC - Ngọc Lan Đà Lạt | Đà Lạt | Khách sạn TTC – Đà Lạt | Đà Lạt | 0.7143 |

## Missed duplicates — City- and type-aware resolver (revised)

| Pair | Name A | City A | Name B | City B | Jaccard |
|---|---|---|---|---|---|
| P0122 | Khu nghỉ dưỡng Hải Dương Intourco Vũng Tàu | Vũng Tàu | Khu nghỉ dưỡng Intourco  Vũng Tàu | Vũng Tàu | 0.8571 |
| P0362 | Khách sạn TTC - Ngọc Lan Đà Lạt | Đà Lạt | Khách sạn TTC – Đà Lạt | Đà Lạt | 0.7143 |