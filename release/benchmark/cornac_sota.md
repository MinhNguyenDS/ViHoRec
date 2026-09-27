# Cornac generative / CF baselines (public three-way split)

Fit on `train.csv`. Best-on-val checkpoint when the model supports it. `test.csv` scored once. Item text = hotel name + city + Facilities/Around/Vicinity/Price, encoded with `paraphrase-multilingual-MiniLM-L12-v2`.

Sequential prefixes follow Cornac leave-last-out: the validation review is in the test-time history but is not a training label. That is one extra observed item relative to GRU4Rec in `run_neural_baselines.py` (train history only).

SANSA is item–item CF (RecSys 2023). It uses the same train-only history and seen-item mask as EASE / UserKNN. SuiteSparse/CHOLMOD cannot build on this Windows env, and Cornac SANSA ICF is unstable on the 19.5%-dense 535-item gramian (the 150k-item sparse path). Reported SANSA is the exact dense LDL — the CHOLMOD density=1 closed form of Spišák et al. at this catalog size. Val-tuned λ=1000.

| Method | MRR | NDCG@5 | Recall@5 | NDCG@10 | Recall@10 | Train (s) | Test (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TIGER | 0.0599 | 0.0495 | 0.0802 | 0.0734 | 0.1566 | 19974 | 155 |
| RPG | 0.0673 | 0.0515 | 0.0802 | 0.0722 | 0.1454 | 2299 | 35 |
| LETTER | 0.0569 | 0.0484 | 0.0802 | 0.0698 | 0.1466 | 28724 | 171 |
| SANSA | 0.0583 | 0.0450 | 0.0727 | 0.0598 | 0.1190 | 2 | 0 |

95% bootstrap CI over users; Wilcoxon on per-user Recall@10 vs UserKNN (R@10 = 0.1291). \* p<0.05.

| Method | R@10 | 95% CI | N@10 | vs UserKNN |
|---|---:|---|---:|---|
| TIGER | 0.1566 | [0.1316, 0.1830] | 0.0734 | 0.0233* |
| RPG | 0.1454 | [0.1216, 0.1704] | 0.0722 | 0.2596 |
| LETTER | 0.1466 | [0.1228, 0.1717] | 0.0698 | 0.2050 |
| SANSA | 0.1190 | [0.0977, 0.1416] | 0.0598 | 0.1441 |

Reference (same freeze, train-only history): UserKNN 0.1291, Hybrid-fixed 0.1353, GRU4Rec 0.1604, EASE λ=1000 0.1203.

## Skipped

- **Companion**: Needs SentimentModality (aspect–opinion tuples from reviews). ViHoRec does not release review text.
- **HypAR**: Review-hypergraph model; the public release has no review documents.
- **DiffGRM**: Paper-scale GPU recipe (d_model=256, 100 epochs, beam 128). This environment is torch CPU-only.
