# Recommenders-team baselines (public three-way split)

Library: [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders). The repo does not contain 2025–2026 *papers*; NEWS last updated April 2025. The models below are the newest **implementations** on `main`: PyTorch SASRec / SSEPT (UniRec port, 2026) and PyTorch NeuMF.

Protocol matches `run_neural_baselines.py`: train on `train.csv`, select epoch on `val.csv`, score `test.csv` once. Train-only history, full 535-item catalogue, mask seen train items. Recommenders' built-in sampled HR@10 (100 negatives) is **not** used.

| Method | MRR | NDCG@5 | Recall@5 | NDCG@10 | Recall@10 | Train (s) | Test (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SASRec | 0.0753 | 0.0580 | 0.0890 | 0.0845 | 0.1717 | 75 | 0 |
| NCF-NeuMF | 0.0681 | 0.0529 | 0.0840 | 0.0726 | 0.1441 | 53 | 2 |
| SSEPT | 0.0706 | 0.0554 | 0.0915 | 0.0810 | 0.1717 | 85 | 1 |

95% bootstrap CI over users; Wilcoxon on per-user Recall@10 vs UserKNN (R@10 = 0.1291). \* p<0.05.

| Method | R@10 | 95% CI | N@10 | vs UserKNN |
|---|---:|---|---:|---|
| SASRec | 0.1717 | [0.1466, 0.1980] | 0.0845 | 0.0007*** |
| NCF-NeuMF | 0.1441 | [0.1203, 0.1692] | 0.0726 | 0.1396 |
| SSEPT | 0.1717 | [0.1454, 0.1980] | 0.0810 | 0.0003*** |

Reference (same freeze, train-only history): UserKNN 0.1291, GRU4Rec 0.1604, LightGCN 0.1454, NeuMF 0.1228, TIGER 0.1566.

## Skipped

- **LightGCN**: Already evaluated in-repo (neural_results.csv, test R@10=0.1454). Recommenders LightGCN is the same 2020 model on the DeepRec yaml stack.
- **SUM**: TensorFlow DeepRec sequential (Lian et al., 2021). Not on the 2026 PyTorch UniRec line; extra TF dependency on Python 3.13.
- **SLi-Rec**: TensorFlow DeepRec (Microsoft, 2019). Same TF stack as SUM.
- **NextItNet**: TensorFlow DeepRec dilated CNN (Yuan et al., 2019).
- **NRMS/NAML/LSTUR**: News recommenders; they need article text. ViHoRec has hotel metadata, not news bodies.
