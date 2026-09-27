# ViHoRec public evaluation protocol

This is the frozen protocol for the revised release. Scripts that search a
hyper-parameter must read `val.csv`; `test.csv` is scored once.

## Task name

**Short-history / temporal leave-one-out recommendation.**

It is not a strict user-cold-start protocol. Eligible users have at least four
interactions in the cleaned corpus. After the last and second-last reviews are
held out, every evaluated user still has at least two training interactions.
The shortest test bucket is therefore train-history length **2**, not 0.

The full corpus *is* cold-start dominated as a **dataset property**: most
reviewer keys have a single interaction and are excluded from this split.
That fact belongs in the characterization table, not in the name of the
benchmark.

## Fold assignment (per user, chronological)

| Fold | Assignment | File |
|---|---|---|
| train | all interactions except the last two | `train.csv` |
| val | second-last interaction | `val.csv` |
| test | last interaction | `test.csv` |

Ordering is per user, not a global calendar cut. One user's training review
can be later than another user's test review. Identifiers are remapped to
contiguous integers in `user_map.csv` / `item_map.csv`.

Current counts (see `split_config.json`):

- 798 users × 535 hotels
- 8,645 train / 798 val / 798 test
- train-history min / median / max = 2 / 5 / 208
- 183 users have exactly two training interactions
- sparsity 97.60%

## Relevance (I10)

The task is implicit next-item ranking. For a user, the single relevant item
is the hotel they reviewed next, **regardless of the numeric score**. Every
crawled rating is a positive (Cornac `rating_threshold = 0.5`). A 3/10 visit
counts the same as a 10/10 visit: the signal is the visit.

A sensitivity variant that keeps only holdouts rated ≥ 7 lives in
`relevance_ge_7/` (train unchanged; 506 val / 512 test users remain). Do not
overwrite the canonical files with that ablation.

## What must not happen

- Selecting λ, `(α_min, n0, τ)`, EASE λ, or neural hyper-parameters on `test.csv`
- Calling the min-k = 4 protocol a cold-start benchmark
- Quoting submitted Adaptive Recall@10 0.1388 / 0.141 or `adaptive_hybrid_results.md` from the two-way run (the val-tuned file now matches `baseline_ci.md`)

## Statistics (Phase 3)

Primary claims use per-user Recall@10 and NDCG@10. Uncertainty is a percentile
bootstrap 95% CI over users (10,000 resamples). Model comparisons are paired
Wilcoxon signed-rank tests on per-user Recall@10 (ties dropped). The pairing
unit is the user: leave-one-out ranking scores are bounded and zero-inflated,
so a Gaussian t-test is the wrong default.
