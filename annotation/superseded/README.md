# Superseded validation artifacts

These files back the validation numbers in the **submitted** version of the
ViHoRec manuscript. They are kept for provenance and should not be used.

| File | What it was | Why it was withdrawn |
|---|---|---|
| `sample_to_validate.csv` | 250-record sample with empty `rater_1..3` columns | The three columns are *criteria*, not people |
| `annotated.csv` | The same sample with the criteria applied | Same |
| `agreement_report.json` | Fleiss' kappa over `rater_1..3` | Not interpretable: two criteria pass every cleaned record by construction, so expected agreement is near 1 and the statistic (kappa = -0.0736) measures nothing about annotator reliability |
| `demo_annotated.csv` | Demonstration labels | Illustrative only |

The replacement protocol separates deterministic checks from human judgement
and is documented in `../GUIDELINES.md`. It is produced by
`scripts/annotation_agreement.py` (automated criteria and hotel-entry labels)
and `scripts/er_candidate_pairs.py` with `scripts/er_evaluate.py`
(entity-resolution pairs).
