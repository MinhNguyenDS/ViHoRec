# ViHoRec — Record-Level Validation (Issue I8)

Sampled records: **248**

## Automated checklist criteria (deterministic)

| Criterion | Pass rate | Failures | Interactions | Hotels |
|---|---|---|---|---|
| `C1_field_validity` | 1.0 | 0 | 1.0 | 1.0 |
| `C2_release_consistency` | 1.0 | 0 | 1.0 | 1.0 |
| `C3_identity_plausibility` | 0.8952 | 26 | 0.8602 | 1.0 |

Records passing all three criteria: **222** (0.8952).

## Human annotation (hotel catalogue entries)

- Records labelled: **62**
- Gold: **62** valid hotels, **0** invalid
- Annotators: **3**; rated by all: **62**
- Unanimous: **61** (0.9839)
- Percent agreement: **0.9839**
- Fleiss' kappa: **-0.0054**
- Cohen's kappa, annotator_1 vs annotator_2: **0.0** (n=62)
- Cohen's kappa, annotator_1 vs annotator_3: **0.0** (n=62)
- Cohen's kappa, annotator_2 vs annotator_3: **None** (n=62)
- Sent to adjudication: **1**

Kappa is not informative here: nearly every hotel was labelled valid,
so chance agreement is already ~1 and a single disagreement drives
kappa to 0 or below. Report **percent agreement** and the one
adjudicated row, not kappa, as the IAA figure.

_Automated criteria are reported as pass rates. Inter-annotator agreement is computed only over human-labelled hotel entries; the criteria are not treated as raters._