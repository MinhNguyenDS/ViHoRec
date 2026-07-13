# ViHoRec manual validation guidelines

Three annotators (co-authors) independently label each sampled record with
**1 = correct/consistent** or **0 = incorrect/inconsistent**.

## Interaction records
1. **Rater 1 (field validity):** rating in [1, 10]; date parseable and within
   2010--2024; `user_id`, `hotel_id`, and `source` non-empty.
2. **Rater 2 (release consistency):** the tuple exists in `interactions.csv`
   with matching rating, date, and source.
3. **Rater 3 (semantic plausibility):** hotel name/location in the catalogue
   are plausible; no obvious duplicate or identity-collision artefact.

## Hotel records
1. **Rater 1:** non-empty canonical `name` and `location`.
2. **Rater 2:** `hotel_id` unique in the release catalogue.
3. **Rater 3:** location is not among flagged multi-location conflicts;
   hotel has at least one interaction in the corpus.

Disagreements are resolved by majority vote for the accuracy estimate.
