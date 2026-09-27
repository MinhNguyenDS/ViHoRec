# ViHoRec manual validation guidelines

Two separate studies are documented here. They answer different questions and
must not be pooled into a single agreement statistic.

| Study | Question | Who decides | Output |
|---|---|---|---|
| §1 Record validation | Is a released record well-formed and plausible? | Code (3 criteria) + an annotator panel on hotel entries | `reports/qc_validation.md` |
| §2 Entity-resolution pairs | Do two hotel names denote the same hotel? | Annotator panel + 1 adjudicator | `reports/er_validation.md` |

**Terminology.** `C1`–`C3` are *criteria* evaluated deterministically by
`annotation_agreement.py`, not people. Inter-annotator agreement is reported
only where a rater made a judgement, over the `annotator_*` columns.
An earlier version of this protocol labelled the three criteria `rater_1`–
`rater_3` and computed Fleiss' kappa across them; that statistic was not
interpretable, because two of the criteria pass every cleaned record by
construction, and it has been withdrawn.

---

## 1. Record validation

Sample: `annotation/qc_sample_to_annotate.csv`, drawn by
`python annotation_agreement.py sample --n 250` (stratified: interactions
balanced across Booking.com, Traveloka and Ivivu, plus hotel catalogue
entries).

### 1.1 Automated criteria

Applied by `python annotation_agreement.py checklist`. Reported as pass rates.

**Interaction records**

1. `C1_field_validity` — `rating` within `[1, 10]`; `date` parseable and within
   2010–2024; `user_id`, `hotel_id` and `source` non-empty.
2. `C2_release_consistency` — the tuple appears in `release/interactions.csv`
   with matching rating, date and source.
3. `C3_identity_plausibility` — the underlying display name is not a crawl-time
   placeholder (`Không tên`, `Guest`, …) and is longer than three characters.
   Checked against `reports/_private_mapping.csv`, which is never published.

**Hotel records**

1. `C1_field_validity` — canonical `name` and `location` are non-empty.
2. `C2_release_consistency` — `hotel_id` exists in `release/hotels.csv`.
3. `C3_identity_plausibility` — the hotel has at least one interaction.

### 1.2 Human judgement

Only hotel catalogue entries carry a human question, because a pseudonymised
interaction row exposes nothing a person can verify beyond what code already
checks.

> **Question.** Does this entry denote exactly one real hotel at this location?

Label `1` when the name identifies a single property that plausibly exists in
the stated city. Label `0` when the entry is a chain-level or generic name that
could cover several properties, when the location contradicts the name, or when
the entry appears to be a fragment.

Procedure: each annotator labels independently without seeing the others'
labels; the gold label is the majority vote; an adjudicator reviews every
non-unanimous entry. Agreement is computed over the fully labelled entries
(see §3.0).

---

## 2. Entity-resolution pair annotation

Sample: `annotation/er_pairs_to_annotate.csv`, drawn by
`python er_candidate_pairs.py`. The sheet is shuffled and shows no similarity
score and no pipeline decision, so annotators cannot anchor on the algorithm.

> **Question.** Do `name_a` and `name_b` refer to the same physical hotel?

Fill each `annotator_*` column with `1` (same hotel) or `0` (different hotels).
Leave `adjudicated` empty unless the panel was not unanimous, in which case the
adjudicator enters the final label there. Use `notes` for anything ambiguous.

### 2.1 Decision rules

Label **1 — same hotel** when the names differ only by:

- diacritics or transliteration (`Phú Quốc` / `Phu Quoc`);
- a property-type prefix or suffix (`Khách sạn X`, `X Hotel`, `Khu nghỉ dưỡng X`,
  `X Resort`);
- word order or punctuation (`Phan Thiet Ocean Dunes Resort` /
  `Khu nghỉ dưỡng Ocean Dunes Phan Thiết`);
- an operator or brand form that resolves to the same property
  (`Mövenpick Resort Phan Thiet` / `Movenpick Phan Thiết`).

Label **0 — different hotels** when any of these hold:

- the properties are in **different cities**, even under an identical brand
  (`Raon Hotel` Đà Nẵng vs `Raon Hotel` Quy Nhơn);
- the names carry different **tier or sub-brand** markers of the same operator
  (`Mường Thanh Grand` vs `Mường Thanh Luxury`; `Majestic` vs `Majestic Premium`);
- the names carry different **branch numbers** (`La Cactus Hotel` vs
  `La Cactus Hotel 2`);
- the property **types** genuinely differ at the same brand (`Raon Hotel` vs
  `Raon Villa`), unless external evidence shows one listing for one property.

### 2.2 Evidence and edge cases

Decide from `name_a`/`name_b`, `city_a`/`city_b` and the platform columns first.
When still unsure, search the hotel name plus city on the source platform and
record the URL in `notes`. If the evidence remains inconclusive, label `0` and
flag it in `notes`: the estimate should not credit the matcher for a merge that
could not be confirmed.

City fields come from the crawl and are occasionally missing or coarse. Treat a
blank city as unknown rather than as a match.

### 2.3 Why the sample looks unbalanced

Pairs are drawn from four strata (`algo_merge`, `top_similar`, `hard_negative`,
`random_negative`). Duplicates are deliberately over-represented relative to the
168k possible pairs so that both error types are observable. `er_evaluate.py`
reweights each stratum back to the full pair space, so annotators should judge
each pair on its own and ignore how often `1` appears.

---

## 3. Annotators

Record who — or what — performed each study before submission. State the kind
as `human` or `LLM`; for an LLM give the exact model slug.

| Study | Annotator 1 | Kind | Annotator 2 | Kind | Annotator 3 | Kind | Adjudicator / verifier | Kind |
|---|---|---|---|---|---|---|---|---|
| §1 Record validation (hotels) | Thuat Thien Nguyen | human | Minh Nhat Ta | human | Minh Hoang Nguyen | human | majority 2–1 (`R0006`); first author | human |
| §2 Entity-resolution pairs | `openai/gpt-5-mini` | LLM | `openai/gpt-4.1-mini` | LLM | `z-ai/glm-5.3-flash` | LLM | Minh Hoang Nguyen (blind 30-row packet) | human |

Hotel-entry labels were produced independently by **Thuat Thien Nguyen**, **Minh Nhat Ta** (University of Information Technology, VNU-HCM), and **Minh Hoang Nguyen** (first author). All three are fluent in Vietnamese and familiar with Vietnamese hotel naming. The three `qc_hotels_annotator{1,2,3}.csv` packets correspond to this panel; the public merge sheet does not name which person filled which column. No separate calibration round was discarded.

The load-bearing ER packet (`er_pairs_human_verification.csv`, 30 pairs) was labelled by Minh Hoang Nguyen without seeing model votes. The residual-error audit (`er_pairs_negative_audited.csv`, 100 unanimous negatives) was labelled by the same person, still blind: **100/100** agreed with the panel.

### 3.0 Panel size and how the gold label is formed

The default panel is three annotators. The gold label is the **majority vote**;
an explicit adjudication overrides it. With an odd panel every item resolves, so
no item is silently dropped for want of agreement, which is what happens with
two annotators whenever they disagree.

Items that were **not unanimous** are still written to
`packets/<study>_adjudication.csv` for a human to check. A 2-1 split is where a
shared blind spot shows up, and it is worth the small amount of extra work.

Agreement is reported as pairwise Cohen's kappa for every annotator pair, plus
Fleiss' kappa across the panel once there are three or more raters. Fleiss'
kappa is appropriate here because the columns are independent raters judging the
same items. It was *not* appropriate in the superseded study, where the same
statistic was computed over three deterministic criteria; see
`annotation/superseded/README.md`.

### 3.1 Model-assisted annotation

`scripts/llm_annotate/annotate.py` can fill an annotator slot with a model
through OpenRouter. It receives the same decision rules as §2.1, sees one item
at a time, and never sees the other annotators' labels or the pipeline's
decision. Different models therefore produce independent label sets that the
same merge and adjudication code scores.

The current panel is three models from three vendors:

| Slot | Model | Notes |
|---|---|---|
| 1 | `openai/gpt-5-mini` | reasoning model; `temperature` omitted, effort `low` |
| 2 | `openai/gpt-4.1-mini` | plain sampling model; `temperature` 0 |
| 3 | `z-ai/glm-5.3-flash` | reasoning always on; effort `low`, `temperature` 1.0, `top_p` 0.95 per vendor guidance, determinism via `seed` |

Three vendors rather than three sizes of one vendor is deliberate: models from
the same family share training data and failure modes, so a unanimous verdict
from them would overstate reliability.

If this route is used, three things are mandatory:

1. **The adjudicator must be human.** A model panel adjudicated by another model
   is not a validated ground truth, because correlated model errors survive it.
   Majority voting narrows the work but does not remove this requirement.
2. **Disclose it everywhere the annotation is described** — the manuscript, the
   datasheet, and the response letter — as *LLM annotation with human
   adjudication*, never as manual annotation.
3. **Report agreement per pairing.** Kappa among models measures model
   consistency, not human reliability, and must be labelled as such. A high
   Fleiss' kappa over three models is evidence that the task is well specified,
   not that the labels are correct.

`annotation/llm_provenance.json` records which model filled which slot, and the
`notes` column of every model-labelled row carries the model slug, its stated
confidence and its reason, so provenance survives even when a CSV is read alone.

### 3.2 Human verification of the load-bearing rows

Requirement 1 above does not mean a person must relabel everything. The reported
numbers do not rest evenly on the sample: precision and the matcher-vs-matcher
comparison are computed entirely from the exhaustively annotated `algo_merge`
stratum, and recall moves only when a positive changes. `annotation_packets.py
verify` collects exactly those rows — every pair in an exhaustive stratum, every
positive, and every item the panel split on — which for the current study is 30
of 426 pairs.

The packet is **blind**: it carries no model label, no gold column and no reason
for selection. Anchoring the annotator would destroy the one statistic that
makes the panel defensible, namely how often a person independently reaches the
same verdict. `merge` reports that as `human_vs_panel`, together with the ids
the human overturned, and the human label takes precedence over the majority
vote wherever it exists.

Report the two together: agreement among the models (task clarity) and agreement
between the human and the panel (label validity). Only the second supports
calling the result a ground truth.

### 3.3 Random audit of unanimous negatives

The 30-row packet does not cover unanimous panel negatives. Those rows do not
move precision or the matcher comparison, but a silent panel error there would
still inflate recall. `annotation_packets.py audit --n 100` draws a uniform
random sample from that remainder and writes:

- annotator sheet: `packets/er_pairs_negative_audit.csv` (blind; no model label)
- coordinator key: `packets/er_pairs_negative_audit_key.csv` (not given to the rater)
- instructions: `packets/HUONG-DAN-AUDIT.md` (same decision rules as §2.1 / `HUONG-DAN.md` §1)

The sheet does **not** tell the rater that the panel voted 0. A clean audit of
100 items puts the 95% rule-of-three upper bound on residual panel error near
3%. **Done:** 100/100 agreed; 0 same-hotel labels; bound ≈ 0.03. See
`reports/er_validation.md`.
