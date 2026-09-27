# ViHoRec — Reviewer-Identity Reliability Audit (Issue I1)

Reviewer identifiers are `HMAC(salt, display_name)`. This report measures
how far that key is from a true user identity.

## 1. Name-space ambiguity

| Property | Value |
|---|---|
| Interactions | 17,911 |
| Distinct display names (= released user ids) | 6,822 |
| Mean / median name length (chars) | 9.79 / 10 |
| Single-token names | 1,046 (15.33%) |
| Interactions under single-token names | 7,247 (40.46%) |
| Names <= 3 characters | 330 (4.84%) |
| Surname + initials only (e.g. `Nguyen H. Y. N.`) | 5,755 (84.36%) |

### Most frequent name keys

| Display name | Interactions | Tokens | Distinct cities |
|---|---|---|---|
| Nguyễn | 210 | 1 | 8 |
| Nguyen | 188 | 1 | 9 |
| Thanh | 172 | 1 | 9 |
| Thi | 154 | 1 | 9 |
| Minh | 129 | 1 | 9 |
| Nguyen T. T. | 117 | 3 | 9 |
| Trần | 106 | 1 | 9 |
| Anh | 104 | 1 | 9 |
| Nguyen T. H. | 103 | 3 | 9 |
| Trang | 97 | 1 | 9 |
| Linh | 95 | 1 | 9 |
| Ngọc | 84 | 1 | 8 |
| Thu | 79 | 1 | 9 |
| Thị | 75 | 1 | 9 |
| Lê | 73 | 1 | 8 |

## 1b. Imputed placeholder names (most severe defect)

**0 interactions (0.0%)** carry a
crawl-time placeholder instead of a display name. Pseudonymisation maps each
placeholder string to one `user_id`, creating a bucket of up to
**0 interactions** that appears in `users.csv`
as the single most active user in the corpus.

| Placeholder | Interactions | Hotels | Cities |
|---|---|---|---|

After removal: 17,911 interactions, 6,822 users, longest history 210 (was 0), 798 users eligible for the min-k=4 split.

## 2. Provable identity collisions

_Placeholder names are excluded here, so these figures describe genuine display names only._

Assumption: one reviewer cannot post reviews for hotels in two different cities on the same calendar day.

| Measure | Value |
|---|---|
| Same-day multi-city events | 171 |
| Name keys with a proven collision | 69 (1.01%) |
| Interactions under colliding keys | 3,513 (19.61%) |
| Released user ids | 6,822 |
| Lower bound on distinct individuals | 6,895 |
| Identity inflation factor (lower bound) | 1.011x |

### Name keys hiding the most people

| Display name | Min. distinct people | Interactions | Cities |
|---|---|---|---|
| Nguyen T. T. H. | 3 | 72 | 9 |
| Thi | 3 | 154 | 9 |
| Thanh | 3 | 172 | 9 |
| Trần | 3 | 106 | 9 |
| Anh | 2 | 104 | 9 |
| Hiếu | 2 | 26 | 7 |
| Dung | 2 | 56 | 8 |
| Chiến | 2 | 4 | 3 |
| Bao | 2 | 20 | 8 |
| Huong | 2 | 30 | 9 |
| Ho | 2 | 12 | 6 |
| Hoang | 2 | 60 | 9 |
| Hoa | 2 | 33 | 8 |
| Lan | 2 | 36 | 9 |
| Le H. T. | 2 | 8 | 5 |

## 3. Sensitivity to stricter identity filters

| Identity policy | Interactions | Users | Mean hist. | Max hist. | Eligible (>=4) | Sparsity |
|---|---|---|---|---|---|---|
| All display names (current release) | 17,911 (100.0%) | 6,822 | 2.63 | 210 | 798 | 99.53% |
| Drop imputed placeholder names | 17,911 (100.0%) | 6,822 | 2.63 | 210 | 798 | 99.53% |
| ... and names with <= 3 characters | 15,397 (85.96%) | 6,492 | 2.37 | 210 | 688 | 99.57% |
| ... and single-token names | 10,664 (59.54%) | 5,776 | 1.85 | 117 | 493 | 99.57% |
| >= 2 tokens, no proven collision | 10,155 (56.7%) | 5,763 | 1.76 | 46 | 481 | 99.58% |

## Reading for the manuscript

- `user_id` must be described as a **pseudonymous reviewer name key**, not a user identity.
- The distinct-user count is an **upper bound on identifiers** and a **lower bound on people**.
- Per-user history lengths are inflated wherever a common name absorbs several travellers.
- Placeholder rows must be removed before any user-level statistic is reported.
- Reported collisions are a **floor**, not an estimate: the same-day-multi-city test can only
  fire for names that already have multiple interactions, so names appearing once are
  unverifiable by construction.