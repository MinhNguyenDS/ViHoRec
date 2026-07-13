# ViHoRec — Data Quality-Control Report

- Raw interactions crawled: **18,274**
- Per-site counts: booking: 7,597, traveloka: 6,273, ivivu: 4,404

## Completeness (missing rate per field)
| Field | Missing | % |
|---|---|---|
| CustomerName | 0 | 0.0% |
| Location | 0 | 0.0% |
| NameHotel | 0 | 0.0% |
| Rating | 0 | 0.0% |
| Date | 0 | 0.0% |

## Duplicates
- Exact duplicate rows: **7** (0.0383%)
- Near-duplicate (same reviewer + canonical hotel + date): **11** (0.0602%)

## Cross-site entity matching (hotels)
- Raw distinct hotel names: **581**
- Canonical hotels after resolution: **560**
- Name variants collapsed: **21** (3.6145%)
- Hotels appearing on >1 site: **78**

## Consistency
- Invalid/out-of-range ratings: **0** (0.0%); observed range [1.0, 10.0]
- Unparsable dates: **0**; observed span ['2011-10-15', '2023-12-09']
- Hotels with conflicting location: **1**

## Cleaned dataset
- Interactions after cleaning: **18,267** (removed 7, 0.0383%)
- Distinct users (by name key): **6,832**
- Distinct canonical hotels: **560**