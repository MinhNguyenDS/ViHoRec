# ViHoRec — Data Quality-Control Report

- Raw interactions crawled: **18,274**
- Per-site counts: booking: 7,597, traveloka: 6,273, ivivu: 4,404

## Completeness (missing rate per field)
| Field | Missing | % |
|---|---|---|
| CustomerName | 357 | 1.9536% |
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
- Hotels appearing on >1 site: **81**

## Consistency
- Invalid/out-of-range ratings: **0** (0.0%); observed range [1.0, 10.0]
- Unparsable dates: **0**; observed span ['2011-10-15', '2023-12-09']
- Hotels with conflicting location: **0**
- Rows with a placeholder reviewer name: **357** (1.9536%) — variants: -, Guest, Khách, Không tên, Na, グエン, 吴振成, 洪, 粟原, 선아

## Cleaned dataset
- Interactions after cleaning: **17,911** (removed 363, 1.9864%; of which 356 had a placeholder reviewer name)
- Distinct users (by name key): **6,822**
- Distinct canonical hotels: **560**