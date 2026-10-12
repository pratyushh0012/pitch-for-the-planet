# Cleaning log — Phase 1 (`src/build_table.py`)

_Generated 2026-10-07._ Full row-level log: `data/processed/cleaning_log.csv`.

## Summary

- Steps: {'missing_bulletin': 46, 'leave_nan': 44, 'interpolate': 16, 'weather_partial_week': 1}
- Weather: 67 district points, daily → EDCD week (tmean/tmin/tmax/rh = mean of daily, rain = sum); national = mean of all districts; `_hot` = mean of hotspot districts (KATHMANDU, LALITPUR, BHAKTAPUR, SUNSARI, MORANG, JHAPA, CHITAWAN, KASKI, MAKWANPUR, DHADING).
- Recovery sources (value of a week whose own bulletin is missing, read from another bulletin):
  - `next_bulletin_prev_week`: overlap 240 weeks, exact agreement 33%, within 10% 82% → NOT used
  - `next_year_same_week`: overlap 316 weeks, exact agreement 25%, within 10% 65% → NOT used

## Coverage per year

|   year |   weeks |   cases_present |   cases_total |   recovered |   interpolated |   weather_present |
|-------:|--------:|----------------:|--------------:|------------:|---------------:|------------------:|
|   2016 |      52 |              39 |             8 |           0 |              2 |                18 |
|   2017 |      52 |              41 |            97 |           0 |              4 |                52 |
|   2018 |      52 |              46 |           155 |           0 |              0 |                52 |
|   2019 |      52 |              52 |         10177 |           0 |              4 |                52 |
|   2020 |      53 |              53 |           426 |           0 |              2 |                53 |
|   2021 |      52 |              52 |           344 |           0 |              0 |                52 |
|   2022 |      52 |              52 |         22865 |           0 |              0 |                52 |
|   2023 |      52 |              52 |         21988 |           0 |              1 |                52 |
|   2024 |      52 |              46 |         29192 |           0 |              1 |                52 |
|   2025 |      52 |              44 |          6025 |           0 |              2 |                52 |

## Every value that was dropped, filled or changed

| step | year | week | column | old | new | reason |
|---|---|---|---|---|---|---|
| missing_bulletin | 2016 | 25 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2016 | 34 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2016 | 39 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2016 | 40 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2016 | 42 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2016 | 47 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2016 | 48 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 3 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 7 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 18 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 22 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 37 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 38 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 39 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 41 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2017 | 52 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2018 | 1 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2018 | 12 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2018 | 13 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2018 | 14 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2018 | 37 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2018 | 38 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2019 | 17 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2019 | 32 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2019 | 36 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2019 | 44 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2020 | 30 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2020 | 42 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2023 | 48 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2024 | 10 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2024 | 11 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2024 | 12 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2024 | 13 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2024 | 14 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2024 | 15 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2024 | 17 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 1 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 2 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 3 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 4 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 5 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 6 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 7 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 9 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 47 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| missing_bulletin | 2025 | 52 | dengue_cases |  |  | no bulletin found on EDCD site (index + slug guesses) |
| interpolate | 2016 | 25 | dengue_cases |  | 0 | single-week gap: mean of neighbours |
| interpolate | 2016 | 34 | dengue_cases |  | 0 | single-week gap: mean of neighbours |
| interpolate | 2017 | 3 | dengue_cases |  | 0 | single-week gap: mean of neighbours |
| interpolate | 2017 | 7 | dengue_cases |  | 0 | single-week gap: mean of neighbours |
| interpolate | 2017 | 18 | dengue_cases |  | 0 | single-week gap: mean of neighbours |
| interpolate | 2017 | 22 | dengue_cases |  | 1 | single-week gap: mean of neighbours |
| interpolate | 2019 | 17 | dengue_cases |  | 1 | single-week gap: mean of neighbours |
| interpolate | 2019 | 32 | dengue_cases |  | 142 | single-week gap: mean of neighbours |
| interpolate | 2019 | 36 | dengue_cases |  | 912 | single-week gap: mean of neighbours |
| interpolate | 2019 | 44 | dengue_cases |  | 372 | single-week gap: mean of neighbours |
| interpolate | 2020 | 30 | dengue_cases |  | 14 | single-week gap: mean of neighbours |
| interpolate | 2020 | 42 | dengue_cases |  | 16 | single-week gap: mean of neighbours |
| interpolate | 2023 | 48 | dengue_cases |  | 256 | single-week gap: mean of neighbours |
| interpolate | 2024 | 17 | dengue_cases |  | 43 | single-week gap: mean of neighbours |
| interpolate | 2025 | 9 | dengue_cases |  | 64 | single-week gap: mean of neighbours |
| interpolate | 2025 | 47 | dengue_cases |  | 138 | single-week gap: mean of neighbours |
| leave_nan | 2016 | 39 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 40 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 41 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 42 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 43 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 44 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 45 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 46 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 47 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 48 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 49 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 50 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2016 | 51 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 37 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 38 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 39 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 41 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 42 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 43 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 44 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 45 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 46 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 47 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2017 | 52 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2018 | 1 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2018 | 12 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2018 | 13 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2018 | 14 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2018 | 37 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2018 | 38 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2024 | 10 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2024 | 11 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2024 | 12 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2024 | 13 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2024 | 14 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2024 | 15 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 1 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 2 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 3 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 4 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 5 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 6 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 7 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| leave_nan | 2025 | 52 | dengue_cases |  |  | gap longer than 1 week: left as NaN (no interpolation) |
| weather_partial_week |  |  | all | 67.0 | dropped | 67 district-weeks with < 7 valid days dropped (edges of the download window) |
