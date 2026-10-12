# DengueWatch Nepal — CLAUDE.md

Project context for Claude Code. Read this whole file before doing anything.

## 1. What this project is

**One line:** forecast dengue surges in Nepal 4 weeks ahead from weather and recent cases, so hospitals can prepare beds, platelets and staff.

- **Owner:** Pratyush (software engineering student, Macquarie University).
- **Follows on from MonsoonWatch** (AGE/gastro forecasting with the same EWARS data). Lesson from MonsoonWatch: rainfall did NOT improve the gastro forecast, so weather's value here must be proven with an ablation test, not assumed.
- **Scope:** national, weekly. District level is a stretch goal only. Never claim district-level results unless they are actually built.

### What it predicts (every week *t*)
1. **Classifier:** probability that cases cross the surge line in any of the next 4 weeks (t+1 … t+4).
2. **Regressor:** expected cases at t+4, with a p10–p90 range.
3. **Traffic light** from the probability: 🟢 normal (< 0.3), 🟠 get ready (0.3–0.6), 🔴 stand by (≥ 0.6 and regressor p90 > 2× surge line). These cut-offs are defaults; tune them on training folds only.

### Why it should work (research basis, for the model card)
- Dengue biology creates a 3–8 week delay between weather and hospital cases: breeding 1–2 weeks; virus matures in the mosquito in ~15 days at 25°C vs ~6.5 days at 30°C; people get sick ~6 days after a bite (range 3–10). Source: Chan & Johansson 2012, https://pmc.ncbi.nlm.nih.gov/articles/PMC3511440
- Temperature effect is a hump: transmission between ~18°C and 34°C, peaking ~26–29°C. Source: Mordecai et al. 2017, https://www.biorxiv.org/content/10.1101/063735v3
- Nepal (Chitwan, 2010–2017, negative binomial model): minimum temperature 2 months earlier, max temperature and humidity were significant; rainfall was not. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6417253
- Nepal's EWARS (118 sentinel sites) has no defined outbreak alarm; endemic channels (mean + 2 SD) have been proposed. Source: https://www.frontiersin.org/journals/tropical-diseases/articles/10.3389/fitd.2023.1217939/pdf
- Burden: ~18k cases in 2019; >54k cases and ≥88 deaths in 2022 with hospitals overwhelmed; 51k cases in 2023. Source: https://reliefweb.int/report/nepal/nepal-dengue-response-dref-final-report-mdrnp014
- Climate does not always improve dengue forecasts (city-dependent). Source: https://edoc.hu-berlin.de/18452/34612
- Do NOT claim an "existing Gandaki dengue forecast" in any output. It has not been verified.

## 2. How to work with Pratyush

- Be terse. Short bullets, no long setup narration.
- At each phase, say what you did, what you found and why you made each non-obvious decision (one line each). He wants to understand failure modes, not just get output.
- Never silently "fix" data. Every cleaning step that drops, fills or changes values gets logged in the relevant report.
- Raw data files are read-only. Never edit, move or delete them. Write everything derived to `data/processed/`.
- Ask before deleting anything or installing anything outside `requirements.txt`.
- Fix random seeds (42) everywhere so results are reproducible.

## 3. Run order

Work through the phases in order. Proceed automatically when a phase passes. Stop only at a FAIL gate or when a decision below says "ask".

### Phase 0 — Data audit (gate)

**Goal:** find the data, check its type and quality, and decide whether modelling can start.

1. List every file in the project folder and its subfolders (ignore `.git`, `venv`, `__pycache__`, `node_modules`, and this project's own output folders).
2. For each file, record its type (csv, xlsx, json, parquet, pdf, other) and size.
3. For each tabular file, record: rows, columns, dtypes, date/year-week range, missing values per column, duplicate rows, duplicate (year, week) keys.
4. Match files to the datasets below by column names (case-insensitive, fuzzy, e.g. `dengue`, `dengue_this_week`, `Dengue`; `T2M`, `PRECTOTCORR`, `RH2M`). If a match is ambiguous, list the candidates and pick the most complete one, saying why.
5. Write `reports/data_audit_report.md` (structure in section 6) and give a short summary in chat.

**Required datasets**

| Dataset | Required content | Notes |
|---|---|---|
| A. Weekly dengue cases (EWARS) | `year`, epi `week`, weekly dengue cases | National level. Target 2017–2025. |
| B. Weather (NASA POWER) | date or year+week; temperature (`T2M`, `T2M_MIN`, `T2M_MAX`), precipitation (`PRECTOTCORR`), humidity (`RH2M`); location (district name or lat/lon) | Daily or weekly. Must cover the case period plus 12 weeks before it. |

**Optional datasets** (use if present, warn if absent)
- C. Number of reporting sites per week (or `reporting_rate_pct`).
- D. District dengue situation reports (PDF) — stretch goal only, do not parse in the first pass.
- E. CHIRPS rainfall.

**Usable season** = a calendar year with dengue values for ≥ 40 of 52 weeks AND no more than 3 missing weeks inside the peak window (weeks 22–48).

**Gate rules**
- **FAIL** (stop, report what is missing and how to get it) if any of:
  - no weekly dengue series found;
  - fewer than 4 usable seasons (rolling-origin testing needs ≥ 3 training seasons + 1 test season);
  - no weather data covering the case period;
  - no temperature variable in the weather data.
- **PASS WITH WARNINGS** (continue, list warnings at the top of the report) if any of:
  - fewer than 6 usable seasons;
  - no reporting-site counts for most years;
  - weather from only one location;
  - suspicious values (negative counts, single-week jumps > 10× with no neighbouring rise, totals that disagree strongly with known EDCD annual totals).
- **PASS** otherwise.

**Known data issues — check for these explicitly**
- In the earlier CSV, dengue had only 14 weeks in 2021 and 28 weeks in 2022. That is a parser gap (some bulletin layouts were missed), not a real absence of dengue. If still present, flag it. If `ewars_scraper.py` is in the folder, explain the likely fix and apply it, keeping the old output file untouched and writing the new output alongside it.
- 2025 had only ~5,778 cases vs ~29,110 in 2024. This is probably real (immunity / serotype shifts), not an error. Flag it, don't drop it.
- EWARS pre-computed columns `prev_week`, `change`, `last_year` are inconsistent across bulletin eras. Discard them and recompute from raw weekly values.
- `reporting_rate_pct` and district columns only exist for 2024–2025. Do not use them as model features. If no reporting-site counts exist for earlier years, skip per-site normalisation and warn.
- NASA POWER uses `-999` as a missing-value marker. Convert to NaN before any calculation.
- EWARS URLs need `https://www.edcd.gov.np` (with `www.`). Requests without it fail silently.
- Week alignment: aggregate daily weather to the SAME week definition EWARS uses. Spot-check 3 bulletins' date ranges against your week mapping. If the EWARS convention can't be confirmed, default to ISO weeks and warn.

### Phase 1 — Build the weekly table

1. Clean dataset A: sort, remove duplicate (year, week) rows (keep the most complete, log which), build a continuous weekly index. Do not interpolate cases across gaps longer than 1 week; leave them as NaN and log them.
2. Aggregate weather to weekly per location, then to national:
   - **Primary:** mean across all available districts.
   - **Sensitivity check:** mean across hotspot districts that exist in the data (e.g. Kathmandu, Lalitpur, Bhaktapur, Sunsari, Morang, Jhapa, Chitwan, Kaski, Makwanpur, Dhading).
3. Join A + B (+ C if present) on (year, week).
4. Save `data/processed/weekly_table.csv` and print a short coverage summary (weeks per year, missing per column).

### Phase 2 — Targets

- **Rate:** `rate_t = cases_t / sites_t` if site counts exist for most years, otherwise `rate_t = cases_t` (and say so in the report).
- **Surge line (per week of year):** mean + 2 SD of the rate for that week of year, pooled over a ±2-week window, computed **only from training seasons inside each fold** (never from the full dataset — that leaks the future).
- **Surge onset:** the line must be crossed for 2 consecutive weeks (reduces false alarms from one-week blips).
- **Classifier target:** `y_cls_t = 1` if a surge onset happens in any of weeks t+1 … t+4.
- **Regression targets:** `y_reg_t = cases_{t+4}` (primary) and the sum of cases over t+1 … t+4 (secondary).

### Phase 3 — Features (use only weeks ≤ t)

| Feature | Definition |
|---|---|
| `tmin_mean_4`, `tmin_mean_8` | mean of weekly min temperature over the last 4 / 8 weeks |
| `tmean_mean_lag4_12` | mean temperature over weeks t−12 … t−4 |
| `heat_above_18_8` | sum over last 8 weeks of max(0, weekly mean temp − 18) |
| `rain_sum_4`, `rain_sum_8` | total rainfall over last 4 / 8 weeks |
| `rh_mean_4` | mean relative humidity over last 4 weeks |
| `warm_wet_weeks_8` | count of last 8 weeks with mean temp ≥ 25°C AND RH ≥ 70% (defaults; may tune on training folds only) |
| `log_cases_0` … `log_cases_3` | log(1 + rate) for this week and the 3 previous weeks |
| `growth_2` | log(1 + rate_t) − log(1 + rate_{t−2}) |
| `log_last_season` | log(1 + total cases in the previous calendar year) — immunity stand-in |
| `woy_sin`, `woy_cos` | sin / cos of 2π × week / 52 |

**Two feature sets (for the ablation):**
- **Set A (no weather):** case features + `log_last_season` + `woy_sin`, `woy_cos`.
- **Set B (with weather):** Set A + all weather features.

Add a unit test (`tests/test_leakage.py`) asserting that every feature at row t is computed only from rows ≤ t, and that targets are never used as inputs.

### Phase 4 — Models

Start simple. ML is a challenger, not the default.

1. **Baselines:**
   - Seasonal naive: predict the same week last year.
   - Persistence: predict this week's value.
   - Endemic-channel alarm: predict "surge" if this week is already above the surge line.
2. **Main model (M1):**
   - Regressor: negative binomial GLM (statsmodels), with log(sites) as an offset if site counts exist. If the dispersion estimate is ~0, fall back to Poisson and say so.
   - Classifier: logistic regression.
   - Standardise features, with the scaler fitted on training folds only.
   - Keep to ≤ 8 features in the main model to avoid overfitting; use the strongest ones from the table.
3. **M2 (optional):** M1 with a spline on temperature (patsy `bs`, df = 3) to capture the hump shape.
4. **M3 challenger:** LightGBM — max_depth ≤ 3, num_leaves ≤ 7, min_data_in_leaf ≥ 20, learning_rate 0.05, ≤ 300 trees, objective `poisson` (regressor) / `binary` (classifier). Quantile objective (0.1, 0.9) for the range.

Run every model on both Set A and Set B.

### Phase 5 — Evaluation (honest testing)

- **Rolling origin by season.** Never use a random train/test split. Train on seasons before season k, test on season k, starting once ≥ 3 training seasons exist. Recompute the surge line, scaler and any tuned thresholds inside each fold.
- **Regression metrics:** MAE, plus skill vs seasonal naive = 1 − MAE_model / MAE_baseline.
- **Classifier metrics:** Brier score; PR-AUC if the test season has positives; surges caught ÷ surges; false alarms per season; lead time = surge onset week − first 🟠 week.
- **Ablation (the key result):** Δ = skill(Set B) − skill(Set A), reported **per season**. Only claim "weather helps" if Δ > 0 in most test seasons. If it doesn't help, say so plainly — that is a valid result.
- Save `reports/results.md` and figures in `reports/figures/`.

**Choosing the final model:** prefer the simplest model unless a more complex one beats it by > 5% skill in the majority of test seasons.

### Phase 6 — Final model + web app

1. Fit the chosen model on all usable data. Save it with joblib to `models/` along with the feature list, scaler and surge-line table.
2. Build a Streamlit app at `app/app.py` using Plotly charts:
   - **Overview:** national weekly dengue timeline with the surge line and shaded surge periods.
   - **Back-test:** pick a season → actual vs predicted cases, weekly probability bars, traffic light per week, lead time achieved.
   - **Try it (what-if):** sliders for temperature, rainfall, humidity, recent weekly cases, growth trend and last season's total → probability, traffic light and predicted cases with range.
   - **Model card:** data sources, features, metrics table, ablation result, limitations (national only, few seasons, weather may help timing more than size).
3. Reuse layout ideas from the MonsoonWatch Streamlit app where useful.
4. The app must run with `streamlit run app/app.py` and load the saved model, not retrain on start-up.

## 4. Project structure

Create what's missing; never move the user's raw files.

```
denguewatch/
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── raw/            # user files (read-only); if raw files live elsewhere, read them in place
│   └── processed/      # weekly_table.csv, features.csv
├── src/
│   ├── audit.py
│   ├── build_table.py
│   ├── targets.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   └── train_final.py
├── models/
├── reports/
│   ├── data_audit_report.md
│   ├── results.md
│   └── figures/
├── app/
│   └── app.py
└── tests/
    └── test_leakage.py
```

## 5. Environment

- MacBook, Python 3 (Anaconda base environment), VS Code.
- `requirements.txt`: pandas, numpy, statsmodels, patsy, scikit-learn, lightgbm, streamlit, plotly, joblib, openpyxl, pytest.

## 6. Report formats

**`reports/data_audit_report.md`**
1. Gate status (PASS / PASS WITH WARNINGS / FAIL) and warnings, at the top.
2. File inventory table: file, type, size, matched dataset.
3. Per dataset: columns and dtypes, date range, usable seasons (weeks per year table), missing values, duplicates, suspicious values.
4. If FAIL: exactly what is missing and the concrete next step to get it.

**`reports/results.md`**
1. Final model chosen and why (one paragraph).
2. Per-season table: model, feature set, MAE, skill, Brier, surges caught, false alarms, lead time.
3. Ablation table: Δ per season and a one-line verdict.
4. Limitations.

## 7. Definition of done

- [ ] Audit report written and the gate passed.
- [ ] Weekly table built with logged cleaning steps.
- [ ] Leakage test passes.
- [ ] Baselines, M1 and M3 evaluated with rolling origin on Set A and Set B.
- [ ] Results and ablation reported honestly.
- [ ] Final model saved.
- [ ] Streamlit app runs locally with all four sections.
- [ ] Short chat summary: gate status, chosen model, does weather help (yes/no + Δ), how to run the app.
