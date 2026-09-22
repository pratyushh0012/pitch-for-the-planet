# MonsoonWatch — Progress Log

## Project snapshot
- **Goal:** Forecast weekly national AGE cases in Nepal 1–4 weeks ahead (Pitch for the Planet)
- **Current phase:** 8 — complete end to end
- **Last updated:** 2026-09-14

## Environment
- Python 3.13.9 (Anaconda). **Always run scripts with `/opt/anaconda3/bin/python3`** —
  plain `python3` resolves to Homebrew's Python with no pandas/requests.
- Key packages: pandas 2.3.3, numpy 2.3.5, scikit-learn 1.7.2, xgboost 3.4.1,
  lightgbm 4.7.0, matplotlib 3.10.6, streamlit 1.51.0, python-docx 1.2.0
- Run the app: `streamlit run app.py`

## Data status
| Dataset | Path | Rows | Range | Status |
|---|---|---|---|---|
| DHM rainfall (raw, wide) | data/raw/Rain_Data_Subik.csv | 23 × 16,075 | 1980-01-01 → 2023-12-31 | ✅ |
| EWARS weekly | data/raw/ewars_national_weekly.csv | 241 | 2021w1 → 2025w51 | ✅ |
| Rainfall weekly (gauges) | data/interim/rainfall_weekly_dhm.csv | 156 | 2021 → 2023 | ✅ |
| NASA POWER daily / weekly | data/interim/nasa_power_{daily,weekly}.csv | 1826 / 260 | 2021 → 2025 | ✅ |
| Rainfall bridged | data/interim/rainfall_weekly_bridged.csv | 258 | 2021 → 2025 | ✅ |
| Feature base table | data/processed/features_base.csv | 259 | 2021w1 → 2025w51 | ✅ |
| Model tables h1–h4 | data/processed/features_h{1..4}.csv | 203–209 | — | ✅ |

## Phase checklist
- [x] Phase 0 — Scaffold
- [x] Phase 1 — Rainfall prep
- [x] Phase 2 — Epi-week alignment
- [x] Phase 3 — Climate bridge
- [x] Phase 4 — Feature table
- [x] Phase 5 — Training
- [x] Phase 6 — Evaluation
- [x] Phase 7 — Web app
- [x] Phase 8 — Report + pitch assets

## Decisions made
| Decision | Reason |
|---|---|
| DHM 23-station rainfall as primary source | Real gauges beat a single satellite grid point; catches localised cloudbursts |
| Rainfall restricted to 2021+ | Pre-2021 patchy (8–9 stations in 2018–20); also matches EWARS start |
| EDCD-native epi-week, NOT ISO week, everywhere | See alignment finding below |
| Reindex onto a complete epi-week grid before any `shift()` | 18 bulletins are missing; naive shift would make "lag 1" secretly a lag 8 |
| Feature set cut from 39 to 20 (`CORE` in train_model.py) | ~140 training rows; 39 features overfit |
| Four target framings compared, not just level | Reported series quadrupled 2021→2025; trees cannot extrapolate past their training max |
| History length (expand / decay / 104-wk window) treated as a tunable | Non-stationary series — how much history to trust is an empirical question |
| **All selection inside 2021–2023; 2024–2025 untouched** | See "the mistake that mattered" below |
| Stacked panels, never dual-axis, for rain-vs-cases | Two y-scales can manufacture any apparent lead/lag by rescaling |

## Epi-week alignment finding — RESOLVED
EDCD weeks are **Sunday-ending, numbered 1–52**, and are **not ISO weeks**
("Week 51, 2025" = 29 Dec 2025 = ISO week 1 of 2026). Using `.isocalendar()`
would misalign climate data by 1–3 weeks with no error raised.

Anchors (date week 1 ends): 2021 → 2021-01-17 (cached PDFs wk1 & wk5, both
Sunday, 28 days apart); 2022 → 2022-01-16 (independently, two ReliefWeb
bulletins wk15 & wk27, both Sunday, 84 days apart, back-solving to the same
anchor). Shift is exactly 364 days, so later years extrapolate on that stride.
Implemented once in `src/common/epiweek.py` and used by every script.

**Validated, not assumed:** the EDCD-native join gives higher AGE↔rainfall
correlation at every lag 0–4 than the ISO join (0.260 vs 0.222 at lag 0).

⚠️ Bulletin header dates are NOT safe anchors on their own — EDCD publishes
multi-week backlogs on one date (2025 wk 11–16 all stamped "2nd May 2025").

## Climate bridge calibration
DHM gauges end 2023-12-31; EWARS runs to 2025. NASA POWER was fetched over the
full 2021–2025 so 2021–2023 forms a calibration overlap.
- Headline: `rain_mean_mm ~ nasa_rain_sum` → **R² = 0.865**, r = 0.93,
  slope 1.150, intercept 3.605. (Plan's abort threshold was R² < 0.50.)
- Held out 2023 entirely, fit on 2021–22 → **R² = 0.894**. Stable, not overfit.
- Estimated annual totals: 2024 = 1841 mm, 2025 = 1782 mm — inside the range of
  the measured years (1791 / 1830 / 1553).
- Every row carries `rain_source_is_dhm` (1 = gauge, 0 = estimate).

## The mistake that mattered
An earlier version selected the model on 2024 and tested on 2025. Every model
then "lost" to the seasonal-naive baseline by 30–44%, which looked like a real
negative result but was an artefact of the protocol: 2024 (+29% YoY) and 2025
(+14% YoY) reward **opposite** forecasters, so tuning on one anti-selects for
the other. Fixed by moving **all** selection inside 2021–2023 (TimeSeriesSplit
CV) and treating 2024–2025 as one untouched hold-out, scored by rolling origin.
Per-year results are still reported separately so the instability stays visible.

## Model results — deployed forecaster
The deployed forecaster at each horizon is a **top-k ensemble** (k chosen by
leave-one-fold-out inside 2021–2023). Scored on the untouched 2024–2025
hold-out by rolling origin:

| Horizon | k | MAE | RMSE | MAPE | MAE 2024 | MAE 2025 | Best baseline | Baseline MAE | Gain |
|---|---|---|---|---|---|---|---|---|---|
| 1 wk | 15 | 52.4 | 69 | 12.5% | 44.8 | 61.2 | persistence | 55.2 | **+5.1%** |
| 2 wk | 30 | 56.9 | 74 | 12.8% | 51.5 | 62.7 | persistence | 67.0 | **+15.1%** |
| 3 wk | 30 | 57.8 | 75 | 13.6% | 57.6 | 58.0 | seasonal + drift | 70.6 | **+18.1%** |
| 4 wk | 30 | 58.1 | 76 | 13.2% | 58.2 | 58.1 | seasonal + drift | 64.6 | **+10.0%** |

Beats the best of four baselines at **every** horizon. The model's error is
nearly flat across horizons (52 → 58) while baselines degrade sharply, so the
advantage *grows* with lead time — which is where the operational value is.

Authoritative files (regenerated by each run): `reports/metrics/model_comparison.csv`,
`reports/metrics/ensemble_selection.csv`, `reports/metrics/ablation.csv`,
`reports/metrics/training_log.txt`, `models/ensemble_h{1..4}.pkl`.

### Why an ensemble rather than one "best" model
Established from **pre-2024 data only**: CV folds barely agree on which
candidate is best (Kendall τ between fold rankings = +0.07 to +0.22). With
three years of weekly data there is not enough signal to identify a single
winner. Averaging the top-k beats the single CV pick at every horizon
(e.g. h=4: 58.1 vs 70.0).

### ⚠️ Negative result: rainfall does not improve accuracy
Ablation (`reports/metrics/ablation.csv`), same ensemble, same hold-out, same protocol:

| Horizon | Full | Disease+season only | Weather+season only | Rainfall worth |
|---|---|---|---|---|
| 1 wk | 52.4 | 51.7 | 80.4 | −1.4% |
| 2 wk | 56.9 | 55.3 | 64.0 | −2.9% |
| 3 wk | 57.8 | 58.0 | 54.9 | +0.3% |
| 4 wk | 58.1 | 57.6 | 59.0 | −0.9% |

Removing **all** rainfall and climate features changes error by <3%. The 23
gauges were meant to be the project's main advantage over satellite data; at
the national weekly scale they add nothing measurable.

Most likely cause: **spatial mismatch** — valley rainfall vs a national case
count. Also, seasonal sin/cos terms already encode when the monsoon happens, so
the ablation measures what rainfall adds *on top of knowing the date*. And the
raw AGE↔rainfall correlation is only 0.26, peaking at lag 0 rather than at the
multi-week lag a contamination mechanism would predict.

This does **not** break the project — the forecaster still beats every baseline
at every horizon — but it changes the honest claim, and the pitch must not
present rainfall as a demonstrated driver of accuracy. Say it plainly if asked;
it is a much stronger position than being caught overclaiming.

### National rainfall test (2026-09-15) — geography was not the missing piece
The project's NASA POWER data was a single Kathmandu grid cell (the same place
as the valley gauges). To test the "spatial mismatch" explanation above, NASA
POWER was fetched for 67 district HQs across all 7 provinces
(`src/ingest/nasa_power_national.py` → 39 distinct grid cells) and turned into
four indices (`src/features/national_rain.py`): Kathmandu cell, all-Nepal,
Terai, and case-weighted (weights from 2024–25 district top-5 lists — uses
test-period knowledge of *where* cases occur, so an optimistic upper bound).

**Does Kathmandu represent Nepal?** Partly. Weekly correlation with all-Nepal
rain 0.88 (anomaly 0.80), but it falls apart westward: Lumbini 0.71, Karnali
0.70, Sudurpashchim 0.55 (anomalies 0.53 / 0.49 / 0.32). The monsoon peaks at
week 25 in the east but week 31–33 in the west. Kailali (12% of top-5 cases)
and Rupandehi/Kapilbastu sit in that poorly-represented west.

**Does national rain improve the forecast?** No, not measurably
(`src/modelling/national_rain_test.py`, same members / hold-out / rolling
origin as the ablation; harness reproduces ablation.csv's no-weather column
exactly). MAE averaged over h=1–4:

| Feature set | Mean MAE |
|---|---|
| Valley gauges (deployed) | 56.3 |
| No weather | 55.6 |
| Kathmandu cell (NASA) | 56.6 |
| All-Nepal (NASA) | 55.4 |
| Terai (NASA) | 55.3 |
| All-Nepal rain anomaly (NASA) | 55.2 |
| Case-weighted (NASA, optimistic) | 55.4 |

National sources beat the valley gauges by ~1 case/week but only tie
"no weather" (±0.4). Differences of this size are noise on ~62 test weeks.

**Why:** unusual rain vs unusual cases (both de-seasonalised) correlates
*negatively* in 2021–23 (−0.22 to −0.27) and ~0 in 2024–25, flipping positive
in the 2025 monsoon — an unstable relationship a model trained on 2021–23
cannot exploit. Weak hint (r = −0.18, n = 77) that wetter weeks coincide with
lower reporting rates, i.e. floods may disrupt reporting as much as they
spread disease. Not established.

**Consequence:** the null result is not just the wrong rainfall. At the
national weekly scale, rainfall adds nothing on top of case history and the
calendar. A climate signal, if present, would need district-level cases.

### Web app simplified (2026-09-16) — weather-free forecaster
`app.py` rewritten as two tabs: **About** (problem, solution, Graphviz method
flowchart, three result charts, SDGs, limitations) and **Make a forecast**
(pick the bulletin week, enter 4 weekly counts, get 1–4 week forecasts with a
likely range, plus the 2024–2025 track record). Old five-page app kept in
`archive/app_v1_detailed.py` (no longer runs).

The app now uses `models/ensemble_noweather_h{1..4}.pkl`
(`src/modelling/build_noweather_ensemble.py`): the same members and k as the
deployed ensemble, refit on the 12 disease + season features. Hold-out scores
reproduce the ablation exactly:

| Horizon | MAE | MAPE | MAE 2024 | MAE 2025 | Best baseline | Gain |
|---|---|---|---|---|---|---|
| 1 wk | 51.7 | 12.1% | 45.8 | 58.5 | persistence 55.2 | **+6.4%** |
| 2 wk | 55.3 | 12.6% | 50.7 | 60.2 | persistence 67.0 | **+17.5%** |
| 3 wk | 58.0 | 13.6% | 56.8 | 59.1 | seasonal + drift 70.6 | **+17.9%** |
| 4 wk | 57.6 | 13.1% | 58.3 | 56.9 | seasonal + drift 64.6 | **+10.9%** |

`predict.py` was rewritten: features are now built from the stored history
with the exact `build_features.py` definitions (verified: max difference
~1e-15 over all complete training rows). The old version computed
`age_roll4_mean` and `age_yoy_ratio` differently from training. The likely
range in the app is the 10th–90th percentile of actual/forecast on the
hold-out (an empirical interval, not a model-based one).

## Gotchas hit
- EDCD site requires the `www.` prefix or requests fail silently
- `Rain_Data_Subik.csv` final column has a trailing `\r` — strip all column names
- Bulletin `prev_week`/`change`/`last_year` columns are inconsistent across eras — recomputed
- EWARS has **18 missing weeks** (2023: 2, 2024: 7, 2025: 9) — must reindex before lagging
- NASA POWER uses −999 as its fill value — mask it or it silently poisons means
- `TimeSeriesSplit` default folds were too small for the CV pool; needed explicit `test_size`
- Ridge lives in a Pipeline, so sample weights pass as `ridge__sample_weight`

## Next action
Nothing blocking; the pipeline is complete end to end. Highest-value additions,
in order:
1. ~~Rainfall matching the geography of the case counts~~ — done
   2026-09-15 with national NASA POWER; no measurable gain (see above).
   Remaining climate route: district-level cases matched to district rain.
2. **EWARS weekly reporting-site counts for 2021–2023**, so cases can be
   modelled *per reporting site* and the surveillance-expansion trend removed
   at source.
3. Prediction intervals rather than point forecasts (the app shows an
   empirical 80% range from hold-out errors; no model-based interval yet).
