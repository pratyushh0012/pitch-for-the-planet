# MonsoonWatch — Project Plan & Methodology

> **For Claude Code:** This is the master plan. Read this first in every session.
> Before doing anything, also read `PROGRESS.md` (create it if missing — see Phase 0).
> After every meaningful step, update `PROGRESS.md`.

---

## 1. What this project is

**MonsoonWatch** forecasts weekly Acute Gastroenteritis (AGE) outbreak risk in Nepal,
1–4 weeks ahead, using rainfall + past disease counts.

Built for Macquarie University's *Pitch for the Planet* competition.
SDG framing: **SDG 3** (Health) + **SDG 6** (Clean Water), supported by **SDG 13** (Climate).

**Why it matters:** Nepal's monsoon (June–Sept) floods contaminate drinking water, causing
AGE spikes that mostly hit children under 5. Nepal's current system (EWARS) only *counts*
cases after they happen — it does not predict.

**What is novel here:** The modelling approach is already validated in the literature
(a 2024 UMD-led study forecast diarrhoeal disease in Nepal/Vietnam/Taiwan using ML).
Our contribution is **not a new algorithm** — it is:
1. Using **recent** data (2021–2025 EWARS) instead of pre-2014 data,
2. Using **real ground rain gauges** (23 DHM stations) instead of a single satellite grid point,
3. Delivering a **runnable local pipeline + web app**, not just a paper.

Be honest about this framing in all pitch materials. Do not overclaim novelty.

---

## 2. Environment & constraints

- **Machine:** MacBook Pro, local execution only. No cloud, no GPU needed.
- **Python:** 3 (Anaconda base env).
- **Stack:** pandas, numpy, scikit-learn, xgboost (or lightgbm), matplotlib, streamlit.
- **The web app must run 100% locally** (`streamlit run app.py`) using local CPU. No hosted inference.
- **Deadline:** mid-September. Prioritise a *working end-to-end model* over polish.

---

## 3. Data assets

### 3.1 Rainfall — `Rain_Data_Subik.csv` (PRIMARY, already on disk)

DHM (Dept. of Hydrology & Meteorology) daily rainfall, Kathmandu Valley.

| Property | Value |
|---|---|
| Shape | 23 rows × 16,075 columns |
| Rows | One row per **station** (23 stations) |
| Columns 1–4 | `Station_ID`, `Longitude`, `Latitude`, `Elevation` |
| Columns 5+ | One column **per day**, named `Jan_01_1980` … `Dec_31_2023` |
| Date range | 1980-01-01 → **2023-12-31** (16,071 days) |
| Elevation range | 1194 m – 1658 m |
| Format | **WIDE** — must be melted to long format |

**Data quality (already verified — do not re-derive, just trust this):**
- Overall missingness per station ranges 2%–53%, but this is almost entirely a
  **pre-2021 problem**.
- For **2021–2023**: every station is 0–3% missing. Monsoon months (Jun–Sep) 2021–2023
  are effectively complete across all 23 stations.
- Some stations had gaps in 2018–2020 (only 8–9 stations reporting), fully recovered by 2021.
- Missing values appear as **empty cells → NaN**.
- Some cells contain `0.01` — treat as trace rainfall (a real value, not a null).

**Known valley-average annual rainfall** (sanity check your aggregation against this):
2021 ≈ 1791 mm, 2022 ≈ 1830 mm, 2023 ≈ 1553 mm.

### 3.2 Disease — `ewars_national_weekly.csv` (already built)

Produced by the existing `ewars_scraper.py`, which parses EDCD EWARS weekly PDF bulletins
across three layout eras (2021–2025).

| Property | Value |
|---|---|
| Rows | ~241 clean weekly rows |
| Reliable columns | `year`, `week`, `age_this_week` — **these three only** |
| Missing AGE values | Zero |
| Signal | Clear monsoon seasonality |

**Columns to DISCARD — do not use as features:**
- `prev_week`, `change`, `last_year` — pre-computed by the bulletins, inconsistent across
  eras. Recompute these yourself from `age_this_week` if needed.
- `reporting_rate_pct` and any district-level columns — only populate for 2024–2025.

**Source note:** There is **no CSV or API** for EWARS anywhere. Only weekly PDFs at
`https://www.edcd.gov.np/ewars` (note: the `www.` prefix is required — requests without it
fail silently). The scraper is the only route. Do not waste time hunting for a bulk download.

### 3.3 Climate bridge — NASA POWER (for 2024–2025 only)

**The gap:** DHM rainfall ends Dec 2023. EWARS runs into 2025.
So ~1.5 years of the disease series has no DHM rainfall.

**Strategy:** Use the existing `nasa_power_fetch.py` (Kathmandu point: rainfall, temperature,
humidity) to cover **2024-01-01 → present**. Blend it with DHM for the overlap period so the
two sources are calibrated (see Phase 3).

---

## 4. Phased execution plan

Work through phases in order. Do not skip ahead. Update `PROGRESS.md` after each phase.

---

### Phase 0 — Scaffold & progress file

1. Create this folder structure:

```
monsoonwatch/
├── PROJECT_PLAN.md          # this file
├── PROGRESS.md              # living log — see below
├── CLAUDE.md                # short pointer file for Claude Code
├── requirements.txt
├── data/
│   ├── raw/                 # Rain_Data_Subik.csv, ewars_national_weekly.csv
│   ├── interim/             # cleaned intermediates
│   └── processed/           # final feature table
├── src/
│   ├── rainfall_prep.py
│   ├── climate_bridge.py
│   ├── build_features.py
│   ├── train_model.py
│   └── evaluate.py
├── models/                  # saved .pkl / .json model artefacts
├── notebooks/               # exploratory only
├── reports/
│   └── figures/
└── app.py                   # Streamlit local web app
```

2. **Create `PROGRESS.md`** using the template in Section 6 of this file.
   This is mandatory. It is how work resumes across sessions.

3. `requirements.txt`: pandas, numpy, scikit-learn, xgboost, matplotlib, streamlit, joblib, requests.

4. `CLAUDE.md` — a 10-line file saying: *"Read PROJECT_PLAN.md then PROGRESS.md before
   starting. Update PROGRESS.md after every step. Data lives in data/raw/."*

---

### Phase 1 — Rainfall preparation (`src/rainfall_prep.py`)

**Goal:** turn the wide 23-station daily file into a tidy weekly rainfall table.

Steps:
1. Load `Rain_Data_Subik.csv`.
2. Split metadata (`Station_ID`, `Longitude`, `Latitude`, `Elevation`) from the daily columns.
3. Parse daily column names with `pd.to_datetime(cols.str.strip(), format='%b_%d_%Y')`.
   **Note:** the final column has a trailing `\r` — strip it.
4. **Melt** to long format: `station_id | date | rainfall_mm`.
5. Filter to **2021-01-01 onwards** (earlier data is too patchy and predates the EWARS series).
6. Map each date to an **ISO epi-week** (`year`, `week`) using `isocalendar()`.
7. Aggregate to weekly, **per station**: sum of daily rainfall.
8. Aggregate across stations into valley-level weekly features:
   - `rain_mean_mm` — mean weekly total across the 23 stations (the main signal)
   - `rain_max_mm` — max across stations (catches **localised cloudbursts** a single grid
     point would miss; these plausibly drive local water contamination)
   - `rain_std_mm` — spread across stations (how uneven the rain was)
   - `wet_days` — mean count of days with rainfall > 1 mm
   - `max_daily_mm` — the single heaviest station-day that week (flood proxy)
9. Save → `data/interim/rainfall_weekly_dhm.csv`.

**Validation before moving on:**
- Sum weekly → annual and check it lands near 1791 / 1830 / 1553 mm for 2021/22/23.
- Confirm no week has fewer than ~20 stations contributing.
- Plot weekly `rain_mean_mm` — it must show an obvious June–Sept hump.

---

### Phase 2 — Epi-week alignment check (CRITICAL — do not skip)

This is the single most likely silent bug in the whole project.

**Problem:** DHM dates are Gregorian calendar days. EWARS week numbers come from EDCD's
bulletin convention. These may not use the same week-start day or the same year-boundary rule.

**Do this:**
1. Take 5–10 EWARS rows with known week numbers where the bulletin also states a date
   (e.g. "12th Epidemiological Week, 2023 · Sunday, 2nd April 2023").
2. Check: does ISO week for 2023-04-02 equal 12 or 13?
3. If there is an offset, **document it in `PROGRESS.md`** and apply a consistent shift
   in `build_features.py`.
4. Also handle the year-boundary edge case (ISO week 1 can start in late December).

Do not proceed to Phase 4 until this is settled and written down.

---

### Phase 3 — Climate bridge for 2024–2025 (`src/climate_bridge.py`)

1. Run `nasa_power_fetch.py` for Kathmandu, **2021-01-01 → present** (deliberately overlapping
   DHM, not just 2024+).
2. Aggregate NASA POWER to the same weekly epi-week structure.
3. **Calibrate on the overlap (2021–2023):** regress `rain_mean_mm` (DHM) on NASA POWER
   weekly rainfall. Record R² and the scaling coefficient in `PROGRESS.md`.
4. For 2024–2025, apply that calibration to produce a **DHM-equivalent** rainfall estimate.
5. Add a boolean column `rain_source_is_dhm` so the model (and the pitch) can distinguish
   ground-truth weeks from estimated weeks.

**If the overlap R² is poor (< ~0.5):** do not force it. Instead, train and report the model
on **2021–2023 only** (where you have real gauge data) and state clearly in the pitch that
2024–2025 is out of scope pending data access. A smaller honest result beats a bigger shaky one.

---

### Phase 4 — Feature table (`src/build_features.py`)

Join disease + rainfall on `year` + `week` (with the Phase 2 offset applied).

**Target variable:** `age_this_week`.

**Features to engineer** (all from data you actually have — no invented variables):

*Rainfall, current week:*
- `rain_mean_mm`, `rain_max_mm`, `rain_std_mm`, `wet_days`, `max_daily_mm`

*Rainfall lags* (the core mechanism — contamination takes time to become illness):
- `rain_mean_lag1`, `rain_mean_lag2`, `rain_mean_lag3`, `rain_mean_lag4`
- `rain_mean_roll3` — 3-week rolling sum (cumulative saturation)
- `max_daily_lag1`, `max_daily_lag2` — heavy-event lags

*Disease autoregressive:*
- `age_lag1`, `age_lag2`, `age_lag3`
- `age_roll4_mean` — recent level
- `age_same_week_last_year` — recompute from `age_this_week`, do **not** use the
  bulletin's `last_year` column

*Seasonality:*
- `week_of_year`, plus `sin(2πw/52)` and `cos(2πw/52)`
- `is_monsoon` — boolean, weeks covering roughly June–September

Save → `data/processed/features_weekly.csv`.

**Drop the first ~5 rows** (they have incomplete lags). Log how many rows survive.

---

### Phase 5 — Model training (`src/train_model.py`)

**Task framing:** Train **four separate models**, one per horizon (1, 2, 3, 4 weeks ahead).
Each predicts `age_this_week` shifted forward by *h* weeks.

**Model:** XGBoost regressor (fall back to LightGBM if install is troublesome).
Keep it small — this is ~240 rows, so heavy models will overfit instantly.
Suggested starting point: `max_depth=3`, `n_estimators=200`, `learning_rate=0.05`, `subsample=0.8`.

**Validation — this must be time-aware:**
- **Never use random train/test split.** Time series leaks badly with random splits.
- Use `TimeSeriesSplit` for cross-validation during tuning.
- Hold out the **final ~20% of weeks chronologically** as the test set, untouched until the end.

**Baselines to beat** (report all three — a model that doesn't beat these is not useful):
1. **Persistence:** predict `age_lag1` (this week = last week).
2. **Seasonal naive:** predict `age_same_week_last_year`.
3. **Rolling mean:** predict `age_roll4_mean`.

Save models to `models/` with `joblib`. Save feature names alongside.

---

### Phase 6 — Evaluation (`src/evaluate.py`)

Report, for each horizon h = 1,2,3,4:
- **MAE** and **RMSE** on the held-out test set
- Same metrics for all three baselines
- **% improvement over the best baseline** — this is your headline pitch number
- Feature importance plot (which lags actually matter — expect rainfall lag-2/lag-3 to rank high
  if the mechanism is real)
- Actual-vs-predicted time series plot for the test period → `reports/figures/`

**Be sceptical of good results.** If the model dramatically beats baselines, first check for
leakage: is any feature accidentally using future information? Re-check the lag shifts.

Write the final numbers into `PROGRESS.md`.

---

### Phase 7 — Local web app (`app.py`)

**Streamlit, runs locally:** `streamlit run app.py`

Four sections:
1. **Overview** — one paragraph on the problem + SDG framing.
2. **Data explorer** — weekly rainfall vs AGE cases on a dual-axis chart, year selector.
   Optionally a map of the 23 stations using their lat/lon.
3. **Forecast** — load the saved models, show the 1–4 week ahead forecast from the most
   recent week in the data, with the historical series behind it.
4. **Model performance** — the metrics table and actual-vs-predicted plot from Phase 6.

**Constraints:**
- Load models once with `@st.cache_resource`; load data with `@st.cache_data`.
- No external API calls at runtime. Everything reads from local files.
- Keep it simple and fast. This is a pitch demo, not a product.

---

### Phase 8 — Pitch assets

- Tagline options
- One-slide methodology diagram (data → features → model → forecast)
- Per-SDG justification (3, 6, 13) — each in one or two sentences
- Headline result: "*X*% better than the current best-guess baseline at *h* weeks ahead"
- **Q&A prep — anticipate these:**
  - *"Isn't this already done?"* → Yes, at prototype level in 2024 research. Ours uses newer
    data, real rain gauges, and actually runs.
  - *"Why national AGE and not district-level?"* → EWARS district columns only populate
    2024–2025; the multi-year seasonality needed for a monsoon model requires the national series.
  - *"What about the 2024–2025 rainfall gap?"* → Answer honestly, cite the Phase 3 calibration.
  - *"Why not cholera?"* → Too few confirmed cases to model.

---

## 5. Rules for Claude Code

1. **Read `PROGRESS.md` at the start of every session.** Do not assume prior state.
2. **Update `PROGRESS.md` after every phase** — and after any surprise, bug, or decision.
3. **One phase at a time.** Show the output, let the human check it, then continue.
4. **Never silently drop rows.** Always print how many rows went in and how many came out.
5. **Print shapes and date ranges** after every major transform.
6. **If a validation check fails, stop and say so.** Do not patch around it quietly.
7. **No invented data.** If something is missing, say it is missing.
8. **Prefer plain-language explanations** of anything statistical — explain the *why*, not just
   the code. Use an everyday analogy where it helps.

---

## 6. `PROGRESS.md` template

Create this file in Phase 0 and keep it current. It is the resume-anywhere file.

```markdown
# MonsoonWatch — Progress Log

## Project snapshot
- **Goal:** Forecast weekly AGE cases in Nepal 1–4 weeks ahead (Pitch for the Planet)
- **Deadline:** mid-September
- **Current phase:** <n>
- **Last updated:** YYYY-MM-DD

## Environment
- Python: <version>
- Key packages + versions:
- Command to run app: `streamlit run app.py`

## Data status
| Dataset | Path | Rows | Date range | Status |
|---|---|---|---|---|
| DHM rainfall (raw) | data/raw/Rain_Data_Subik.csv | 23 | 1980-01-01 → 2023-12-31 | ✅ on disk |
| EWARS weekly | data/raw/ewars_national_weekly.csv | ~241 | 2021 → 2025 | ✅ on disk |
| Rainfall weekly | data/interim/rainfall_weekly_dhm.csv | | | ⬜ |
| NASA POWER bridge | data/interim/nasa_power_weekly.csv | | | ⬜ |
| Feature table | data/processed/features_weekly.csv | | | ⬜ |

## Phase checklist
- [ ] Phase 0 — Scaffold
- [ ] Phase 1 — Rainfall prep
- [ ] Phase 2 — Epi-week alignment
- [ ] Phase 3 — Climate bridge
- [ ] Phase 4 — Feature table
- [ ] Phase 5 — Training
- [ ] Phase 6 — Evaluation
- [ ] Phase 7 — Web app
- [ ] Phase 8 — Pitch assets

## Decisions made
<!-- Date | Decision | Why -->
| Date | Decision | Reason |
|---|---|---|
| | Use DHM 23-station rainfall as primary source | Real gauges beat single satellite grid point |
| | Restrict rainfall to 2021+ | Pre-2021 too patchy; also matches EWARS start |

## Epi-week alignment finding
<!-- Fill in during Phase 2. Offset: none / +1 / -1. Evidence: -->

## Model results
| Horizon | MAE | RMSE | Best baseline MAE | % improvement |
|---|---|---|---|---|
| 1 week | | | | |
| 2 weeks | | | | |
| 3 weeks | | | | |
| 4 weeks | | | | |

## Gotchas hit
<!-- Anything that wasted time. Save future-you the trouble. -->
- EDCD site requires `www.` prefix or requests fail silently
- `Rain_Data_Subik.csv` final column has trailing `\r`
- Bulletin columns `prev_week`/`change`/`last_year` are inconsistent — recompute instead

## Next action
<!-- The single next thing to do. Be specific. -->
```

---

## 7. Definition of done

- [ ] A trained model that **measurably beats all three baselines** at ≥1 forecast horizon
- [ ] A local Streamlit app that loads and shows a forecast
- [ ] `PROGRESS.md` complete enough that someone else could pick it up cold
- [ ] Honest documentation of every limitation (data gap, national-only scope, no district detail)
