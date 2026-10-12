# MonsoonWatch

Forecasting monsoon diarrhoeal-disease (AGE) risk in Nepal, 1–4 weeks ahead.
Built for Macquarie University's *Pitch for the Planet*.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Everything is pre-built and committed, so the app runs immediately. Use
`/opt/anaconda3/bin/python3` explicitly on this machine — plain `python3`
resolves to Homebrew's Python, which has no pandas.

## Rebuild the whole pipeline

Run every script from the project root — all paths are relative to it.

```bash
# optional: re-scrape EWARS bulletins into data/raw/ (network, slow)
python src/ingest/ewars_scraper.py --years 2021 2022 2023 2024 2025

python src/ingest/rainfall_prep.py        # 23-station wide file -> weekly valley features
python src/ingest/nasa_power_fetch.py     # satellite climate (network)
python src/features/climate_bridge.py     # calibrate satellite to gauges, extend to 2025
python src/features/build_features.py     # join, engineer lags, one table per horizon
python src/modelling/train_model.py       # search 180 candidates/horizon  (~1 hour)
python src/modelling/select_ensemble.py   # choose ensemble size, build deployed models
python src/modelling/ablation.py          # does rainfall actually help?   (~15 min)
python src/ingest/nasa_power_national.py # NASA POWER at 67 district HQs (network)
python src/features/national_rain.py     # national rain indices + representativeness
python src/modelling/national_rain_test.py # does NATIONAL rain help? (~30 min)
python src/modelling/build_noweather_ensemble.py # weather-free ensemble the app uses (~7 min)
python src/reporting/make_figures.py all  # every figure in the report
python src/reporting/make_report.py       # the Word report
python src/reporting/make_poster.py       # the A3 pitch poster (PowerPoint)
```

## Project structure

| Path | What it is |
|---|---|
| `app.py` | Streamlit web app — two tabs: About (background, method flowchart, results) and Make a forecast |
| `src/common/` | Shared helpers: EDCD epi-week calendar, chart style |
| `src/ingest/` | Sources → tidy tables: EWARS PDF scraper, NASA POWER fetch, DHM rain-gauge prep |
| `src/features/` | Satellite-to-gauge climate bridge, per-horizon feature tables |
| `src/modelling/` | Training, ensemble selection, ablation, weather-free ensemble, and `predict.py` (used by the app) |
| `src/reporting/` | Figures and the Word report |
| `data/raw/` `data/interim/` `data/processed/` | Inputs, intermediates, model-ready tables |
| `models/` | Deployed forecasters `ensemble_h{1..4}.pkl`, single-model fallbacks, `champions.json` |
| `reports/MonsoonWatch_Project_Report.docx` | Full project report |
| `reports/figures/` | All figures |
| `reports/metrics/` | Backtests, model comparison, ensemble selection, ablation, training log |
| `docs/PROJECT_PLAN.md` | The original plan |
| `docs/PROGRESS.md` | Living log: decisions, gotchas, results |

## Headline results

Scored on an untouched 2024–2025 hold-out, rolling origin (retrain each week):

| Horizon | MAE | Best baseline | Gain |
|---|---|---|---|
| 1 week | 52.4 | 55.2 (persistence) | +5.1% |
| 2 weeks | 56.9 | 67.0 (persistence) | +15.1% |
| 3 weeks | 57.8 | 70.6 (seasonal+drift) | +18.1% |
| 4 weeks | 58.1 | 64.6 (seasonal+drift) | +10.0% |

Beats the best of four baselines at every horizon, and the margin **grows with
lead time** — model error is nearly flat (52→58) while baselines degrade.

**Read `docs/PROGRESS.md` before pitching.** It records one result that matters:
the rain-gauge data does *not* improve accuracy at the national weekly scale.

---

# DengueWatch Nepal

4-week-ahead national dengue surge forecasts (spec: `dengue_project.md`). Shares the repo with MonsoonWatch; its own
files are listed below.

```bash
streamlit run app/app.py                       # app (loads models/dengue/denguewatch.joblib, no retraining)

# rebuild from scratch (run from project root, /opt/anaconda3/bin/python)
python src/ingest/ewars_harvest.py             # download all EWARS bulletins 2016-2025 (network)
python src/ingest/ewars_dengue_parse.py        # -> data/raw/ewars_dengue_weekly_v2.csv
python src/ingest/nasa_power_v2.py             # NASA POWER incl. T2M_MIN/MAX (network); then merge to
                                               #    data/raw/nasa_power_v2_daily.csv.gz (see data_audit_report)
python src/audit.py                            # Phase 0 gate -> reports/data_audit_report.md
python src/build_table.py                      # -> data/processed/weekly_table.csv, reports/cleaning_log.md
python -m pytest tests                         # leakage tests
python src/evaluate.py                         # rolling-origin back-test -> reports/results.md, figures/dengue/
python src/train_final.py                      # -> models/dengue/denguewatch.joblib
```

Note: this folder is iCloud-synced and the disk is nearly full, so macOS evicts files right after they are written.
That's why bulletin texts live in one archive (`data/raw/ewars_bulletin_texts.jsonl.gz`); set `EWARS_PDF_DIR` to a
local folder when re-harvesting.

## Put the MonsoonWatch demo online (Streamlit Community Cloud)

The QR code on the slides (`reports/MonsoonWatch_3MT_slides.pptx`, `reports/figures/dengue/demo_qr.png`) points to
**https://monsoonwatch-nepal.streamlit.app**. To make that address work:

1. Push this folder to a GitHub repository (public, or private with Streamlit access).
2. Go to https://share.streamlit.io → **Create app** → **Deploy a public app from GitHub**.
3. Repository: your repo · Branch: `main` · Main file path: `app/app.py`.
4. App URL: type **`monsoonwatch-nepal`** (must match the QR code).
5. **Advanced settings** → Python version **3.13** (the saved model was built with it).
6. Deploy. Streamlit installs `app/requirements.txt` automatically.
7. Scan the QR on your phone to check it opens.

If `monsoonwatch-nepal` is already taken, pick another name and regenerate the QR for the new address.
