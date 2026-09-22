"""
Forecast from the few numbers a surveillance officer actually has.

The app asks for the origin week and the latest four weekly AGE counts. Every
other model input -- the 4-week average, year-on-year growth, last year's
counts, the calendar terms -- is computed here from the stored EWARS history
with the same definitions as src/features/build_features.py, so a row built
here for a past week matches the training row for that week.

The app uses the weather-free ensemble (models/ensemble_noweather_h*.pkl,
built by build_noweather_ensemble.py): weather added no accuracy
(reports/metrics/ablation.csv), so nobody has to look up rainfall.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from src.common.epiweek import epiweek_to_end_date  # noqa: E402
from src.modelling.train_model import predict_rows  # noqa: E402

BASE_PATH = "data/processed/features_base.csv"
MODEL_DIR = "models"
HORIZONS = [1, 2, 3, 4]
FIRST_YEAR = 2021


def t_of(year, week):
    """Position on the complete epi-week grid (2021 wk 1 = 0, 52 weeks/year)."""
    return (int(year) - FIRST_YEAR) * 52 + int(week) - 1


def yw_of(t):
    return FIRST_YEAR + int(t) // 52, int(t) % 52 + 1


def load_history(path=BASE_PATH):
    """National weekly AGE counts indexed by grid position t (gaps are NaN)."""
    b = pd.read_csv(path)
    return pd.Series(b["age_this_week"].values,
                     index=(b.year - FIRST_YEAR) * 52 + b.week - 1,
                     name="age").sort_index()


def origin_range(hist):
    """First and last origin week a forecast can be made from.

    Needs last year's 8-week window (t-60 on record) and, for the 4-week
    forecast, last year's count for the target week.
    """
    return 60, int(hist.index.max()) + 52 - max(HORIZONS)


def load_models(kind="noweather"):
    out = {}
    for h in HORIZONS:
        p = os.path.join(MODEL_DIR, f"ensemble_{kind}_h{h}.pkl")
        if os.path.exists(p):
            out[h] = joblib.load(p)
    return out


def build_row(hist, year, week, recent, horizon):
    """Model input row for one origin week.

    recent: AGE counts for [this week, 1, 2, 3 weeks ago].
    Older weeks come from the history; a week that is not on record is taken
    as the average of `recent` if it is recent, or interpolated between its
    neighbours if it is a year back.
    """
    T = t_of(year, week)
    filled = hist.interpolate(limit_area="inside")
    level = float(np.mean(recent))

    def lag(k):
        if k < 4:
            return float(recent[k])
        v = filled.get(T - k, np.nan)
        return float(v) if np.isfinite(v) else level

    def year_back(t):
        return float(filled.get(t, np.nan))

    a = [lag(k) for k in range(9)]
    last_year_8wk = np.mean([year_back(T - k) for k in range(53, 61)])
    ty, tw = yw_of(T + horizon)
    row = {
        "age_now": a[0], "age_lag1": a[1], "age_lag2": a[2], "age_lag3": a[3],
        "age_roll4_mean": float(np.mean(a[1:5])),
        "age_diff1": a[0] - a[1],
        "age_now_last_year": year_back(T - 52),
        "age_yoy_ratio": float(np.mean(a[1:9]) / last_year_8wk),
        "age_target_week_last_year": year_back(T + horizon - 52),
        "tgt_week_of_year": float(tw),
        "tgt_sin": float(np.sin(2 * np.pi * tw / 52)),
        "tgt_cos": float(np.cos(2 * np.pi * tw / 52)),
        "tgt_is_monsoon": float(22 <= tw <= 39),
    }
    return pd.DataFrame([row]), {"target_year": ty, "target_week": tw,
                                 "target_date": epiweek_to_end_date(ty, tw)}


def forecast(bundle, row):
    """Average of the ensemble members' forecasts, plus each member's own."""
    parts = np.array([
        predict_rows(m["model"], row, bundle["features"], m["framing"],
                     m["blend_w"])[0] for m in bundle["members"]], float)
    return float(np.nanmean(parts)), parts


def forecast_all(hist, models, year, week, recent):
    out = {}
    for h, bundle in models.items():
        row, info = build_row(hist, year, week, recent, h)
        pred, parts = forecast(bundle, row)
        out[h] = {**info, "pred": pred, "members": parts,
                  "last_year": float(row["age_target_week_last_year"].iloc[0])}
    return out
