"""
Phase 4 -- Feature table.

Joins the EWARS disease series to the bridged rainfall series and engineers
lag/rolling/seasonal features.

FORECAST FRAMING (this is the leakage-critical part):
  Each row is an ORIGIN WEEK t -- the last week for which data exists at
  forecast time. Every feature is observed AT OR BEFORE t. The target is the
  AGE count at week t+h. So predicting 4 weeks ahead never sees weeks
  t+1..t+4, including their rainfall.

  Seasonal features (week-of-year, sin/cos, is_monsoon) describe the TARGET
  week t+h, not the origin week -- the calendar is known in advance, so this
  is legitimate.

  `age_target_week_last_year` is the count 52 weeks before the TARGET week,
  which is in the past at time t. Legitimate, and it is what the seasonal-naive
  baseline uses.

MISSING WEEKS: EWARS has 18 interior missing weeks (2023: 2, 2024: 7, 2025: 9).
The series is reindexed onto a COMPLETE epi-week grid before any shift(), so a
"lag 1" is always genuinely one calendar week back and never silently jumps a
gap. Rows whose features or target land on a missing week are dropped.

Outputs:
  data/processed/features_base.csv     (origin-week features, for the app)
  data/processed/features_h{1..4}.csv  (model-ready, one per horizon)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from src.common.epiweek import epiweek_to_end_date  # noqa: E402

EWARS = "data/raw/ewars_national_weekly.csv"
RAIN = "data/interim/rainfall_weekly_bridged.csv"
OUT_BASE = "data/processed/features_base.csv"
HORIZONS = [1, 2, 3, 4]

RAIN_COLS = ["rain_mean_mm", "rain_max_mm", "rain_std_mm",
             "wet_days", "max_daily_mm"]
CLIM_COLS = ["temp_c_mean", "temp_c_max", "humidity_pct_mean"]


def build_grid(dis, rain):
    """Complete (year, week) grid spanning the disease series."""
    years = range(int(dis.year.min()), int(dis.year.max()) + 1)
    grid = pd.DataFrame([(y, w) for y in years for w in range(1, 53)],
                        columns=["year", "week"])
    lo = (dis.year.min(), dis[dis.year == dis.year.min()].week.min())
    hi = (dis.year.max(), dis[dis.year == dis.year.max()].week.max())
    key = grid.year * 100 + grid.week
    grid = grid[(key >= lo[0] * 100 + lo[1]) & (key <= hi[0] * 100 + hi[1])]
    return grid.reset_index(drop=True)


def main():
    dis = pd.read_csv(EWARS)[["year", "week", "age_this_week"]]
    rain = pd.read_csv(RAIN)
    print(f"[load] EWARS {len(dis)} weeks, rainfall {len(rain)} weeks")

    grid = build_grid(dis, rain)
    print(f"[grid] complete epi-week grid: {len(grid)} weeks "
          f"({grid.iloc[0].year}w{grid.iloc[0].week} -> "
          f"{grid.iloc[-1].year}w{grid.iloc[-1].week})")

    df = (grid.merge(dis, on=["year", "week"], how="left")
               .merge(rain, on=["year", "week"], how="left"))
    n_missing_age = df.age_this_week.isna().sum()
    n_missing_rain = df.rain_mean_mm.isna().sum()
    print(f"[grid] missing AGE weeks:  {n_missing_age} "
          f"{df[df.age_this_week.isna()].apply(lambda r: f'{int(r.year)}w{int(r.week)}', axis=1).tolist()}")
    print(f"[grid] missing rain weeks: {n_missing_rain}")

    df = df.sort_values(["year", "week"]).reset_index(drop=True)
    df["t"] = np.arange(len(df))
    df["date"] = [epiweek_to_end_date(y, w)
                  for y, w in zip(df.year, df.week)]

    a = df["age_this_week"]

    # ---- Disease features, observed at or before origin week t ----
    df["age_now"] = a
    for L in (1, 2, 3, 4):
        df[f"age_lag{L}"] = a.shift(L)
    df["age_roll4_mean"] = a.shift(1).rolling(4).mean()
    df["age_roll4_std"] = a.shift(1).rolling(4).std()
    df["age_roll8_mean"] = a.shift(1).rolling(8).mean()
    df["age_diff1"] = a - a.shift(1)
    df["age_diff2"] = a - a.shift(2)
    df["age_now_last_year"] = a.shift(52)
    # Level trend: how much higher is this year running vs a year ago?
    df["age_yoy_ratio"] = (a.shift(1).rolling(8).mean()
                           / a.shift(53).rolling(8).mean())

    # ---- Rainfall features, observed at or before t ----
    for c in RAIN_COLS + CLIM_COLS:
        df[c] = df[c]
    for L in (1, 2, 3, 4):
        df[f"rain_mean_lag{L}"] = df["rain_mean_mm"].shift(L)
    df["rain_mean_roll3"] = df["rain_mean_mm"].rolling(3).sum()
    df["rain_mean_roll6"] = df["rain_mean_mm"].rolling(6).sum()
    df["max_daily_lag1"] = df["max_daily_mm"].shift(1)
    df["max_daily_lag2"] = df["max_daily_mm"].shift(2)
    df["wet_days_roll3"] = df["wet_days"].rolling(3).sum()
    df["temp_lag1"] = df["temp_c_mean"].shift(1)
    df["humidity_lag1"] = df["humidity_pct_mean"].shift(1)
    df["humidity_roll3"] = df["humidity_pct_mean"].rolling(3).mean()

    df.to_csv(OUT_BASE, index=False)
    print(f"[save] base table {df.shape} -> {OUT_BASE}")

    feature_cols = [c for c in df.columns if c not in
                    ("year", "week", "age_this_week", "t", "date",
                     "n_stations")]

    # ---- Per-horizon tables ----
    for h in HORIZONS:
        d = df.copy()
        d["horizon"] = h
        d["target"] = a.shift(-h)
        d["target_year"] = d["year"].shift(-h)
        d["target_week"] = d["week"].shift(-h)
        # Seasonality of the TARGET week (calendar is known ahead of time)
        tw = d["target_week"]
        d["tgt_week_of_year"] = tw
        d["tgt_sin"] = np.sin(2 * np.pi * tw / 52)
        d["tgt_cos"] = np.cos(2 * np.pi * tw / 52)
        d["tgt_is_monsoon"] = tw.between(22, 39).astype(float)
        # Same week last year, relative to the TARGET week
        d["age_target_week_last_year"] = a.shift(-h + 52)

        cols = (["year", "week", "date", "horizon", "target_year",
                 "target_week", "target"]
                + feature_cols
                + ["tgt_week_of_year", "tgt_sin", "tgt_cos", "tgt_is_monsoon",
                   "age_target_week_last_year"])
        d = d[cols]

        rows_in = len(d)
        # Must have: a target, a current level (needed by every framing and
        # baseline), and the core lag block.
        need = ["target", "age_now", "age_lag1", "age_lag2", "age_lag3",
                "rain_mean_mm", "rain_mean_lag4", "age_roll4_mean"]
        d = d.dropna(subset=need)
        out = f"data/processed/features_h{h}.csv"
        d.to_csv(out, index=False)
        print(f"[h={h}] {rows_in} grid rows -> {len(d)} usable rows "
              f"({rows_in - len(d)} dropped for missing target/lags) "
              f"| origin {int(d.iloc[0].year)}w{int(d.iloc[0].week)} -> "
              f"{int(d.iloc[-1].year)}w{int(d.iloc[-1].week)} -> {out}")

    print(f"\n[features] {len(feature_cols) + 5} predictors per horizon")


if __name__ == "__main__":
    main()
