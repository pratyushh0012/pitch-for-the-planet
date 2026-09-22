"""
Phase 1 -- DHM rainfall preparation.

Turns data/raw/Rain_Data_Subik.csv (23 stations x 16,071 daily columns, WIDE)
into a tidy weekly valley-level rainfall table on the EDCD epi-week calendar.

Output: data/interim/rainfall_weekly_dhm.csv
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from src.common.epiweek import assign_epiweek  # noqa: E402

RAW = "data/raw/Rain_Data_Subik.csv"
OUT_WEEKLY = "data/interim/rainfall_weekly_dhm.csv"
OUT_STATIONS = "data/interim/rainfall_stations.csv"
OUT_DAILY = "data/interim/rainfall_daily_valley.csv"

META = ["Station_ID", "Longitude", "Latitude", "Elevation"]
START = "2021-01-01"


def load_wide():
    df = pd.read_csv(RAW)
    # Final column name carries a trailing \r; strip whitespace off all names.
    df.columns = [c.strip() for c in df.columns]
    print(f"[load] raw shape: {df.shape[0]} stations x {df.shape[1]} cols")
    return df


def melt_long(df):
    day_cols = [c for c in df.columns if c not in META]
    dates = pd.to_datetime(pd.Series(day_cols), format="%b_%d_%Y")
    print(f"[melt] {len(day_cols)} daily columns, "
          f"{dates.min().date()} -> {dates.max().date()}")

    long = df.melt(id_vars=META, value_vars=day_cols,
                   var_name="daycol", value_name="rainfall_mm")
    rows_in = len(long)
    long["date"] = pd.to_datetime(long["daycol"], format="%b_%d_%Y")
    long = long.drop(columns=["daycol"])
    long = long.rename(columns={"Station_ID": "station_id"})
    long["rainfall_mm"] = pd.to_numeric(long["rainfall_mm"], errors="coerce")
    print(f"[melt] long rows: {rows_in}")
    return long


def filter_recent(long):
    rows_in = len(long)
    long = long[long["date"] >= START].copy()
    print(f"[filter] >= {START}: {rows_in} -> {len(long)} rows "
          f"({rows_in - len(long)} dropped as pre-2021)")

    n_null = long["rainfall_mm"].isna().sum()
    print(f"[filter] missing daily values in window: {n_null} "
          f"({100 * n_null / len(long):.2f}%)")
    long = long.dropna(subset=["rainfall_mm"])
    print(f"[filter] after dropping NaN station-days: {len(long)} rows")
    return long


def weekly_features(long):
    long = assign_epiweek(long, date_index=False, date_col="date")

    # Per station, per epi-week
    per_station = long.groupby(["year", "week", "station_id"]).agg(
        station_week_mm=("rainfall_mm", "sum"),
        station_wet_days=("rainfall_mm", lambda s: int((s > 1.0).sum())),
        station_max_daily=("rainfall_mm", "max"),
        station_days=("rainfall_mm", "size"),
    ).reset_index()

    # Drop partial weeks at the edges of the record (a full epi-week needs 7
    # days of observation for at least the typical station).
    days_per_week = per_station.groupby(["year", "week"])["station_days"].max()
    partial = days_per_week[days_per_week < 7].index.tolist()
    if partial:
        print(f"[weekly] dropping {len(partial)} partial epi-weeks: {partial}")
        per_station = per_station[
            ~per_station.set_index(["year", "week"]).index.isin(partial)
        ]

    weekly = per_station.groupby(["year", "week"]).agg(
        rain_mean_mm=("station_week_mm", "mean"),
        rain_max_mm=("station_week_mm", "max"),
        rain_std_mm=("station_week_mm", "std"),
        wet_days=("station_wet_days", "mean"),
        max_daily_mm=("station_max_daily", "max"),
        n_stations=("station_id", "nunique"),
    ).reset_index()

    print(f"[weekly] {len(weekly)} epi-weeks, "
          f"{weekly.year.min()}w{weekly.week.min()} -> "
          f"{weekly.year.max()}w{weekly.week.max()}")
    return weekly, per_station


def validate(long, weekly):
    print("\n--- VALIDATION ---")
    ok = True

    # 1. Annual valley-average rainfall vs known DHM figures
    known = {2021: 1791, 2022: 1830, 2023: 1553}
    ann = (long.assign(cy=long["date"].dt.year)
               .groupby(["cy", "station_id"])["rainfall_mm"].sum()
               .groupby("cy").mean())
    for y, expect in known.items():
        if y in ann.index:
            got = ann.loc[y]
            pct = 100 * (got - expect) / expect
            flag = "OK " if abs(pct) < 10 else "!! "
            if abs(pct) >= 10:
                ok = False
            print(f"{flag}{y} valley-mean annual rainfall: {got:7.1f} mm "
                  f"(known {expect} mm, {pct:+.1f}%)")

    # 2. Station coverage
    low = weekly[weekly["n_stations"] < 20]
    if len(low):
        ok = False
        print(f"!! {len(low)} weeks have <20 stations reporting")
        print(low[["year", "week", "n_stations"]].to_string(index=False))
    else:
        print(f"OK all {len(weekly)} weeks have >=20 of 23 stations "
              f"(min {weekly.n_stations.min()})")

    # 3. Monsoon hump
    prof = weekly.groupby("week")["rain_mean_mm"].mean()
    peak = int(prof.idxmax())
    monsoon_share = (weekly[weekly.week.between(22, 39)]["rain_mean_mm"].sum()
                     / weekly["rain_mean_mm"].sum())
    print(f"OK rainfall peaks at epi-week {peak} "
          f"({prof.max():.1f} mm); weeks 22-39 carry "
          f"{100 * monsoon_share:.0f}% of annual rain")
    if not (18 <= peak <= 32):
        ok = False
        print("!! peak week outside expected monsoon window")

    print("--- VALIDATION " + ("PASSED" if ok else "HAS WARNINGS") + " ---\n")
    return ok


def main():
    df = load_wide()
    stations = df[META].rename(columns={"Station_ID": "station_id"})
    stations.to_csv(OUT_STATIONS, index=False)
    print(f"[save] {len(stations)} stations -> {OUT_STATIONS}")

    long = melt_long(df)
    long = filter_recent(long)

    daily_valley = long.groupby("date").agg(
        rain_mean_mm=("rainfall_mm", "mean"),
        rain_max_mm=("rainfall_mm", "max"),
        n_stations=("station_id", "nunique"),
    ).reset_index()
    daily_valley.to_csv(OUT_DAILY, index=False)
    print(f"[save] {len(daily_valley)} valley-daily rows -> {OUT_DAILY}")

    weekly, _ = weekly_features(long)
    validate(long, weekly)

    weekly.to_csv(OUT_WEEKLY, index=False)
    print(f"[save] {len(weekly)} weekly rows -> {OUT_WEEKLY}")
    print(weekly.head().to_string(index=False))
    print(weekly.describe().round(2).to_string())


if __name__ == "__main__":
    main()
