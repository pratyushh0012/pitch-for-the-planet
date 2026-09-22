"""
NASA POWER fetch -- Kathmandu point climate, aggregated onto the EDCD epi-week
calendar (NOT ISO week -- see src/common/epiweek.py for why).

Unlike the original version this keeps the DAILY series too, so the climate
bridge can rebuild the same feature shapes DHM gives us (wet days, heaviest
single day) instead of only weekly totals.

Outputs:
  data/interim/nasa_power_daily.csv
  data/interim/nasa_power_weekly.csv
"""

import os
import sys
import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from src.common.epiweek import assign_epiweek  # noqa: E402

LAT, LON = 27.7172, 85.3240
START, END = "20210101", "20251231"
PARAMS = "PRECTOTCORR,T2M,RH2M"

URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters={PARAMS}&community=AG"
    f"&longitude={LON}&latitude={LAT}"
    f"&start={START}&end={END}&format=JSON"
)

OUT_DAILY = "data/interim/nasa_power_daily.csv"
OUT_WEEKLY = "data/interim/nasa_power_weekly.csv"


def fetch_daily():
    resp = requests.get(URL, timeout=90)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename(columns={"PRECTOTCORR": "rainfall_mm",
                            "T2M": "temp_c",
                            "RH2M": "humidity_pct"})
    # NASA POWER uses -999 as its fill value.
    df = df.mask(df <= -900)
    print(f"[nasa] {len(df)} daily rows "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    print(f"[nasa] missing: {df.isna().sum().to_dict()}")
    return df


def aggregate(df):
    df = assign_epiweek(df, date_index=True)
    weekly = df.groupby(["year", "week"]).agg(
        nasa_rain_sum=("rainfall_mm", "sum"),
        nasa_rain_mean=("rainfall_mm", "mean"),
        nasa_rain_max_daily=("rainfall_mm", "max"),
        nasa_wet_days=("rainfall_mm", lambda s: int((s > 1.0).sum())),
        temp_c_mean=("temp_c", "mean"),
        temp_c_max=("temp_c", "max"),
        humidity_pct_mean=("humidity_pct", "mean"),
        n_days=("rainfall_mm", "size"),
    ).reset_index()
    full = weekly[weekly.n_days == 7].drop(columns=["n_days"])
    if len(full) < len(weekly):
        print(f"[nasa] dropped {len(weekly) - len(full)} partial epi-weeks")
    print(f"[nasa] {len(full)} complete epi-weeks "
          f"{full.year.min()}w{full.week.min()} -> "
          f"{full.year.max()}w{full.week.max()}")
    return df, full


if __name__ == "__main__":
    daily = fetch_daily()
    daily_ew, weekly = aggregate(daily)
    daily_ew.to_csv(OUT_DAILY)
    weekly.to_csv(OUT_WEEKLY, index=False)
    print(f"[save] {OUT_DAILY}, {OUT_WEEKLY}")
    print(weekly.head().to_string(index=False))
