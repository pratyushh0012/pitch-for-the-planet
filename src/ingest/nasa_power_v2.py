#!/usr/bin/env python3
"""
nasa_power_v2.py -- NASA POWER daily weather for DengueWatch.

Same 67 district-HQ points as nasa_power_national.py, but:
  * adds T2M_MIN and T2M_MAX (the Chitwan study found min temp most predictive);
  * starts 2016-09-01 so every case week from 2017 has >= 12 weeks of lead-in;
  * stores the API response UNCHANGED (raw NASA column names, -999 kept), so the
    -999 -> NaN conversion is an explicit, logged step in build_table.py.

Output: data/raw/nasa_power_points_v2/{DISTRICT}.csv (one per point, daily)
"""
import datetime as dt
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.ingest.nasa_power_national import DISTRICTS  # noqa: E402

PARAMS = "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M"
START = "20160901"
END = (dt.date.today() - dt.timedelta(days=10)).strftime("%Y%m%d")  # POWER lags a few days
CACHE = "data/raw/nasa_power_points_v2"


def fetch(name, lat, lon, session):
    path = os.path.join(CACHE, name.replace(" ", "_") + ".csv")
    if os.path.exists(path):
        return "cached"
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters={PARAMS}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={START}&end={END}&format=JSON")
    for attempt in range(5):
        try:
            r = session.get(url, timeout=180)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 4:
                raise
            print(f"   retry {name}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    df = pd.DataFrame(r.json()["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df.insert(0, "lat", lat)
    df.insert(1, "lon", lon)
    df.to_csv(path)
    time.sleep(1)
    return f"{len(df)} days"


def main():
    os.makedirs(CACHE, exist_ok=True)
    s = requests.Session()
    for i, (name, (_, _, lat, lon)) in enumerate(DISTRICTS.items(), 1):
        print(f"[{i:2d}/{len(DISTRICTS)}] {name}: {fetch(name, lat, lon, s)}", flush=True)


if __name__ == "__main__":
    main()
