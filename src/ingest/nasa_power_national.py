"""
NASA POWER fetch -- NATIONAL coverage, one point per district headquarters.

The original fetch (nasa_power_fetch.py) takes a single point in Kathmandu,
which is the same place the DHM valley gauges measure. The disease series is
national, so this pulls the same NASA POWER variables for the headquarters
town of ~65 districts across all 7 provinces, aggregated onto the EDCD
epi-week calendar.

Coordinates are approximate district-HQ locations (to ~0.05 deg). NASA POWER
precipitation is on a ~0.5 x 0.625 deg grid, so small coordinate error does not
change which grid cell a point falls in. Several nearby HQs (e.g. the three
Kathmandu Valley districts) share a cell; indices built from this file must
de-duplicate cells before averaging -- see src/features/national_rain.py.

Raw per-point responses are cached in data/raw/nasa_power_points/ so reruns
do not hit the API again.

Output: data/interim/nasa_power_districts_weekly.csv
        (one row per district x epi-week)
"""

import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from src.common.epiweek import assign_epiweek  # noqa: E402

START, END = "20210101", "20251231"
PARAMS = "PRECTOTCORR,T2M,RH2M"
CACHE = "data/raw/nasa_power_points"
OUT = "data/interim/nasa_power_districts_weekly.csv"

# district: (province, is_terai, lat, lon)  -- HQ town in the comment
DISTRICTS = {
    # Koshi
    "JHAPA":          ("Koshi", 1, 26.544, 88.094),          # Bhadrapur
    "MORANG":         ("Koshi", 1, 26.455, 87.270),          # Biratnagar
    "SUNSARI":        ("Koshi", 1, 26.607, 87.148),          # Inaruwa
    "ILAM":           ("Koshi", 0, 26.909, 87.927),
    "TAPLEJUNG":      ("Koshi", 0, 27.351, 87.670),
    "SANKHUWASABHA":  ("Koshi", 0, 27.374, 87.203),          # Khandbari
    "DHANKUTA":       ("Koshi", 0, 26.983, 87.343),
    "OKHALDHUNGA":    ("Koshi", 0, 27.316, 86.504),
    "UDAYAPUR":       ("Koshi", 0, 26.790, 86.700),          # Gaighat
    "SOLUKHUMBU":     ("Koshi", 0, 27.500, 86.580),          # Salleri
    # Madhesh
    "SAPTARI":        ("Madhesh", 1, 26.540, 86.745),        # Rajbiraj
    "SIRAHA":         ("Madhesh", 1, 26.654, 86.208),
    "DHANUSHA":       ("Madhesh", 1, 26.729, 85.925),        # Janakpur
    "MAHOTTARI":      ("Madhesh", 1, 26.649, 85.800),        # Jaleshwar
    "SARLAHI":        ("Madhesh", 1, 26.857, 85.558),        # Malangwa
    "RAUTAHAT":       ("Madhesh", 1, 26.766, 85.278),        # Gaur
    "BARA":           ("Madhesh", 1, 27.031, 85.001),        # Kalaiya
    "PARSA":          ("Madhesh", 1, 27.012, 84.877),        # Birgunj
    # Bagmati
    "KATHMANDU":      ("Bagmati", 0, 27.717, 85.324),
    "LALITPUR":       ("Bagmati", 0, 27.667, 85.317),
    "BHAKTAPUR":      ("Bagmati", 0, 27.672, 85.428),
    "KAVREPALANCHOK": ("Bagmati", 0, 27.622, 85.543),        # Dhulikhel
    "MAKWANPUR":      ("Bagmati", 0, 27.428, 85.032),        # Hetauda
    "CHITAWAN":       ("Bagmati", 1, 27.683, 84.435),        # Bharatpur
    "DHADING":        ("Bagmati", 0, 27.866, 84.906),
    "SINDHULI":       ("Bagmati", 0, 27.207, 85.917),
    "NUWAKOT":        ("Bagmati", 0, 27.918, 85.144),        # Bidur
    "SINDHUPALCHOK":  ("Bagmati", 0, 27.777, 85.713),        # Chautara
    "DOLAKHA":        ("Bagmati", 0, 27.667, 86.050),        # Charikot
    # Gandaki
    "KASKI":          ("Gandaki", 0, 28.209, 83.985),        # Pokhara
    "TANAHU":         ("Gandaki", 0, 27.978, 84.275),        # Damauli
    "MYAGDI":         ("Gandaki", 0, 28.346, 83.566),        # Beni
    "GORKHA":         ("Gandaki", 0, 28.000, 84.628),
    "SYANGJA":        ("Gandaki", 0, 28.096, 83.874),
    "BAGLUNG":        ("Gandaki", 0, 28.272, 83.590),
    "LAMJUNG":        ("Gandaki", 0, 28.231, 84.379),        # Besisahar
    "NAWALPUR":       ("Gandaki", 1, 27.640, 84.120),        # Kawasoti
    # Lumbini
    "RUPANDEHI":      ("Lumbini", 1, 27.505, 83.452),        # Siddharthanagar
    "KAPILBASTU":     ("Lumbini", 1, 27.541, 83.055),        # Taulihawa
    "PARASI":         ("Lumbini", 1, 27.530, 83.670),        # Ramgram
    "PALPA":          ("Lumbini", 0, 27.867, 83.548),        # Tansen
    "ARGHAKHANCHI":   ("Lumbini", 0, 27.975, 83.126),        # Sandhikharka
    "GULMI":          ("Lumbini", 0, 28.069, 83.250),        # Tamghas
    "DANG":           ("Lumbini", 1, 28.036, 82.487),        # Ghorahi
    "BANKE":          ("Lumbini", 1, 28.050, 81.617),        # Nepalgunj
    "BARDIYA":        ("Lumbini", 1, 28.229, 81.343),        # Gulariya
    "PYUTHAN":        ("Lumbini", 0, 28.100, 82.860),
    "ROLPA":          ("Lumbini", 0, 28.300, 82.630),        # Liwang
    # Karnali
    "SURKHET":        ("Karnali", 0, 28.601, 81.633),        # Birendranagar
    "DAILEKH":        ("Karnali", 0, 28.844, 81.710),
    "SALYAN":         ("Karnali", 0, 28.375, 82.160),
    "RUKUM WEST":     ("Karnali", 0, 28.630, 82.480),        # Musikot
    "JUMLA":          ("Karnali", 0, 29.274, 82.183),
    "JAJARKOT":       ("Karnali", 0, 28.700, 82.190),
    "KALIKOT":        ("Karnali", 0, 29.140, 81.620),        # Manma
    "HUMLA":          ("Karnali", 0, 29.970, 81.830),        # Simikot
    "DOLPA":          ("Karnali", 0, 28.990, 82.820),
    "MUGU":           ("Karnali", 0, 29.550, 82.150),        # Gamgadhi
    # Sudurpashchim
    "KAILALI":        ("Sudurpashchim", 1, 28.698, 80.594),  # Dhangadhi
    "KANCHANPUR":     ("Sudurpashchim", 1, 28.963, 80.178),  # Bhimdatta
    "DOTI":           ("Sudurpashchim", 0, 29.261, 80.940),  # Dipayal
    "DADELDHURA":     ("Sudurpashchim", 0, 29.300, 80.583),
    "BAJURA":         ("Sudurpashchim", 0, 29.448, 81.470),  # Martadi
    "DARCHULA":       ("Sudurpashchim", 0, 29.846, 80.545),
    "BAITADI":        ("Sudurpashchim", 0, 29.520, 80.470),
    "ACHHAM":         ("Sudurpashchim", 0, 29.140, 81.280),  # Mangalsen
    "BAJHANG":        ("Sudurpashchim", 0, 29.550, 81.200),  # Chainpur
}


def fetch_point(name, lat, lon, session):
    path = os.path.join(CACHE, name.replace(" ", "_") + ".csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters={PARAMS}&community=AG"
           f"&longitude={lon}&latitude={lat}"
           f"&start={START}&end={END}&format=JSON")
    for attempt in range(4):
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 3:
                raise
            print(f"   retry {name}: {e}")
            time.sleep(5 * (attempt + 1))
    df = pd.DataFrame(r.json()["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename(columns={"PRECTOTCORR": "rainfall_mm", "T2M": "temp_c",
                            "RH2M": "humidity_pct"})
    df = df.mask(df <= -900)          # NASA POWER fill value
    df.to_csv(path)
    time.sleep(0.5)
    return df


def main():
    os.makedirs(CACHE, exist_ok=True)
    session = requests.Session()
    frames = []
    for i, (name, (prov, terai, lat, lon)) in enumerate(DISTRICTS.items(), 1):
        df = fetch_point(name, lat, lon, session)
        n_missing = int(df["rainfall_mm"].isna().sum())
        print(f"[{i:2d}/{len(DISTRICTS)}] {name:15s} {prov:14s} "
              f"{len(df)} days, {n_missing} missing rain")
        d = assign_epiweek(df, date_index=True)
        w = d.groupby(["year", "week"]).agg(
            rain_sum=("rainfall_mm", "sum"),
            rain_max_daily=("rainfall_mm", "max"),
            wet_days=("rainfall_mm", lambda s: int((s > 1.0).sum())),
            temp_c_mean=("temp_c", "mean"),
            humidity_pct_mean=("humidity_pct", "mean"),
            n_days=("rainfall_mm", "size"),
            rain_signature=("rainfall_mm", lambda s: round(float(s.sum()), 3)),
        ).reset_index()
        w = w[w.n_days == 7].drop(columns=["n_days"])
        w.insert(0, "district", name)
        w.insert(1, "province", prov)
        w.insert(2, "is_terai", terai)
        w.insert(3, "lat", lat)
        w.insert(4, "lon", lon)
        # Identify the NASA grid cell by the whole daily rain series: points in
        # the same cell return identical data.
        w["cell_id"] = pd.util.hash_pandas_object(
            df["rainfall_mm"].round(3), index=False).sum()
        frames.append(w.drop(columns=["rain_signature"]))

    out = pd.concat(frames, ignore_index=True)
    out["cell_id"] = out.groupby("cell_id").ngroup()
    out.to_csv(OUT, index=False)
    n_cells = out.cell_id.nunique()
    print(f"\n[save] {len(out)} rows, {out.district.nunique()} districts, "
          f"{n_cells} distinct NASA grid cells -> {OUT}")
    shared = (out.drop_duplicates("district").groupby("cell_id")["district"]
                 .apply(list))
    for cid, names in shared[shared.apply(len) > 1].items():
        print(f"   shared cell {cid}: {', '.join(names)}")


if __name__ == "__main__":
    main()
