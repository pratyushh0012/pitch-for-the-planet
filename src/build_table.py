#!/usr/bin/env python3
"""
build_table.py -- Phase 1: one row per EDCD epi-week, cases + weather (+ sites).

Every step that drops, fills or changes a value is appended to the cleaning log
(data/processed/cleaning_log.csv and reports/cleaning_log.md).

Outputs:
  data/processed/weekly_table.csv
  data/processed/cleaning_log.csv, reports/cleaning_log.md
"""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import epi_calendar as cal  # noqa: E402

CASES = os.path.join(ROOT, "data/raw/ewars_dengue_weekly_v2.csv")
WEATHER = os.path.join(ROOT, "data/raw/nasa_power_v2_daily.csv.gz")
OUT = os.path.join(ROOT, "data/processed/weekly_table.csv")
LOG_CSV = os.path.join(ROOT, "data/processed/cleaning_log.csv")
LOG_MD = os.path.join(ROOT, "reports/cleaning_log.md")
FIRST_YEAR, LAST_YEAR = 2016, 2025
HOTSPOTS = ["KATHMANDU", "LALITPUR", "BHAKTAPUR", "SUNSARI", "MORANG", "JHAPA",
            "CHITAWAN", "KASKI", "MAKWANPUR", "DHADING"]
AGREE_MIN = 0.90          # a recovery source is used only if it matches direct values this often

LOG = []


def log(step, year, week, column, old, new, reason):
    LOG.append({"step": step, "year": year, "week": week, "column": column,
                "old": old, "new": new, "reason": reason})


def load_cases():
    raw = pd.read_csv(CASES)
    anchors, _ = cal.calibrate(raw)
    raw = raw[raw.year.between(FIRST_YEAR, LAST_YEAR)].copy()

    # duplicates: keep the most complete row
    raw["_filled"] = raw.notna().sum(axis=1)
    raw = raw.sort_values(["year", "week", "_filled"], ascending=[True, True, False])
    for (y, w), g in raw[raw.duplicated(["year", "week"], keep=False)].groupby(["year", "week"]):
        for r in g.iloc[1:].itertuples():
            log("dedupe", y, w, "row", r.source_file, g.iloc[0].source_file,
                "duplicate (year, week): kept the most complete row")
    raw = raw.drop_duplicates(["year", "week"]).drop(columns="_filled")

    idx = cal.all_weeks(anchors, FIRST_YEAR, LAST_YEAR)
    df = idx.merge(raw, on=["year", "week"], how="left")
    for r in df[df.source_file.isna()].itertuples():
        log("missing_bulletin", r.year, r.week, "dengue_cases", None, None,
            "no bulletin found on EDCD site (index + slug guesses)")
    df["cases_source"] = np.where(df.dengue_cases.notna(), "bulletin:" + df.method.astype(str), None)
    return df, anchors


def recover(df):
    """Fill weeks with no direct value from other bulletins that printed it:
       (1) next week's bulletin 'previous week' column;
       (2) next year's bulletin 'same week last year' value.
    Each source is used only if it agrees with direct values >= AGREE_MIN of the time."""
    d = df.reset_index(drop=True)
    nxt_prev = d.prev_reported.shift(-1)                        # printed in week t+1's bulletin
    nxt_year = d.set_index(["year", "week"]).last_year_rep
    ly = pd.Series([nxt_year.get((y + 1, w), np.nan) for y, w in zip(d.year, d.week)])
    report = {}
    for name, src in [("next_bulletin_prev_week", nxt_prev), ("next_year_same_week", ly)]:
        both = d.dengue_cases.notna() & src.notna()
        agree = float((d.dengue_cases[both] == src[both]).mean()) if both.any() else np.nan
        close = float((abs(d.dengue_cases[both] - src[both]) <= np.maximum(2, 0.1 * d.dengue_cases[both])).mean()) \
            if both.any() else np.nan
        report[name] = {"overlap": int(both.sum()), "exact_agree": agree, "within_10pct": close}
        if both.sum() >= 20 and close >= AGREE_MIN:
            fill = d.dengue_cases.isna() & src.notna()
            for i in d.index[fill]:
                log("recover", d.year[i], d.week[i], "dengue_cases", None, src[i],
                    f"value printed in another bulletin ({name}); source agrees with direct "
                    f"values {close:.0%} of {both.sum()} overlapping weeks (within 10%)")
            d.loc[fill, "dengue_cases"] = src[fill]
            d.loc[fill, "cases_source"] = "recovered:" + name
        else:
            report[name]["used"] = False
    return d, report


def interpolate_single_gaps(d):
    s = d.dengue_cases
    gap = s.isna()
    single = gap & s.shift(1).notna() & s.shift(-1).notna()
    for i in d.index[single]:
        v = round((s[i - 1] + s[i + 1]) / 2)
        log("interpolate", d.year[i], d.week[i], "dengue_cases", None, v,
            "single-week gap: mean of neighbours")
        d.loc[i, "dengue_cases"] = v
        d.loc[i, "cases_source"] = "interpolated"
    for i in d.index[d.dengue_cases.isna()]:
        log("leave_nan", d.year[i], d.week[i], "dengue_cases", None, None,
            "gap longer than 1 week: left as NaN (no interpolation)")
    return d


def load_weather(anchors):
    w = pd.read_csv(WEATHER, parse_dates=["date"])
    vars_ = ["T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR", "RH2M"]
    for c in vars_:
        n = int((w[c] <= -900).sum())
        if n:
            log("nasa_fill", None, None, c, -999, "NaN", f"{n} NASA POWER fill values -> NaN")
        w[c] = w[c].mask(w[c] <= -900)
    yw = [cal.date_to_epiweek(d, anchors) for d in w.date.dt.date.unique()]
    m = pd.DataFrame(yw, columns=["year", "week"])
    m["date"] = pd.to_datetime(w.date.dt.date.unique())
    w = w.merge(m, on="date")
    wk = w.groupby(["district", "year", "week"]).agg(
        tmean=("T2M", "mean"), tmin=("T2M_MIN", "mean"), tmax=("T2M_MAX", "mean"),
        rain=("PRECTOTCORR", "sum"), rh=("RH2M", "mean"), n_days=("T2M", "count")).reset_index()
    partial = wk[wk.n_days < 7]
    if len(partial):
        log("weather_partial_week", None, None, "all", len(partial), "dropped",
            f"{len(partial)} district-weeks with < 7 valid days dropped (edges of the download window)")
    wk = wk[wk.n_days == 7].drop(columns="n_days")
    cols = ["tmean", "tmin", "tmax", "rain", "rh"]
    nat = wk.groupby(["year", "week"])[cols].mean()
    hot = wk[wk.district.isin(HOTSPOTS)].groupby(["year", "week"])[cols].mean().add_suffix("_hot")
    missing_hot = sorted(set(HOTSPOTS) - set(wk.district))
    if missing_hot:
        log("hotspots", None, None, "district", ",".join(missing_hot), None, "hotspot districts not in data")
    return nat.join(hot).reset_index(), wk.district.nunique()


def main():
    df, anchors = load_cases()
    df, rec_report = recover(df)
    df = interpolate_single_gaps(df)

    wx, n_dist = load_weather(anchors)
    table = df[["year", "week", "week_end", "dengue_cases", "cases_source",
                "sites_reporting", "sites_total"]].rename(columns={"dengue_cases": "cases"})
    table = table.merge(wx, on=["year", "week"], how="left")
    # weather starts 2016-09-01 (enough lead-in for 2017+); earlier 2016 weeks keep NaN weather
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    table.to_csv(OUT, index=False)
    log_df = pd.DataFrame(LOG)
    log_df.to_csv(LOG_CSV, index=False)

    # ---- coverage summary ----
    body = table[table.year >= FIRST_YEAR]
    cov = body.groupby("year").agg(weeks=("week", "size"), cases_present=("cases", "count"),
                                   cases_total=("cases", "sum"),
                                   recovered=("cases_source", lambda s: s.astype(str).str.startswith("recovered").sum()),
                                   interpolated=("cases_source", lambda s: (s == "interpolated").sum()),
                                   weather_present=("tmean", "count"))
    print(cov.to_string())
    print("missing per column:", body.isna().sum()[lambda s: s > 0].to_dict())
    print("recovery sources:", rec_report)

    counts = log_df.step.value_counts().to_dict() if len(log_df) else {}
    md = ["# Cleaning log — Phase 1 (`src/build_table.py`)\n",
          f"_Generated {dt.date.today()}._ Full row-level log: `data/processed/cleaning_log.csv`.\n",
          "## Summary\n", f"- Steps: {counts}",
          f"- Weather: {n_dist} district points, daily → EDCD week (tmean/tmin/tmax/rh = mean of daily, "
          "rain = sum); national = mean of all districts; `_hot` = mean of hotspot districts "
          f"({', '.join(HOTSPOTS)}).",
          "- Recovery sources (value of a week whose own bulletin is missing, read from another bulletin):"]
    for k, v in rec_report.items():
        md.append(f"  - `{k}`: overlap {v['overlap']} weeks, exact agreement {v['exact_agree']:.0%}, "
                  f"within 10% {v['within_10pct']:.0%} → {'used' if v.get('used', True) else 'NOT used'}")
    md += ["\n## Coverage per year\n", cov.to_markdown() if hasattr(cov, "to_markdown") else cov.to_string(),
           "\n## Every value that was dropped, filled or changed\n"]
    show = log_df[log_df.step.isin(["dedupe", "recover", "interpolate", "leave_nan", "missing_bulletin",
                                     "nasa_fill", "weather_partial_week", "hotspots"])]
    md.append("| step | year | week | column | old | new | reason |\n|---|---|---|---|---|---|---|")
    for r in show.itertuples():
        md.append(f"| {r.step} | {'' if pd.isna(r.year) else int(r.year)} | "
                  f"{'' if pd.isna(r.week) else int(r.week)} | {r.column} | "
                  f"{'' if pd.isna(r.old) else r.old} | {'' if pd.isna(r.new) else r.new} | {r.reason} |")
    open(LOG_MD, "w").write("\n".join(md) + "\n")
    print(f"-> {OUT} ({len(table)} rows), {LOG_MD}")


if __name__ == "__main__":
    main()
