#!/usr/bin/env python3
"""
audit.py -- Phase 0 data audit + gate for DengueWatch Nepal.

Lists every file, profiles every table, matches tables to the required datasets
(A: weekly dengue, B: NASA POWER weather, C: reporting sites), checks the known
data issues and decides PASS / PASS WITH WARNINGS / FAIL.

Output: reports/data_audit_report.md  (+ short summary on stdout)
Exit code 1 on FAIL so the pipeline stops.
"""
import datetime as dt
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import epi_calendar as cal  # noqa: E402

OUT = os.path.join(ROOT, "reports", "data_audit_report.md")
SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".ipynb_checkpoints",
             ".pytest_cache", ".claude"}
# DengueWatch's own outputs -- not inputs, so not audited
OWN_OUTPUTS = {"reports/figures/dengue", "models/dengue", "app", "tests"}
OWN_FILES = {"data/processed/weekly_table.csv", "data/processed/features.csv",
             "data/processed/cleaning_log.csv", "reports/data_audit_report.md",
             "reports/results.md"}
BULK_DIRS = {"data/raw/ewars_pdfs"}          # summarised as one row (hundreds of PDFs)
TABULAR = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".gz"}

# Published annual totals (all-source national surveillance, NOT EWARS sentinel-only),
# from the sources cited in CLAUDE.md. EWARS sees a subset, so ratio < 1 is expected.
KNOWN_ANNUAL = {2019: 17992, 2022: 54784, 2023: 51243}

DENGUE_PAT = re.compile(r"dengue", re.I)
WEATHER_COLS = {"t2m": "temp", "temp_c": "temp", "temp_c_mean": "temp", "t2m_min": "tmin",
                "t2m_max": "tmax", "prectotcorr": "rain", "rainfall_mm": "rain", "rain_sum": "rain",
                "rh2m": "rh", "humidity_pct": "rh", "humidity_pct_mean": "rh"}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def ftype(path):
    p = path.lower()
    for ext in (".csv.gz", ".jsonl.gz"):
        if p.endswith(ext):
            return ext[1:]
    return os.path.splitext(p)[1][1:] or "other"


def inventory():
    rows = []
    for dirpath, dirnames, files in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and os.path.normpath(os.path.join(rel_dir, d)) not in OWN_OUTPUTS]
        if rel_dir in BULK_DIRS:
            sizes = [os.path.getsize(os.path.join(dirpath, f)) for f in files]
            exts = pd.Series([ftype(f) for f in files]).value_counts().to_dict()
            rows.append({"file": rel_dir + "/ (" + ", ".join(f"{v} {k}" for k, v in exts.items()) + ")",
                         "type": "bulk", "bytes": sum(sizes)})
            dirnames[:] = []
            continue
        for f in files:
            if f == ".DS_Store":
                continue
            rel = os.path.normpath(os.path.join(rel_dir, f))
            if rel in OWN_FILES:
                continue
            rows.append({"file": rel, "type": ftype(f),
                         "bytes": os.path.getsize(os.path.join(dirpath, f))})
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def read_table(path):
    full = os.path.join(ROOT, path)
    t = ftype(path)
    if t in ("csv", "csv.gz"):
        return pd.read_csv(full, low_memory=False)
    if t in ("xlsx", "xls"):
        return pd.read_excel(full)
    if t == "parquet":
        return pd.read_parquet(full)
    if t == "jsonl.gz":
        return None                       # bulletin text archive: not tabular data
    if t == "json":
        try:
            return pd.read_json(full)
        except ValueError:
            return None
    return None


def profile(df):
    cols = {str(c).lower(): c for c in df.columns}
    p = {"rows": len(df), "cols": df.shape[1], "dup_rows": int(df.duplicated().sum())}
    p["missing"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    if "year" in cols and "week" in cols:
        y, w = df[cols["year"]], df[cols["week"]]
        keys = [cols["year"], cols["week"]]
        if "district" in cols:
            keys.append(cols["district"])
        p["dup_keys"] = int(df.duplicated(keys).sum())
        p["range"] = f"{int(y.min())}-W{int(w[y == y.min()].min()):02d} .. {int(y.max())}-W{int(w[y == y.max()].max()):02d}"
    else:
        dcol = next((df.columns[i] for i, c in enumerate(df.columns)
                     if str(c).lower() in ("date", "unnamed: 0", "")), None)
        if dcol is not None:
            d = pd.to_datetime(df[dcol], errors="coerce")
            if d.notna().mean() > 0.9:
                p["range"] = f"{d.min().date()} .. {d.max().date()}"
                p["dup_keys"] = int(d.duplicated().sum()) if "district" not in cols else \
                    int(df.duplicated([dcol, cols["district"]]).sum())
    return p


def classify(df):
    lc = [str(c).lower() for c in df.columns]
    tags = set()
    if any(DENGUE_PAT.search(c) for c in lc) and "year" in lc and "week" in lc:
        tags.add("A")
    wv = {WEATHER_COLS[c] for c in lc if c in WEATHER_COLS}
    if "temp" in wv or "tmin" in wv:
        tags.add("B")
    if any(c in lc for c in ("sites_reporting", "reporting_rate_pct")):
        tags.add("C")
    return tags, wv


def usable_seasons(s):
    """s: DataFrame year, week, value. Usable = >=40/52 weeks with values AND
    <=3 missing weeks inside the peak window 22-48."""
    out = []
    for y, g in s.groupby("year"):
        have = set(g.loc[g.value.notna(), "week"].astype(int))
        n = len(have & set(range(1, 53)))
        peak_missing = len(set(range(22, 49)) - have)
        out.append({"year": int(y), "weeks_with_value": n, "peak_missing": peak_missing,
                    "total": g.value.sum(min_count=1),
                    "usable": n >= 40 and peak_missing <= 3})
    return pd.DataFrame(out)


def suspicious(s):
    """Negative counts; single-week jumps >10x with no neighbouring rise."""
    v = s.set_index(["year", "week"]).value
    flags = [f"{y}-W{w:02d}: negative ({x})" for (y, w), x in v.items() if x < 0]
    arr = v.values
    for i in range(1, len(arr) - 1):
        a, b, c = arr[i - 1], arr[i], arr[i + 1]
        if np.isnan([a, b, c]).any():
            continue
        if b > 10 * max(a, 1) and b > 10 * max(c, 1) and b >= 20:
            y, w = v.index[i]
            flags.append(f"{y}-W{w:02d}: isolated spike {int(a)} -> {int(b)} -> {int(c)}")
    return flags


def md_table(df):
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r) + " |")
    return "\n".join(lines) + "\n"


def main():
    inv = inventory()
    profiles, tags = {}, {}
    for f, t in zip(inv.file, inv.type):
        if t not in ("csv", "csv.gz", "xlsx", "xls", "json", "parquet"):
            continue
        try:
            df = read_table(f)
        except Exception as e:                   # noqa: BLE001
            profiles[f] = {"error": str(e)}
            continue
        if df is None:
            continue
        profiles[f] = profile(df)
        profiles[f]["dtypes"] = ", ".join(f"{c}:{d}" for c, d in df.dtypes.astype(str).items())
        tg, wv = classify(df)
        profiles[f]["weather_vars"] = sorted(wv)
        tags[f] = tg
    inv["matched"] = inv.file.map(lambda f: ",".join(sorted(tags.get(f, []))) or "")
    inv["size"] = inv.bytes.map(human)

    warnings, fails, notes = [], [], []

    # ---------------- Dataset A ----------------
    a_cands = [f for f, t in tags.items() if "A" in t]
    a_stats = {}
    for f in a_cands:
        df = read_table(f)
        col = "dengue_cases" if "dengue_cases" in df else "dengue_this_week"
        s = df[["year", "week", col]].rename(columns={col: "value"})
        a_stats[f] = (usable_seasons(s), s)
    A = None
    if a_stats:
        A = max(a_stats, key=lambda f: (a_stats[f][0].usable.sum(), a_stats[f][1].value.notna().sum()))
        notes.append(f"Dataset A candidates: {', '.join(a_cands)}. Chosen: `{A}` "
                     f"(most usable seasons and most non-missing weeks).")
    else:
        fails.append("No weekly dengue series found.")

    # ---------------- Dataset B ----------------
    b_cands = {f: profiles[f]["weather_vars"] for f, t in tags.items() if "B" in t}
    B = "data/raw/nasa_power_v2_daily.csv.gz" if "data/raw/nasa_power_v2_daily.csv.gz" in b_cands else \
        (max(b_cands, key=lambda f: len(b_cands[f])) if b_cands else None)
    if B is None:
        fails.append("No weather data with a temperature variable.")
    else:
        notes.append(f"Dataset B candidates: {len(b_cands)} files. Chosen: `{B}` -- the only one "
                     "with T2M_MIN/T2M_MAX, all 67 district points, and a start date >= 12 weeks before 2017.")

    # ---------------- calendar + coverage ----------------
    calib_note, anchors, checks = "", None, None
    if A:
        raw = read_table(A)
        if "header_date" in raw:
            anchors, checks = cal.calibrate(raw)
    usable = a_stats[A][0] if A else pd.DataFrame()
    season_years = usable[usable.year.between(2016, 2025)] if len(usable) else usable
    n_usable = int(season_years.usable.sum()) if len(season_years) else 0

    wx = None
    if B:
        wx = read_table(B)
        wx["date"] = pd.to_datetime(wx["date"])
        n999 = {c: int((wx[c] == -999).sum()) for c in ("T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR", "RH2M") if c in wx}
        notes.append(f"NASA POWER `-999` markers in chosen weather file: {n999} (converted to NaN in build_table.py).")
        if A and anchors:
            first_case_year = int(season_years[season_years.usable].year.min()) if n_usable else 2017
            need_from = cal.epiweek_end(first_case_year, 1, anchors) - dt.timedelta(weeks=12)
            if wx.date.min().date() > need_from:
                warnings.append(f"Weather starts {wx.date.min().date()}, after the 12-week lead-in "
                                f"needed for {first_case_year} ({need_from}).")
            last_case = raw.dropna(subset=["dengue_cases"]).iloc[-1]
            if wx.date.max().date() < cal.epiweek_end(int(last_case.year), int(last_case.week), anchors):
                fails.append("Weather does not cover the end of the case period.")
        if not {"T2M", "T2M_MIN", "T2M_MAX"} & set(wx.columns):
            fails.append("No temperature variable in weather data.")
        if wx["district"].nunique() < 2:
            warnings.append("Weather from only one location.")

    if A and n_usable < 4:
        fails.append(f"Only {n_usable} usable seasons (need >= 4: 3 training + 1 test).")
    elif A and n_usable < 6:
        warnings.append(f"Only {n_usable} usable seasons (< 6): few test seasons, wide uncertainty.")

    # ---------------- Dataset C ----------------
    site_years = []
    if A:
        raw = read_table(A)
        if "sites_reporting" in raw:
            cov = raw.groupby("year").sites_reporting.apply(lambda s: s.notna().mean())
            site_years = [int(y) for y, v in cov.items() if v >= 0.5]
        # judged on USABLE seasons: normalising needs a site count for every modelled week
        useful = set(season_years[season_years.usable].year) if len(season_years) else set()
        covered = sorted(useful & set(site_years))
        if len(covered) <= len(useful) / 2:
            warnings.append(f"Reporting-site counts cover only {len(covered)}/{len(useful)} usable seasons "
                            f"({covered}; none for 2020-2023, the big outbreak years): rate = raw cases, "
                            "no per-site normalisation.")
        # structural change in the network itself
        tot = raw.groupby("year").sites_total.max().dropna()
        warnings.append("Sentinel network grew from ~49 sites (2016) to 118 (May 2019+) and 134 (2025); "
                        "pre-2019 case levels are not comparable (2016-2018 annual totals < 200).")
        if len(tot):
            notes.append("Sentinel network size (max functional sites stated per year): "
                         + ", ".join(f"{int(y)}: {int(v)}" for y, v in tot.items())
                         + ". 2020-2023 bulletins state 'current number of sentinel sites is 118'. "
                         "The network roughly doubled after May 2019 -- pre-2019 counts are not "
                         "comparable in level with later years.")

    # ---------------- suspicious ----------------
    susp = suspicious(a_stats[A][1].sort_values(["year", "week"])) if A else []
    if A:
        tot = a_stats[A][0].set_index("year").total
        for y, known in KNOWN_ANNUAL.items():
            if y in tot.index and pd.notna(tot[y]):
                r = tot[y] / known
                tag = "OK" if 0.25 <= r <= 1.5 else "DISAGREES"
                susp.append(f"{y}: EWARS total {int(tot[y]):,} vs published national {known:,} "
                            f"(ratio {r:.2f}) -- {tag}")
                if tag == "DISAGREES":
                    warnings.append(f"{y} EWARS total disagrees strongly with known national total.")
    if any("negative" in s or "spike" in s for s in susp):
        warnings.append("Suspicious values found (see Dataset A section).")

    # ---------------- known issues ----------------
    known = []
    old = "data/raw/ewars_national_weekly.csv"
    if old in a_stats:
        o = a_stats[old][0].set_index("year")
        known.append(f"**2021/2022 parser gap (old CSV):** still present in `{old}` -- 2021: "
                     f"{o.weeks_with_value.get(2021)} weeks, 2022: {o.weeks_with_value.get(2022)} weeks. "
                     "Cause: the MonsoonWatch regex needed digits + exact phrasing and missed number-words "
                     "('Eight case of dengue was reported'), glued text ('2857cases ofDengue') and the "
                     "comparison table. Fix: `src/ingest/ewars_dengue_parse.py` (old file untouched; new "
                     "output `data/raw/ewars_dengue_weekly_v2.csv`). The original URL guesser also only "
                     "matched late-2025 slugs; `src/ingest/ewars_harvest.py` reads the real links from the "
                     "45 index pages instead (+ slug guesses for weeks missing from the index).")
    if A:
        t = a_stats[A][0].set_index("year").total
        if 2024 in t and 2025 in t:
            known.append(f"**2025 low season:** {int(t[2025]):,} cases vs {int(t[2024]):,} in 2024. "
                         "Bulletins are complete for the season, so this is treated as real "
                         "(immunity / serotype shift). Kept, not dropped.")
    known.append("**Pre-computed EWARS columns** (`prev_week`, `change`, `last_year`): inconsistent across "
                 "eras. Not used as features; weekly values recomputed from raw this-week counts. "
                 "`prev_reported` is used ONLY to recover the value of a week whose own bulletin is "
                 "missing (logged in Phase 1), after checking it agrees with the direct value elsewhere.")
    known.append("**`reporting_rate_pct` / district columns:** only 2024-2025 in the old CSV; never used as features.")
    known.append("**EDCD URL:** all requests use `https://www.edcd.gov.np` (with `www.`).")
    if checks is not None:
        bad = checks[(checks.source == "bulletin headers") & ~checks.rule_matches]
        known.append("**Week alignment:** EDCD weeks are Sunday-ending, numbered from the week ending on the "
                     "first Sunday on/after 11 Jan. Calibrated per year from bulletin header dates (mode of "
                     "header_date - 7*(week-1)). " + ("Rule matches every calibrated year. " if bad.empty else
                     f"Rule FAILS for {bad.year.tolist()} -- header anchors used. ")
                     + "2020 has 53 weeks (its week 53 ended 10 Jan 2021). `src/common/epiweek.py` assumes 52 "
                     "weeks every year, so it is one week off for 2016-2020; DengueWatch uses "
                     "`src/epi_calendar.py` instead.")

    # spot-check 3 bulletins
    spot = pd.DataFrame()
    if A and anchors:
        raw = read_table(A).dropna(subset=["header_date"])
        picks = raw[(raw.year.isin([2017, 2020, 2023])) & (raw.week.isin([30, 53, 38]))]
        picks = picks.groupby("year").head(1)
        if len(picks) < 3:
            picks = raw.sample(3, random_state=42)
        spot = pd.DataFrame([{
            "bulletin": r.source_file, "header date": r.header_date,
            "mapped week": "{}-W{:02d}".format(*cal.date_to_epiweek(dt.date.fromisoformat(r.header_date), anchors)),
            "week range (Mon-Sun)": f"{cal.epiweek_end(r.year, r.week, anchors) - dt.timedelta(days=6)} .. "
                                    f"{cal.epiweek_end(r.year, r.week, anchors)}",
        } for r in picks.itertuples()])

    status = "FAIL" if fails else ("PASS WITH WARNINGS" if warnings else "PASS")

    # ---------------- write ----------------
    L = [f"# Data audit report — DengueWatch Nepal\n",
         f"_Generated by `src/audit.py` on {dt.date.today()}._\n",
         f"## 1. Gate status: **{status}**\n"]
    if fails:
        L.append("**Failures**\n" + "\n".join(f"- {x}" for x in fails) + "\n")
    if warnings:
        L.append("**Warnings**\n" + "\n".join(f"- {x}" for x in warnings) + "\n")
    L.append("**Decisions**\n" + "\n".join(f"- {x}" for x in notes) + "\n")

    L.append("## 2. File inventory\n")
    L.append(md_table(inv[["file", "type", "size", "matched"]]))
    L.append("\n`matched`: A = weekly dengue, B = weather, C = reporting sites. "
             "Bulletin PDFs/text cache summarised as one row.\n")

    L.append("## 3. Datasets\n")
    if A:
        p = profiles[A]
        L.append(f"### A. Weekly dengue cases — `{A}`\n")
        L.append(f"- Rows: {p['rows']}, columns: {p['cols']}, range: {p.get('range')}\n"
                 f"- Duplicate rows: {p['dup_rows']}, duplicate (year, week): {p.get('dup_keys')}\n"
                 f"- dtypes: {p['dtypes']}\n"
                 f"- Missing per column: {p['missing']}\n")
        meth = read_table(A).method.value_counts(dropna=False).to_dict()
        L.append(f"- Extraction method per bulletin: {meth}\n")
        L.append("\n**Usable seasons** (≥ 40/52 weeks with a value AND ≤ 3 missing in weeks 22–48):\n\n")
        L.append(md_table(a_stats[A][0].assign(total=lambda d: d.total.map(lambda x: f"{x:,.0f}"))))
        L.append(f"\nUsable seasons 2016–2025: **{n_usable}**.\n")
        L.append("\n**Suspicious values / sanity checks**\n" + ("\n".join(f"- {s}" for s in susp) or "- none") + "\n")
        for f in a_cands:
            if f != A:
                L.append(f"\n_Other A candidate `{f}`:_\n\n" + md_table(a_stats[f][0]))
    if B:
        p = profiles[B]
        L.append(f"\n### B. Weather (NASA POWER) — `{B}`\n")
        L.append(f"- Rows: {p['rows']}, columns: {p['cols']}, range: {p.get('range')}, "
                 f"districts: {wx.district.nunique()}\n"
                 f"- Duplicate rows: {p['dup_rows']}, duplicate (date, district): {p.get('dup_keys')}\n"
                 f"- dtypes: {p['dtypes']}\n- Missing per column: {p['missing'] or 'none'}\n")
        L.append("\nOther weather files (MonsoonWatch, not used: no T2M_MIN/MAX, start 2021): "
                 + ", ".join(f"`{f}`" for f in b_cands if f != B and "nasa_power_points/" not in f)
                 + f", plus {sum('nasa_power_points/' in f for f in b_cands)} per-district files in "
                 "`data/raw/nasa_power_points/`.\n")
    L.append("\n### C. Reporting sites\n")
    L.append(f"- Years with site counts for ≥ 50% of weeks: {site_years}\n")
    L.append("\n### D/E. District sitreps (PDF), CHIRPS\n- Not present. District level is out of scope; "
             "CHIRPS not needed (NASA POWER precipitation used).\n")

    L.append("\n## 4. Known data issues — checked\n" + "\n".join(f"- {k}" for k in known) + "\n")
    if checks is not None:
        L.append("\n**Calendar calibration**\n\n" + md_table(checks))
    if len(spot):
        L.append("\n**Spot-check of 3 bulletins** (header date → mapped epi-week)\n\n" + md_table(spot))

    L.append("\n## 5. Per-table profiles\n")
    for f, p in profiles.items():
        if "error" in p:
            L.append(f"- `{f}`: could not read ({p['error']})\n")
            continue
        miss = sum(p["missing"].values())
        L.append(f"- `{f}`: {p['rows']} rows × {p['cols']} cols; range {p.get('range', 'n/a')}; "
                 f"dup rows {p['dup_rows']}; dup keys {p.get('dup_keys', 'n/a')}; missing cells {miss}\n")
    if fails:
        L.append("\n## 6. What is missing and how to get it\n" + "\n".join(f"- {x}" for x in fails) + "\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L))
    print(f"GATE: {status}")
    for x in fails:
        print("  FAIL:", x)
    for x in warnings:
        print("  WARN:", x)
    if A:
        print(a_stats[A][0].to_string(index=False))
    print(f"-> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
