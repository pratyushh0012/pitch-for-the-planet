#!/usr/bin/env python3
"""
EWARS Weekly Bulletin scraper
=============================
Turns Nepal's EDCD EWARS weekly bulletin PDFs into tidy CSVs for ML.

Outputs two files, written to data/raw/ (run from the project root):
  1) ewars_national_weekly.csv  -- the MAIN modelling table:
        one row per (year, week) with national case totals per disease,
        plus reporting-rate columns (data-quality covariates).
  2) ewars_age_districts_long.csv -- OPTIONAL district exploration:
        one row per (year, week, district) for the top-5 AGE districts.

WHY TWO FILES: the bulletins give COMPLETE national weekly totals, but only
the TOP-5 districts by name each week. So the national table is your clean
modelling target; the district table is a partial, exploratory extra.

USAGE
-----
  # 1. Test the parser on the bundled sample (no internet needed):
  python src/ingest/ewars_scraper.py --self-test

  # 2. Do a real run (needs open internet -- run on YOUR machine, not a sandbox):
  python src/ingest/ewars_scraper.py --years 2020 2021 2022 2023 2024 2025

Dependencies:  pip install pdfplumber requests
"""

import argparse
import csv
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE = "https://www.edcd.gov.np"
# The friendly download URL pattern seen on the site, e.g.
#   https://edcd.gov.np/resources/download/ewars-bulletin51-2025
# It redirects (302) to the real hashed PDF, e.g. /uploads/resource/<hash>.pdf
DOWNLOAD_URL = BASE + "/resources/download/ewars-bulletin{week}-{year}"
INDEX_URL = BASE + "/ewars"           # archive listing page (link harvest)
PDF_DIR = "data/raw/ewars_pdfs"       # where downloaded PDFs are cached
OUT_NATIONAL = "data/raw/ewars_national_weekly.csv"
OUT_DISTRICTS = "data/raw/ewars_age_districts_long.csv"
MAX_WEEK = 53                        # ISO years can have 53 weeks

# The six EWARS priority diseases, keyed by the label prefix used in the PDF
# comparative-analysis table. Value = short column name for the CSV.
DISEASES = {
    "AGE": "Number of AGE Cases Reported",
    "SARI": "Number of SARI Cases Reported",
    "Cholera": "Number of Confirmed Cholera Cases Reported",
    "Dengue": "Number of Confirmed Dengue Cases Reported",
    "KalaAzar": "Number of Confirmed Kala Azar Cases Reported",
    "Malaria": "Number of Confirm Malaria Cases Reported",
}

# ---------------------------------------------------------------------------
# PARSING  (this is the part that matters -- fully tested on real text)
# ---------------------------------------------------------------------------
def _normalize(text):
    """Collapse the PDF's hard line-wraps into clean single-spaced text.
    This is essential: labels and sentences wrap mid-line in the extracted
    text (e.g. 'Confirmed Cholera Cases\\nReported 0 0 0 0')."""
    return " ".join(text.split())


def parse_national_totals(text):
    """Return {disease: {'this','prev','change','last_year'}}.

    EDCD has used THREE bulletin layouts over the years, so we try them in
    order of richness and take the first that hits:

    Era C (2025+, tabular):   'AGE 574 677 103 ■ 885'
        -> prev, this, change, last_year  (a stray +/- marker may sit before
           the last number, so we tolerate non-digits between fields)

    Era A/B (2021-2024, prose): '211 cases of AGE were reported this week.'
        -> gives THIS week only. We also grab last-year from the sibling
           line '208 cases of AGE were reported this week in 2023.' and the
           end-of-report comparison table 'AGE 211 225' for prev-week.

    The AGE (diarrhoeal) count is what the project needs, so we make sure it
    is always captured if it exists in any form.
    """
    norm = _normalize(text)
    out = {}

    # ---- Era C: the rich tabular row ('Number of AGE Cases Reported ...'
    #      in 2025, or just 'AGE ...' with 4 numbers). Tolerate a marker
    #      glyph (■/▪ etc.) or stray non-digits between the change and
    #      last-year columns. ----
    for short, label in DISEASES.items():
        # Try the verbose label first, then the bare disease name.
        # (Kala-azar renders as 'Kala azar' in the 2025 table, so allow a space.)
        bare = r"Kala\s*azar" if short == "KalaAzar" else re.escape(short)
        for lead in (re.escape(label), rf"\b{bare}\b"):
            pat = lead + r"\s+(\d+)\s+(\d+)\s+(-?\d+)\s+\D{0,4}(\d+)"
            m = re.search(pat, norm)
            if m:
                prev, this, change, last = map(int, m.groups())
                out[short] = {"prev": prev, "this": this,
                              "change": change, "last_year": last}
                break

    # ---- Era A/B prose fallback for AGE (and the other diseases) ----
    # Only fill in what the tabular pass didn't already get.
    for short in DISEASES:
        if short in out:
            continue
        this = _prose_this_week(norm, short)
        if this is None:
            continue
        rec = {"this": this, "prev": None, "change": None,
               "last_year": _prose_last_year(norm, short)}
        # end-of-report comparison table, e.g. 'AGE 192 203' -> (this, prev-ish)
        mt = re.search(rf"\b{re.escape(short)}\b\s+(\d+)\s+(\d+)\b", norm)
        if mt:
            # first number matches 'this'; second is the neighbouring week
            a, b = int(mt.group(1)), int(mt.group(2))
            if a == this:
                rec["prev"] = b  # comparison is with the adjacent week
        out[short] = rec

    return out


# Disease name as it appears in prose (may differ from our short key).
_PROSE_NAME = {
    "AGE": r"AGE",
    "SARI": r"SARI",
    "Cholera": r"[Cc]holera",
    "Dengue": r"[Dd]engue",
    "KalaAzar": r"[Kk]ala[- ]?azar",
    "Malaria": r"[Mm]alaria",
}


def _prose_this_week(norm, short):
    """'211 cases of AGE were reported this week' -> 211.
    Also handles the glued extraction '208cases ofAGE'."""
    name = _PROSE_NAME.get(short, re.escape(short))
    # number ... cases ... of <disease> ... reported this week (NOT '... in YYYY')
    m = re.search(
        rf"(\d+)\s*[Cc]ases?\s+of\s*{name}\s+(?:are|were|was)\s+reported\s+this\s+week(?!\s+in\b)",
        norm)
    if m:
        return int(m.group(1))
    return None


def _prose_last_year(norm, short):
    """'226 cases of AGE were reported this week in 2020' -> 226."""
    name = _PROSE_NAME.get(short, re.escape(short))
    m = re.search(
        rf"(\d+)\s*[Cc]ases?\s+of\s*{name}\s+(?:are|were|was)\s+reported\s+this\s+week\s+in\s+\d{{4}}",
        norm)
    if m:
        return int(m.group(1))
    return None


def parse_reporting_rate(text):
    """Return (rate_pct, sites_reporting, sites_total) or (None, None, None).
    Reporting rate is a data-quality covariate: when fewer sites report,
    case counts drop for non-epidemiological reasons."""
    norm = _normalize(text)
    m = re.search(
        r"reporting rate of Week\s+\d+\s+is\s+([\d.]+)%\s+with\s+(\d+)\s+out of\s+(\d+)\s+sentinel sites",
        norm,
    )
    if m:
        return float(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def parse_top_districts(text, disease="AGE"):
    """Return [(district, cases), ...] for the named disease's summary line.
    Handles the two-line wrap and both 'case' and 'cases'."""
    norm = _normalize(text)
    # Lead-in is usually 'Most of the AGE cases were reported from ...'
    # but for rare diseases it can be 'AGE case(s) were reported from ...'.
    m = re.search(
        rf"{re.escape(disease)}\s+cases?\s+(?:were|was)\s+reported from\s+(.+?)\.",
        norm,
        flags=re.IGNORECASE,
    )
    if not m:
        return []
    segment = m.group(1)
    # Extract 'NAME (N case[s])' pairs. District names are ALL CAPS and may
    # contain spaces (e.g. 'NAWALPARASI WEST'). The [A-Z] start skips the
    # connectors ', ' and 'and ' between entries.
    pairs = re.findall(r"([A-Z][A-Za-z ]*?)\s*\((\d+)\s*cases?\)", segment)
    result = []
    for name, n in pairs:
        name = name.strip()
        # defensive: strip a stray leading connector if one slipped in
        name = re.sub(r"^(and|And)\s+", "", name).strip()
        if name:
            result.append((name, int(n)))
    return result


def parse_bulletin(text, week, year):
    """Combine the pieces into one flat record for the national CSV,
    plus the district list."""
    totals = parse_national_totals(text)
    rate, rep, tot = parse_reporting_rate(text)
    districts = parse_top_districts(text, "AGE")

    row = {"year": year, "week": week,
           "reporting_rate_pct": rate,
           "sites_reporting": rep, "sites_total": tot}
    for short in DISEASES:
        t = totals.get(short, {})
        row[f"{short.lower()}_this_week"] = t.get("this")
        row[f"{short.lower()}_prev_week"] = t.get("prev")
        row[f"{short.lower()}_change"] = t.get("change")
        row[f"{short.lower()}_last_year"] = t.get("last_year")
    # keep the AGE districts inline too, as a JSON string, for convenience
    row["age_top_districts"] = json.dumps(districts)
    return row, districts


# ---------------------------------------------------------------------------
# NETWORK  (run on a machine with open internet)
# ---------------------------------------------------------------------------
def download_bulletin(session, week, year):
    """Download one EWARS bulletin by its known URL pattern.
    Returns local path, or None if that week doesn't exist.

    The EDCD download route is stable and guessable:
      https://www.edcd.gov.np/resources/download/ewars-bulletin{week}-{year}
    It 302-redirects to the real (hashed) PDF; requests follows that for us.
    """
    os.makedirs(PDF_DIR, exist_ok=True)
    dest = os.path.join(PDF_DIR, f"ewars_{year}_wk{week:02d}.pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest  # cached from a previous run

    url = DOWNLOAD_URL.format(week=week, year=year)
    try:
        r = session.get(url, timeout=60, allow_redirects=True)
        # A real bulletin comes back as a PDF (by header or magic bytes).
        is_pdf = ("pdf" in r.headers.get("Content-Type", "").lower()
                  or r.content[:4] == b"%PDF")
        if r.status_code == 200 and is_pdf and len(r.content) > 1000:
            with open(dest, "wb") as f:
                f.write(r.content)
            return dest
    except Exception as e:
        print(f"  [wk{week:02d}-{year}: {e}]")
    return None


def extract_text(pdf_path):
    """Extract all text from a PDF with pdfplumber."""
    import pdfplumber
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def run(years):
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})

    # Try every (week, year) via the known download-URL pattern. Weeks that
    # don't exist simply come back empty and are skipped -- no HTML parsing,
    # nothing fragile to break.
    targets = [(w, y) for y in sorted(years) for w in range(1, MAX_WEEK + 1)]
    print(f"Trying {len(targets)} (week, year) combinations "
          f"for years {sorted(years)} ...\n")

    national_rows, district_rows = [], []
    got, missing = 0, 0
    for week, year in targets:
        path = download_bulletin(session, week, year)
        if not path:
            missing += 1
            continue
        try:
            text = extract_text(path)
            row, districts = parse_bulletin(text, week, year)
            if row.get("age_this_week") is not None:
                national_rows.append(row)
                for d, n in districts:
                    district_rows.append({"year": year, "week": week,
                                          "district": d, "age_cases": n})
                got += 1
                print(f"  OK   wk{week:02d}-{year}  AGE={row['age_this_week']}")
            else:
                # Downloaded but no AGE total -> likely the NEW 2026 layout.
                print(f"  SKIP wk{week:02d}-{year}  (AGE total not found; "
                      f"different layout?)")
        except Exception as e:
            print(f"  ERR  wk{week:02d}-{year}: {e}")
        time.sleep(0.4)  # be polite to the government server

    print(f"\nDownloaded & parsed {got} bulletins; "
          f"{missing} week-slots had no file.")
    write_csvs(national_rows, district_rows)


def write_csvs(national_rows, district_rows):
    if national_rows:
        national_rows.sort(key=lambda r: (r["year"], r["week"]))
        cols = (["year", "week", "reporting_rate_pct", "sites_reporting",
                 "sites_total"]
                + [f"{d.lower()}_{s}" for d in DISEASES
                   for s in ("this_week", "prev_week", "change", "last_year")]
                + ["age_top_districts"])
        with open(OUT_NATIONAL, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(national_rows)
        print(f"\nWrote {OUT_NATIONAL} ({len(national_rows)} weeks)")
    if district_rows:
        with open(OUT_DISTRICTS, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["year", "week", "district", "age_cases"])
            w.writeheader()
            w.writerows(district_rows)
        print(f"Wrote {OUT_DISTRICTS} ({len(district_rows)} rows)")


# ---------------------------------------------------------------------------
# SELF-TEST  (proves the parser works on real bulletin text, no internet)
# ---------------------------------------------------------------------------
def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(here, "sample_wk51_2025.txt")
    with open(sample) as f:
        text = f.read()

    print("=== NATIONAL TOTALS ===")
    totals = parse_national_totals(text)
    for d, v in totals.items():
        print(f"  {d:9s} this={v['this']:>4}  prev={v['prev']:>4}  "
              f"change={v['change']:>4}  last_year={v['last_year']:>4}")

    print("\n=== REPORTING RATE ===")
    print("  ", parse_reporting_rate(text))

    print("\n=== TOP AGE DISTRICTS ===")
    for name, n in parse_top_districts(text, "AGE"):
        print(f"  {name:20s} {n}")

    print("\n=== FLAT RECORD (as written to CSV) ===")
    row, districts = parse_bulletin(text, week=51, year=2025)
    for k, v in row.items():
        print(f"  {k}: {v}")

    # sanity assertions
    assert totals["AGE"]["this"] == 287, "AGE this-week mismatch"
    assert totals["Cholera"]["this"] == 0, "Cholera mismatch"
    assert parse_reporting_rate(text)[0] == 95.5, "rate mismatch"
    dd = dict(parse_top_districts(text, "AGE"))
    assert dd["PARSA"] == 20 and dd["KATHMANDU"] == 18 and dd["LALITPUR"] == 13
    assert len(dd) == 5, "expected exactly 5 AGE districts"
    # make sure disease lines don't bleed into each other:
    # NAWALPARASI WEST belongs to Dengue, must NOT appear under AGE
    assert "NAWALPARASI WEST" not in dd, "district bleed between diseases"
    print("\nAll assertions passed. Parser works on real bulletin text.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="Run the parser on the bundled sample (no internet).")
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2022, 2023, 2024, 2025],
                    help="Years to scrape, e.g. --years 2020 2021 2022")
    args = ap.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args.years)
