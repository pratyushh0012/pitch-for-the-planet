#!/usr/bin/env python3
"""
ewars_harvest.py -- download every EWARS weekly bulletin listed on the EDCD site.

Why this exists: the original scraper guessed URLs as
`ewars-bulletin{week}-{year}`, but that slug only exists for late 2025. Older
bulletins use slugs like `ewars-weekly-bulletin-12th-week-2019` and some have no
year at all. So instead of guessing, we read the 45 index pages and key each
bulletin by the (year, week) in its TITLE, falling back to the published date.

Outputs (run from project root):
  data/raw/ewars_pdfs/index.csv               title, published, url, year, week
  data/raw/ewars_pdfs/ewars_{year}_wk{NN}.pdf cached PDFs
  data/raw/ewars_pdfs/ewars_{year}_wk{NN}.txt pdfplumber text (cache)
"""
import csv
import html
import os
import re
import sys
import time

import requests

BASE = "https://www.edcd.gov.np"        # 'www.' is required
PDF_DIR = os.environ.get("EWARS_PDF_DIR", "data/raw/ewars_pdfs")  # override when the project dir is iCloud-evicted
INDEX_CSV = os.path.join(PDF_DIR, "index.csv")
UA = {"User-Agent": "Mozilla/5.0 (research scraper; DengueWatch Nepal)"}

BLOCK = re.compile(
    r"<h2>(?P<title>[^<]+)</h2>\s*<p>Published:\s*(?P<pub>[\d-]+)</p>.*?"
    r'href="(?P<url>https://www\.edcd\.gov\.np/resources/download/[^"]+)"',
    re.S)


def parse_title(title, pub):
    """'EWARS Weekly Bulletin 12th week 2019' -> (2019, 12).
    Year missing from title -> infer from published date (a week-52 bulletin
    published in January belongs to the previous year)."""
    t = html.unescape(title)
    m = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)*\s*(?:week|wk)", t, re.I) \
        or re.search(r"bulletin\s*[-_#]?\s*(\d{1,2})\b", t, re.I)
    week = int(m.group(1)) if m else None
    y = re.search(r"(20\d{2})", t)
    pub_year, pub_month = int(pub[:4]), int(pub[5:7])
    if y:
        year = int(y.group(1))
    else:
        year = pub_year - 1 if (week and week >= 40 and pub_month <= 3) else pub_year
    return year, week


def harvest(session, pages=60):
    rows, seen = [], set()
    for p in range(1, pages + 1):
        r = session.get(f"{BASE}/ewars?page={p}", timeout=60)
        hits = BLOCK.findall(r.text)
        if not hits:
            break
        for title, pub, url in hits:
            if url in seen:
                continue
            seen.add(url)
            year, week = parse_title(title, pub)
            rows.append({"title": html.unescape(title).strip(), "published": pub,
                         "url": url, "year": year, "week": week})
        time.sleep(0.3)
    return fix_duplicates(rows)


def fix_duplicates(rows):
    """Two titles claiming the same (year, week) -- e.g. '32nd Week 2023'
    (pub 08-20) and '32rd Week 2023' (pub 08-27, a typo for 33rd). If the
    later-published one is exactly 7 days later and week+1 is unclaimed,
    reassign it to week+1; otherwise keep the later one (assume re-issue)."""
    import datetime as dt
    rows.sort(key=lambda r: r["published"])
    claimed = {}
    for r in rows:
        key = (r["year"], r["week"])
        if r["week"] and key in claimed:
            prev = claimed[key]
            gap = (dt.date.fromisoformat(r["published"])
                   - dt.date.fromisoformat(prev["published"])).days
            nxt = (r["year"], r["week"] + 1)
            if 5 <= gap <= 9 and nxt not in claimed:
                r["note"] = f"title says wk{r['week']}; reassigned to wk{r['week'] + 1} (published {gap}d after the other wk{r['week']})"
                r["week"] += 1
                key = nxt
            else:
                prev["note"] = "superseded by later bulletin with same (year, week)"
                prev["week_superseded"] = True
        claimed[key] = r
    return rows


def guess_missing(session, rows, years):
    """The index omits most of 2024 and early 2025. Try the slug patterns seen
    on the site for every (year, week) the index doesn't have."""
    have = {(r["year"], r["week"]) for r in rows if r["week"]}
    extra = []
    for y in years:
        for w in range(1, 53):
            if (y, w) in have:
                continue
            for slug in (f"ewars-bulletin{w}-{y}", f"ewars-bulletin-{w}-{y}",
                         f"ewars-weekly-bulletin-{w}{_ord(w)}-week-{y}"):
                url = f"{BASE}/resources/download/{slug}"
                try:
                    r = session.head(url, timeout=30, allow_redirects=True)
                except Exception:
                    continue
                if r.status_code == 200:
                    extra.append({"title": f"(guessed slug) {slug}", "published": "",
                                  "url": url, "year": y, "week": w,
                                  "note": "not in index; found by slug guess"})
                    break
            time.sleep(0.2)
    return extra


def _ord(n):
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def download(session, row):
    if not row["week"]:
        return None
    dest = os.path.join(PDF_DIR, f"ewars_{row['year']}_wk{int(row['week']):02d}.pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    for attempt in range(3):
        try:
            r = session.get(row["url"], timeout=90, allow_redirects=True)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                with open(dest, "wb") as f:
                    f.write(r.content)
                return dest
            print(f"  non-pdf {row['url']} status={r.status_code}")
            return None
        except Exception as e:
            print(f"  retry {attempt} {row['url']}: {e}")
            time.sleep(2)
    return None


def to_text(pdf):
    txt = pdf[:-4] + ".txt"
    if os.path.exists(txt):
        return txt
    import pdfplumber
    try:
        with pdfplumber.open(pdf) as doc:
            text = "\n".join((pg.extract_text() or "") for pg in doc.pages)
    except Exception as e:
        print(f"  text fail {pdf}: {e}")
        return None
    with open(txt, "w") as f:
        f.write(text)
    return txt


def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    s = requests.Session()
    s.headers.update(UA)
    rows = harvest(s)
    print(f"index: {len(rows)} bulletins listed", flush=True)
    extra = guess_missing(s, rows, range(2016, 2027))
    print(f"slug guesses found {len(extra)} more", flush=True)
    rows += extra
    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["title", "published", "url", "year", "week",
                                          "note", "week_superseded"])
        w.writeheader()
        w.writerows(rows)
    print(f"index: {len(rows)} rows -> {INDEX_CSV}")
    rows = [r for r in rows if not r.get("week_superseded")]
    ok = 0
    for i, row in enumerate(rows):
        pdf = download(s, row)
        if pdf and to_text(pdf):
            ok += 1
        if i % 25 == 0:
            print(f"  {i}/{len(rows)} ok={ok}", flush=True)
        time.sleep(0.3)
    print(f"done: {ok}/{len(rows)} downloaded + text-extracted")


if __name__ == "__main__":
    sys.exit(main())
