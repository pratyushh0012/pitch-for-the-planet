#!/usr/bin/env python3
"""
ewars_dengue_parse.py -- weekly national dengue counts from cached EWARS bulletins.

Why a new parser: the MonsoonWatch scraper only found dengue in 14 weeks of
2021 and 28 of 2022. Cause: its prose regex required digits and the exact
phrase 'cases of Dengue were reported this week', but bulletins often write
numbers as words ('Eleven cases', 'No case', 'Zero case', 'One cases'), glue
words together ('2857cases ofDengue'), or put the count in a comparison table.
This parser tries each layout in a fixed priority order and records which one
hit, so every value is traceable to a method.

Layouts handled (priority order, first hit wins for `this week`):
  table4  2024+    'Dengue 2010 2323 313 ■ 1528'  (prev, this, change, last_year)
  table2  2020-23  'Disease Week 38 Week 37 ... Dengue 29 27' (this, prev)
  prose   2016+    '<N|words> cases of dengue are/were reported this week'
                   'No/Zero case of Dengue is reported this week'

Also captured (for cross-checks and gap recovery, never as model features):
  prev_reported    previous week's count as printed in THIS bulletin
  last_year_rep    same week last year as printed in this bulletin
  ytd              'N cases of dengue reported so far this year' (2016-18)
  sites_reporting / sites_total  reporting status where the bulletin states it

Input:  data/raw/ewars_bulletin_texts.jsonl.gz (built on first run from the
        ewars_harvest.py .txt cache)
Output: data/raw/ewars_dengue_weekly_v2.csv  (old ewars_national_weekly.csv untouched)
"""
import glob
import os
import re

import pandas as pd

PDF_DIR = os.environ.get("EWARS_PDF_DIR", "data/raw/ewars_pdfs")
TEXTS = "data/raw/ewars_bulletin_texts.jsonl.gz"   # all bulletin texts, one file
OUT = "data/raw/ewars_dengue_weekly_v2.csv"

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMTOK = set(_UNITS) | set(_TENS) | {"no", "hundred", "thousand", "and", "only"}


def number_before(t, pos):
    """Read the count that ends right before t[pos]: digits ('1,531', glued
    '2857') or a run of number words ('Fifty-nine', 'One hundred and five')."""
    head = t[max(0, pos - 60):pos].rstrip()
    m = re.search(r"(\d[\d,]*)$", head)
    if m:
        return int(m.group(1).replace(",", ""))
    toks = re.split(r"([\s-]+)", head)
    words = []
    for tok in reversed(toks):
        if re.fullmatch(r"[\s-]*", tok):
            continue
        low = tok.lower().strip("(•:")
        if low in _NUMTOK:
            words.insert(0, low)
        else:
            break
    while words and words[0] in ("and",):
        words.pop(0)
    return words_to_int(" ".join(words)) if words else None


def words_to_int(s):
    """'Fifty-nine' -> 59, 'One hundred and five' -> 105, 'No' -> 0, '1,531' -> 1531."""
    s = s.strip().lower().replace(",", "")
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for tok in re.split(r"[\s-]+", s):
        if tok in ("and", "only", ""):
            continue
        if tok == "no":
            return 0
        if tok in _UNITS:
            cur += _UNITS[tok]
        elif tok in _TENS:
            cur += _TENS[tok]
        elif tok == "hundred":
            cur = max(cur, 1) * 100
        elif tok == "thousand":
            total += max(cur, 1) * 1000
            cur = 0
        else:
            return None
    return total + cur


def norm(text):
    return " ".join(text.split())


def parse_table4(t):
    # 'Dengue 2010 2323 313 ■ 1528' or wrapped 'Confirmed Dengue Cases 58 45 -13 ■ 77 Reported'
    m = re.search(r"\bDengue(?:\s+Cases)?\s+(\d+)\s+(\d+)\s+(-?\d+)\s+\D{0,4}?(\d+)\b", t)
    if m:
        prev, this, change, last = map(int, m.groups())
        if this - prev == change:        # internal consistency check
            return this, prev, last
    return None


def parse_table2(t, week):
    # 'Disease(s) Week 38 Week 37 ... Dengue 29 27' (spaces sometimes missing)
    m = re.search(rf"Week\s*{week}\s*Week\s*{week - 1}\b(.{{0,200}}?)\bDengue\s+(\d+)\s+(\d+)", t)
    if m:
        return int(m.group(2)), int(m.group(3))
    return None


_VERB = r"(?:is|are|was|were|has\s*been|have\s*been)\s*reported"
PROSE_THIS = [
    # '<N> cases of dengue are/were reported this week' (not '... this week in 2018')
    rf"cases?\s*of\s*dengue\s*(?:cases?\s*)?{_VERB}\s*(?:in\s*)?this\s*(?:week|wee\b)(?!\s*in\s*\d)",
    # '<N> Dengue cases are reported this week'
    rf"dengue\s*cases?\s*{_VERB}\s*(?:in\s*)?this\s*week(?!\s*in\s*\d)",
]


def parse_prose(t):
    for pat in PROSE_THIS:
        for m in re.finditer(pat, t, re.I):
            # skip DPHO / non-EWARS side reports ('34 Dengue cases ... through DPHO')
            tail = t[m.end():m.end() + 30].lower()
            if "through dpho" in tail:
                continue
            v = number_before(t, m.start())
            if v is not None:
                return v
    return None


def parse_last_year(t):
    m = re.search(rf"cases?\s*of\s*dengue\s*{_VERB}\s*this\s*week\s*in\s*20\d\d", t, re.I)
    if m:
        return number_before(t, m.start())
    if re.search(r"no case of dengue was reported this week in\s*20\d\d", t, re.I):
        return 0
    return None


def parse_ytd(t):
    m = re.search(rf"cases?\s*of\s*dengue\s*{_VERB}\s*so\s*far\s*this\s*year", t, re.I)
    return number_before(t, m.start()) if m else None


def parse_sites(t):
    m = re.search(r"Out of (\d+) functional EWARS sentinel sites, report was received from (\d+)", t)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"with (\d+) out of (\d+) sentinel sites reporting", t)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


_MONTHS = "january february march april may june july august september october november december".split()


def parse_header_date(t):
    """Bulletin date from the first ~600 chars: '29th May 2016', '6 August 2017',
    'September 29, 2024'. This is the Sunday the epi-week ENDS (publication may
    lag), used to calibrate the EDCD calendar."""
    head = t[:600]
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*(" + "|".join(_MONTHS) + r")\s*,?\s*(20\d\d)", head, re.I)
    if m:
        d, mon, y = int(m.group(1)), _MONTHS.index(m.group(2).lower()) + 1, int(m.group(3))
    else:
        m = re.search(r"(" + "|".join(_MONTHS) + r")\s*(\d{1,2})(?:st|nd|rd|th)?,?\s*(20\d\d)", head, re.I)
        if not m:
            return None
        mon, d, y = _MONTHS.index(m.group(1).lower()) + 1, int(m.group(2)), int(m.group(3))
    try:
        return f"{y:04d}-{mon:02d}-{d:02d}"
    except ValueError:
        return None


def parse_text(name, text):
    y, w = map(int, re.search(r"ewars_(\d{4})_wk(\d+)", name).groups())
    t = norm(text)
    rec = {"year": y, "week": w, "dengue_cases": None, "method": None,
           "prev_reported": None, "last_year_rep": None, "ytd": parse_ytd(t),
           "header_date": parse_header_date(norm(text)), "source_file": name}
    rec["sites_reporting"], rec["sites_total"] = parse_sites(t)
    rec["last_year_rep"] = parse_last_year(t)
    prose = parse_prose(t)
    t4 = parse_table4(t)
    t2 = parse_table2(t, w) if w > 1 else None
    # Priority: table4 (internally consistent: this - prev == change) > prose > table2.
    # Prose beats table2 because in the 5 bulletins where they disagree the table
    # looks like a later revision (late reports added), not the as-published count.
    if t4:
        rec.update(dengue_cases=t4[0], method="table4", prev_reported=t4[1], last_year_rep=t4[2])
    elif prose is not None:
        rec.update(dengue_cases=prose, method="prose", prev_reported=t2[1] if t2 else None)
    elif t2:
        rec.update(dengue_cases=t2[0], method="table2", prev_reported=t2[1])
    # cross-check flag: prose and table disagree
    if t2 and prose is not None and t2[0] != prose:
        rec["conflict"] = f"table2={t2[0]} prose={prose} (prose used)"
    if t4 and prose is not None and t4[0] != prose:
        rec["conflict"] = f"table4={t4[0]} prose={prose} (table4 used)"
    return rec


def load_texts():
    """Bulletin texts, from the consolidated archive if present, else from the
    per-bulletin .txt cache (which is then consolidated). One archive file
    instead of ~480 small ones: this project folder is iCloud-synced and
    small files get evicted, making per-file reads very slow."""
    if os.path.exists(TEXTS):
        return pd.read_json(TEXTS, lines=True)
    files = sorted(glob.glob(os.path.join(PDF_DIR, "ewars_*_wk*.txt")))
    df = pd.DataFrame({"name": [os.path.basename(f)[:-4] for f in files],
                       "text": [open(f).read() for f in files]})
    df.to_json(TEXTS, orient="records", lines=True)
    return df


def main():
    texts = load_texts()
    df = pd.DataFrame([parse_text(n, t) for n, t in zip(texts.name, texts.text)])
    df = df[df.year >= 2016].sort_values(["year", "week"])
    df.to_csv(OUT, index=False)
    print(f"{len(df)} bulletins -> {OUT}")
    print(df.groupby("year").agg(bulletins=("week", "size"),
                                 dengue_found=("dengue_cases", "count"),
                                 total=("dengue_cases", "sum")).to_string())
    print(df.method.value_counts(dropna=False).to_string())
    if "conflict" in df:
        print("conflicts:\n", df[df.conflict.notna()][["year", "week", "dengue_cases", "method", "conflict"]].to_string())


if __name__ == "__main__":
    main()
