"""
epi_calendar.py -- EDCD epi-week calendar, calibrated from the bulletins themselves.

Why not src/common/epiweek.py: it assumes 52 weeks every year and extrapolates
backwards from 2021 in 364-day steps. The bulletins show 2020 had a WEEK 53
(header '53rd Epidemiological Week, 10 January 2021'), so every pre-2021 week in
that module is shifted by one week. MonsoonWatch only used 2021+, so it never
hit this; DengueWatch goes back to 2016, so it matters. That module is left
untouched (MonsoonWatch still imports it).

Method: each bulletin header carries the date the epi-week ends (a Sunday).
anchor(year) = header_date - 7 * (week - 1) is the end of week 1. Publication
sometimes lags (backlogs get stamped with one date), so take the MODE per year.
Years with no usable header fall back to the rule observed in every calibrated
year: week 1 ends on the first Sunday on or after 11 January.
"""
import datetime as dt
from collections import Counter

import pandas as pd


def rule_anchor(year):
    d = dt.date(year, 1, 11)
    return d + dt.timedelta(days=(6 - d.weekday()) % 7)   # next Sunday (weekday 6)


def calibrate(bulletins):
    """bulletins: DataFrame with year, week, header_date. Returns
    (anchors {year: date}, checks DataFrame) -- checks has one row per year."""
    cand = {}
    for y, w, h in bulletins[["year", "week", "header_date"]].dropna().itertuples(index=False):
        d = dt.date.fromisoformat(h) - dt.timedelta(days=7 * (int(w) - 1))
        if d.weekday() == 6:                               # only on-schedule Sundays
            cand.setdefault(int(y), []).append(d)
    years = range(min(bulletins.year), max(bulletins.year) + 2)
    anchors, rows = {}, []
    for y in years:
        c = Counter(cand.get(y, []))
        if c:
            a, n = c.most_common(1)[0]
            anchors[y] = a
            rows.append({"year": y, "anchor_week1_end": a, "source": "bulletin headers",
                         "n_agree": n, "n_headers": sum(c.values()),
                         "rule_anchor": rule_anchor(y), "rule_matches": a == rule_anchor(y)})
        else:
            anchors[y] = rule_anchor(y)
            rows.append({"year": y, "anchor_week1_end": anchors[y], "source": "rule (no headers)",
                         "n_agree": 0, "n_headers": 0, "rule_anchor": rule_anchor(y),
                         "rule_matches": True})
    return anchors, pd.DataFrame(rows)


def weeks_in_year(anchors, y):
    nxt = anchors.get(y + 1, rule_anchor(y + 1))
    return (nxt - anchors[y]).days // 7


def date_to_epiweek(d, anchors):
    for y in (d.year + 1, d.year, d.year - 1):
        if y not in anchors:
            continue
        start = anchors[y] - dt.timedelta(days=6)
        off = (d - start).days
        if 0 <= off < 7 * weeks_in_year(anchors, y):
            return y, off // 7 + 1
    return None, None


def epiweek_end(year, week, anchors):
    return anchors[year] + dt.timedelta(days=7 * (int(week) - 1))


def all_weeks(anchors, first_year, last_year):
    """Continuous (year, week, week_end) index."""
    rows = []
    for y in range(first_year, last_year + 1):
        for w in range(1, weeks_in_year(anchors, y) + 1):
            rows.append((y, w, epiweek_end(y, w, anchors)))
    return pd.DataFrame(rows, columns=["year", "week", "week_end"])
