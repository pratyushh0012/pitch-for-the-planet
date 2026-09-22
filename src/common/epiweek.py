"""
epiweek.py -- EDCD (Nepal) epidemiological-week calendar.

EDCD's bulletin week numbering is NOT ISO week. Using .isocalendar() to join
climate data to EWARS disease counts misaligns the two series by 1-3 weeks
depending on the time of year.

Confirmed rule: EDCD epi-weeks run SUNDAY-ENDING, numbered 1-52 per year
(no week 53 appears in any cached bulletin).

Anchors (the date on which week 1 of that year ENDS):
  2021 -> 2021-01-17   from our own cached PDFs (wk1 & wk5, both Sunday,
                       exactly 28 days apart)
  2022 -> 2022-01-16   independently back-solved from two ReliefWeb-hosted
                       bulletins (wk15 & wk27, both Sunday, 84 days apart)
The 2021->2022 shift is exactly 364 days (= 52 published weeks), so every
other year is extrapolated on the same 364-day stride.

CAUTION: bulletin PDF header dates are not reliable anchors on their own --
EDCD sometimes publishes a multi-week backlog on a single date (2025 weeks
11-16 are all stamped "2nd May 2025" despite being distinct bulletins).
Only bulletins confirmed published on-schedule (a Sunday) were used above.
"""

import datetime as dt

ANCHOR_2021 = dt.date(2021, 1, 17)

ANCHORS = {2021: ANCHOR_2021}
for _y in range(2020, 2016, -1):
    ANCHORS[_y] = ANCHORS[_y + 1] - dt.timedelta(days=364)
for _y in range(2022, 2028):
    ANCHORS[_y] = ANCHORS[_y - 1] + dt.timedelta(days=364)


def date_to_epiweek(d):
    """Map a datetime.date to EDCD (year, week). Returns (None, None) if
    the date falls outside the anchor table."""
    for y in (d.year - 1, d.year, d.year + 1):
        if y not in ANCHORS:
            continue
        week1_start = ANCHORS[y] - dt.timedelta(days=6)
        offset = (d - week1_start).days
        if 0 <= offset < 7 * 52:
            return y, offset // 7 + 1
    return None, None


def epiweek_to_end_date(year, week):
    """Inverse: the Sunday on which (year, week) ends. Useful for plotting
    a weekly series on a real calendar axis."""
    if year not in ANCHORS:
        return None
    return ANCHORS[year] + dt.timedelta(days=7 * (int(week) - 1))


def assign_epiweek(df, date_index=True, date_col=None):
    """Add EDCD 'year'/'week' columns to a daily DataFrame."""
    src = df.index if date_index else df[date_col]
    yw = [date_to_epiweek(d.date() if hasattr(d, "date") else d) for d in src]
    out = df.copy()
    out["year"] = [x[0] for x in yw]
    out["week"] = [x[1] for x in yw]
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    out["week"] = out["week"].astype(int)
    return out
