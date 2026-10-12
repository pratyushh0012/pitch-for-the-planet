"""
targets.py -- Phase 2: surge line, surge onsets and prediction targets.

rate_t = cases_t (site counts missing for 2020-2023 -- see data_audit_report.md).

The surge line is ALWAYS computed from a given set of training seasons and then
applied to any week; it is never computed from the full dataset inside a fold.
"""
import numpy as np
import pandas as pd

HORIZON = 4
WINDOW = 2          # +/- weeks pooled per week-of-year
Z = 2.0             # mean + Z * SD


def woy(week):
    """Week-of-year for seasonal maths: EDCD week 53 (2020 only) is treated as 52."""
    return np.minimum(np.asarray(week), 52)


METHODS = ("meansd", "q3")


def surge_line(table, train_years, method="meansd"):
    """Per week-of-year threshold from the rate in that week-of-year, pooled over a
    +/-2-week circular window, using ONLY `train_years`.

    meansd  mean + 2 SD (the CLAUDE.md default). With 3-7 seasons that mix ~0-case
            years and outbreak years, an outbreak season inflates the SD so much
            that it never crosses its own line -> almost no positive training labels.
    q3      upper quartile (Bortman 1999 endemic channel: 'epidemic zone' above Q3
            of previous years). Robust to one extreme season."""
    tr = table[table.year.isin(train_years)].dropna(subset=["cases"])
    w = woy(tr.week)
    rows = []
    for k in range(1, 53):
        dist = np.abs(w - k)
        dist = np.minimum(dist, 52 - dist)              # circular: wk52 neighbours wk1
        vals = tr.cases.values[dist <= WINDOW]
        line = vals.mean() + Z * vals.std(ddof=1) if method == "meansd" else np.quantile(vals, 0.75)
        rows.append({"woy": k, "mean": vals.mean(), "sd": vals.std(ddof=1), "q3": np.quantile(vals, 0.75),
                     "line": line, "n": len(vals)})
    return pd.DataFrame(rows)


def add_line(table, line):
    t = table.copy()
    t["woy"] = woy(t.week)
    return t.merge(line[["woy", "line"]], on="woy", how="left").sort_values(["year", "week"]).reset_index(drop=True)


def add_targets(t):
    """Needs columns cases, line. Adds:
    above      cases > line (NaN if cases missing)
    onset      first week of a run of >= 2 consecutive above-line weeks
    y_cls      1 if an onset occurs in any of t+1..t+4
    y_reg      cases at t+4            (primary)
    y_reg_sum  sum of cases t+1..t+4   (secondary)
    target_end index offset of the last week each target looks at (for leakage-safe splits)"""
    t = t.copy()
    above = (t.cases > t.line).astype(float).where(t.cases.notna())
    nxt = above.shift(-1)
    prev = above.shift(1)
    onset = ((above == 1) & (nxt == 1) & (prev != 1)).astype(float)
    onset[above.isna() | nxt.isna()] = np.nan
    t["above"], t["onset"] = above, onset
    fut_onsets = pd.concat([onset.shift(-h) for h in range(1, HORIZON + 1)], axis=1)
    y = (fut_onsets == 1).any(axis=1).astype(float)
    # unknown if any needed future onset flag is missing and none of the known ones is 1
    y[fut_onsets.isna().any(axis=1) & (y == 0)] = np.nan
    t["y_cls"] = y
    t["y_reg"] = t.cases.shift(-HORIZON)
    t["y_reg_sum"] = sum(t.cases.shift(-h) for h in range(1, HORIZON + 1))
    t.loc[pd.concat([t.cases.shift(-h) for h in range(1, HORIZON + 1)], axis=1).isna().any(axis=1), "y_reg_sum"] = np.nan
    # y_cls at t looks at onset flags up to t+4, and onset at s needs cases at s+1 -> t+5
    t["target_end"] = HORIZON + 1
    return t


TARGET_COLS = ["above", "onset", "y_cls", "y_reg", "y_reg_sum", "line", "target_end"]
