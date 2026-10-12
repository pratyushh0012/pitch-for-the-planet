"""
Leakage tests: features at row t depend only on rows <= t, and no target column
is ever a model input.

Method: build features on the real table, then scramble every value AFTER row t
(cases and weather) and rebuild. Features at rows <= t must not change.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import dengue_features as F  # noqa: E402
from src import targets as T  # noqa: E402

TABLE = os.path.join(ROOT, "data/processed/weekly_table.csv")
INPUTS = ["cases", "tmean", "tmin", "tmax", "rain", "rh"]


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(TABLE).sort_values(["year", "week"]).reset_index(drop=True)


@pytest.mark.parametrize("cut", [60, 150, 260, 330, 400, 480])
def test_features_use_only_past(table, cut):
    base = F.build_features(table)
    rng = np.random.default_rng(42)
    scrambled = table.copy()
    for c in INPUTS:
        n = len(scrambled) - cut - 1
        scrambled.loc[cut + 1:, c] = rng.uniform(0, 5000, n)
    pert = F.build_features(scrambled)
    # log_last_season uses the previous *calendar year*; rows in the cut's own year
    # are unaffected only if that year is complete before the cut -- the feature
    # reads year-1 totals, which end before any row of year y, so it is still past-only.
    a = base.loc[:cut, F.SET_B].to_numpy(dtype=float)
    b = pert.loc[:cut, F.SET_B].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True), "a feature at row <= t changed when rows > t changed"


def test_future_change_moves_targets_not_features(table):
    """Sanity check of the test itself: scrambling the future DOES move targets."""
    line = T.surge_line(table, [2018, 2019, 2020])
    t1 = T.add_targets(T.add_line(table, line))
    s = table.copy()
    s.loc[301:, "cases"] = 9999
    t2 = T.add_targets(T.add_line(s, line))
    assert not np.allclose(t1.y_reg.iloc[290:300], t2.y_reg.iloc[290:300], equal_nan=True)


def test_targets_never_inputs():
    for feats in (F.SET_A, F.SET_B, F.M1_A, F.M1_B):
        assert not set(feats) & set(T.TARGET_COLS + ["cases"]), "target or raw same-week column used as feature"
        assert not any(f.startswith("y_") for f in feats)


def test_surge_line_uses_only_train_years(table):
    """Changing a non-training year must not change the surge line."""
    a = T.surge_line(table, [2018, 2019, 2020])
    s = table.copy()
    s.loc[s.year == 2022, "cases"] = 1e6
    b = T.surge_line(s, [2018, 2019, 2020])
    assert np.allclose(a.line, b.line)
