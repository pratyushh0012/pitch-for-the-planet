#!/usr/bin/env python3
"""
train_final.py -- Phase 6: fit the chosen models on ALL usable seasons and save.

Reads the choice from reports/evaluation_summary.json (written by evaluate.py).
Saves models/dengue/denguewatch.joblib containing both M1 variants (A: no
weather, B: weather) so the app's what-if page can show either, plus which
component is the official regressor / classifier, the feature lists, the surge
line table, thresholds and metadata. Also writes data/processed/features.csv.
"""
import datetime as dt
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import dengue_features as F  # noqa: E402
from src import models as M  # noqa: E402
from src import targets as T  # noqa: E402
from src.evaluate import AMBER, GREEN, fold_frame, usable_seasons  # noqa: E402

OUT_DIR = os.path.join(ROOT, "models/dengue")
TABLE = os.path.join(ROOT, "data/processed/weekly_table.csv")


def main():
    summ = json.load(open(os.path.join(ROOT, "reports/evaluation_summary.json")))
    chosen_reg, chosen_cls, method = summ["chosen"], summ["chosen_cls"], summ["line_method"]
    table = pd.read_csv(TABLE).sort_values(["year", "week"]).reset_index(drop=True)
    years = usable_seasons(table)
    t, line = fold_frame(table, years, method)
    warm = F.warm_cutoff(table, years)
    t.to_csv(os.path.join(ROOT, "data/processed/features.csv"), index=False)

    fitted = {}
    for key in sorted({chosen_reg, chosen_cls, "M1_A", "M1_B"}):
        mdl = M.make_model(key)
        need = mdl.feats + ([mdl.spline_col] if getattr(mdl, "spline_col", None) else [])
        tr = t[t.year.isin(years) & t[need].notna().all(axis=1) & t.y_reg.notna() & t.y_cls.notna()]
        fitted[key] = mdl.fit(tr)
        print(f"{key}: {len(tr)} training weeks, {int(tr.y_cls.sum())} positive, "
              f"family={getattr(mdl, 'family_used', '-')}, alpha={getattr(mdl, 'alpha', float('nan')):.3f}")

    # current forecast from the latest week with complete features
    last = t[t[F.SET_B].notna().all(axis=1)].iloc[-1:]
    reg = fitted[chosen_reg].predict(last).iloc[0]
    cls = fitted[chosen_cls].predict(last).iloc[0]
    bundle = {
        "models": fitted, "chosen_reg": chosen_reg, "chosen_cls": chosen_cls,
        "features": {k: m.feats for k, m in fitted.items()},
        "surge_line": line, "line_method": method, "thresholds": {"green": GREEN, "amber": AMBER, "red_p90_x_line": 2.0},
        "warm_cutoff": warm,
        # training range of every feature -- the app clamps what-if inputs to it
        "feature_range": {c: (float(t.loc[t.year.isin(years), c].min()), float(t.loc[t.year.isin(years), c].max()))
                          for c in F.SET_B}, "train_years": years, "trained_on": str(dt.date.today()),
        "latest": {"year": int(last.year.iloc[0]), "week": int(last.week.iloc[0]),
                   "mean": float(reg["mean"]), "p10": float(reg.p10), "p90": float(reg.p90),
                   "prob": float(cls.prob)},
        "evaluation": summ,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "denguewatch.joblib")
    joblib.dump(bundle, path)
    print(f"latest week {bundle['latest']['year']}-W{bundle['latest']['week']:02d}: "
          f"t+4 = {reg['mean']:.0f} [{reg.p10:.0f}–{reg.p90:.0f}], P(surge onset in 4 wk) = {cls.prob:.2f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
