"""
Does NATIONAL rainfall improve the forecast, where valley rainfall did not?

Same protocol as ablation.py, so the numbers are directly comparable with
reports/metrics/ablation.csv: the deployed top-k ensemble members (same model
types, framings, schemes and hyperparameters), the same 2024-2025 hold-out
weeks, the same rolling-origin retraining. Only the 8 weather columns in the
feature set are swapped for another rainfall source.

The hyperparameters were tuned with valley rainfall in the feature set. That
slightly favours the deployed version, so any gain from a swapped-in source is
if anything understated.

Feature sets:
  no weather          disease + season only (reproduces ablation.csv -- a
                      check that this harness matches the original)
  ktm point (NASA)    Kathmandu grid cell, NASA for all years (no gauge bridge)
  national (NASA)     all-Nepal average
  terai (NASA)        southern plains average
  national anomaly    all-Nepal, rain expressed as wetter/drier than normal
  case-weighted       districts weighted by AGE case share -- uses 2024-25
                      district lists, so an optimistic upper bound

Output: reports/metrics/national_rain_test.csv
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.modelling.train_model import (  # noqa: E402
    CORE, TEST_YEARS, model_specs, rolling_origin, metrics)

RAIN = ["rain_mean_mm", "rain_mean_lag1", "rain_mean_lag2", "rain_mean_lag3",
        "rain_mean_roll3", "max_daily_mm", "humidity_pct_mean", "temp_c_mean"]
BASE = [c for c in CORE if c not in RAIN]          # disease + season, 12 cols


def weather(tag, anom=False):
    r = f"{tag}_rain_anom" if anom else f"{tag}_rain"
    return [r, f"{r}_lag1", f"{r}_lag2", f"{r}_lag3", f"{r}_roll3",
            f"{tag}_max_daily", f"{tag}_humidity", f"{tag}_temp"]


SETS = {
    "no weather": BASE,
    "ktm point (NASA)": BASE + weather("ktm"),
    "national (NASA)": BASE + weather("nat"),
    "terai (NASA)": BASE + weather("terai"),
    "national anomaly (NASA)": BASE + weather("nat", anom=True),
    "case-weighted (NASA, optimistic)": BASE + weather("casew"),
}


def main():
    cmp = pd.read_csv("reports/metrics/model_comparison.csv")
    ens_sel = pd.read_csv("reports/metrics/ensemble_selection.csv")
    abl = pd.read_csv("reports/metrics/ablation.csv")
    nat = pd.read_csv("data/interim/national_rain_weekly.csv")
    specs = model_specs()
    out, t0 = [], time.time()

    for h in [1, 2, 3, 4]:
        d0 = pd.read_csv(f"data/processed/features_h{h}.csv")
        d = d0.merge(nat, on=["year", "week"], how="left")
        assert len(d) == len(d0) and (d.year.values == d0.year.values).all()
        d = d.reset_index(drop=True)
        ev = d[(d.target_year >= TEST_YEARS[0])
               & d["age_target_week_last_year"].notna()
               & d["age_roll4_mean"].notna()].index.to_numpy()
        ev_d = d.loc[ev]
        y = ev_d["target"].values
        y24 = (ev_d.target_year == 2024).values
        y25 = (ev_d.target_year == 2025).values
        missing = {c: int(d.loc[ev, c].isna().sum())
                   for s in SETS.values() for c in s if c in nat.columns}
        if any(missing.values()):
            print(f"  !! h={h} NaN in test rows: "
                  f"{ {k: v for k, v in missing.items() if v} }")

        k = int(ens_sel[ens_sel.horizon == h].iloc[0]["k"])
        cc = (cmp[(cmp.horizon == h) & (cmp.framing != "-")]
              .sort_values("cv_MAE").head(k))
        members = [{"build": specs[r.model][0], "params": json.loads(r.params),
                    "framing": r.framing, "scheme": r.scheme,
                    "blend_w": float(r.blend_w)} for _, r in cc.iterrows()]

        a = abl[abl.horizon == h].iloc[0]
        row = {"horizon": h, "k": k,
               "valley gauges (deployed)": a["full (deployed)"]}
        for tag, feats in SETS.items():
            preds = [rolling_origin(d, feats, m["build"], m["params"],
                                    m["framing"], m["scheme"], m["blend_w"],
                                    ev) for m in members]
            e = np.nanmean(np.vstack(preds), axis=0)
            ok = np.isfinite(e)
            row[tag] = round(mean_absolute_error(y[ok], e[ok]), 1)
            row[f"{tag} | 2024"] = round(
                mean_absolute_error(y[ok & y24], e[ok & y24]), 1)
            row[f"{tag} | 2025"] = round(
                mean_absolute_error(y[ok & y25], e[ok & y25]), 1)
            row[f"{tag} | MAPE"] = round(metrics(y[ok], e[ok])["MAPE"], 1)
            print(f"  h={h} {tag:34s} MAE {row[tag]:5.1f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        chk = row["no weather"] - a["no weather"]
        print(f"  h={h} harness check: no-weather {row['no weather']} vs "
              f"ablation.csv {a['no weather']} (diff {chk:+.1f})", flush=True)
        out.append(row)

    df = pd.DataFrame(out)
    df.to_csv("reports/metrics/national_rain_test.csv", index=False)
    main_cols = (["horizon", "valley gauges (deployed)"] + list(SETS))
    print("\nMAE on 2024-2025 hold-out (lower is better)")
    print(df[main_cols].to_string(index=False))
    print("[save] reports/metrics/national_rain_test.csv")


if __name__ == "__main__":
    main()
