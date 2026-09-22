"""
Does rainfall actually earn its place in the model?

The whole pitch rests on rainfall carrying predictive signal, so this tests it
directly instead of assuming it. The deployed ensemble is rebuilt on three
feature sets and re-scored on the same hold-out weeks by the same rolling-origin
protocol:

  full            everything (disease + seasonality + rainfall + climate)
  no_weather      disease history + seasonality only
  weather_only    rainfall + climate + seasonality, no disease history

Output: reports/metrics/ablation.csv
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.modelling.train_model import (  # noqa: E402
    CORE, TEST_YEARS, model_specs, rolling_origin)

RAIN = ["rain_mean_mm", "rain_mean_lag1", "rain_mean_lag2", "rain_mean_lag3",
        "rain_mean_roll3", "max_daily_mm", "humidity_pct_mean", "temp_c_mean"]
SEASON = ["tgt_sin", "tgt_cos", "tgt_is_monsoon", "tgt_week_of_year"]
DISEASE = [c for c in CORE if c not in RAIN + SEASON]

SETS = {
    "full (deployed)": CORE,
    "no weather": DISEASE + SEASON,
    "weather + season only": RAIN + SEASON,
}


def main():
    cmp = pd.read_csv("reports/metrics/model_comparison.csv")
    ens_sel = pd.read_csv("reports/metrics/ensemble_selection.csv")
    specs = model_specs()
    out = []

    for h in [1, 2, 3, 4]:
        d = pd.read_csv(f"data/processed/features_h{h}.csv").reset_index(
            drop=True)
        ev = d[(d.target_year >= TEST_YEARS[0])
               & d["age_target_week_last_year"].notna()
               & d["age_roll4_mean"].notna()].index.to_numpy()
        ev_d = d.loc[ev]
        y = ev_d["target"].values
        k = int(ens_sel[ens_sel.horizon == h].iloc[0]["k"])
        cc = (cmp[(cmp.horizon == h) & (cmp.framing != "-")]
              .sort_values("cv_MAE").head(k))
        members = [{"build": specs[r.model][0], "params": json.loads(r.params),
                    "framing": r.framing, "scheme": r.scheme,
                    "blend_w": float(r.blend_w)} for _, r in cc.iterrows()]

        row = {"horizon": h, "k": k}
        for tag, feats in SETS.items():
            fc = [c for c in feats if c in d.columns]
            preds = [rolling_origin(d, fc, m["build"], m["params"],
                                    m["framing"], m["scheme"], m["blend_w"],
                                    ev) for m in members]
            e = np.nanmean(np.vstack(preds), axis=0)
            ok = np.isfinite(e)
            row[tag] = round(mean_absolute_error(y[ok], e[ok]), 1)
        row["rainfall contribution"] = round(
            100 * (row["no weather"] - row["full (deployed)"])
            / row["no weather"], 1)
        out.append(row)
        print(f"h={h}: full {row['full (deployed)']:.1f} | "
              f"no weather {row['no weather']:.1f} | "
              f"weather only {row['weather + season only']:.1f} | "
              f"rainfall worth {row['rainfall contribution']:+.1f}%")

    df = pd.DataFrame(out)
    df.to_csv("reports/metrics/ablation.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("[save] reports/metrics/ablation.csv")


if __name__ == "__main__":
    main()
