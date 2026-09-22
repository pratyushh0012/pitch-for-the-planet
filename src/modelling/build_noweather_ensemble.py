"""
Weather-free version of the deployed ensemble, used by the web app.

The ablation (reports/metrics/ablation.csv) and the national rainfall test
(reports/metrics/national_rain_test.csv) both found that removing every weather
feature leaves hold-out accuracy unchanged or slightly better. So the app only
needs recent case counts and the date. This rebuilds the deployed top-k
ensemble with exactly the same members (model type, hyperparameters, target
framing, history scheme, blend weight, k) on the 12 disease + season features,
scores it on the same 2024-2025 rolling-origin hold-out, and refits it on all
data.

Members and k were chosen by select_ensemble.py (2021-2023 CV only) and are
reused unchanged -- nothing here is selected on test years.

Outputs:
  models/ensemble_noweather_h{1..4}.pkl       same layout as ensemble_h*.pkl
  reports/metrics/backtest_noweather_h{1..4}.csv
"""

import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.modelling.train_model import (  # noqa: E402
    CORE, TEST_YEARS, model_specs, rolling_origin, fit_one, metrics)

WEATHER = ["rain_mean_mm", "rain_mean_lag1", "rain_mean_lag2",
           "rain_mean_lag3", "rain_mean_roll3", "max_daily_mm",
           "humidity_pct_mean", "temp_c_mean"]
BASE = [c for c in CORE if c not in WEATHER]          # 12 cols


def main():
    cmp = pd.read_csv("reports/metrics/model_comparison.csv")
    ens_sel = pd.read_csv("reports/metrics/ensemble_selection.csv")
    specs = model_specs()
    t0 = time.time()

    for h in [1, 2, 3, 4]:
        d = pd.read_csv(f"data/processed/features_h{h}.csv").reset_index(
            drop=True)
        ev = d[(d.target_year >= TEST_YEARS[0])
               & d["age_target_week_last_year"].notna()
               & d["age_roll4_mean"].notna()].index.to_numpy()
        ev_d = d.loc[ev]
        y = ev_d["target"].values
        y24 = (ev_d.target_year == 2024).values
        y25 = (ev_d.target_year == 2025).values

        k = int(ens_sel[ens_sel.horizon == h].iloc[0]["k"])
        cc = (cmp[(cmp.horizon == h) & (cmp.framing != "-")]
              .sort_values("cv_MAE").head(k))
        members = [{"name": r.model, "build": specs[r.model][0],
                    "params": json.loads(r.params), "framing": r.framing,
                    "scheme": r.scheme, "blend_w": float(r.blend_w)}
                   for _, r in cc.iterrows()]

        preds = [rolling_origin(d, BASE, m["build"], m["params"],
                                m["framing"], m["scheme"], m["blend_w"], ev)
                 for m in members]
        ens = np.nanmean(np.vstack(preds), axis=0)
        ok = np.isfinite(ens)
        mm = metrics(y[ok], ens[ok])
        m24 = mean_absolute_error(y[ok & y24], ens[ok & y24])
        m25 = mean_absolute_error(y[ok & y25], ens[ok & y25])

        bases = cmp[(cmp.horizon == h) & (cmp.framing == "-")]
        bb = bases.loc[bases.test_MAE.idxmin()]
        imp = 100 * (bb.test_MAE - mm["MAE"]) / bb.test_MAE

        bt = pd.read_csv(f"reports/metrics/backtest_h{h}.csv")
        assert len(bt) == len(ev)
        assert (bt.year.values == ev_d.year.values).all()
        assert (bt.week.values == ev_d.week.values).all()
        bt = bt.drop(columns=["pred_single"], errors="ignore")
        bt["pred_with_weather"] = bt["pred"]
        bt["pred"] = ens
        bt["champion"] = f"No-weather ensemble top-{k}"
        bt.to_csv(f"reports/metrics/backtest_noweather_h{h}.csv", index=False)

        fitted = []
        for m in members:
            fm = fit_one(m["build"], m["params"], d, BASE, m["framing"],
                         m["scheme"])
            if fm is not None:
                fitted.append({"model": fm, "framing": m["framing"],
                               "scheme": m["scheme"], "blend_w": m["blend_w"],
                               "name": m["name"], "params": m["params"]})
        joblib.dump({"members": fitted, "features": BASE, "horizon": h,
                     "k": k, "test_MAE": mm["MAE"], "test_RMSE": mm["RMSE"],
                     "test_MAPE": mm["MAPE"], "test2024_MAE": float(m24),
                     "test2025_MAE": float(m25),
                     "best_baseline": bb.model,
                     "best_baseline_MAE": float(bb.test_MAE),
                     "improvement_pct": float(imp)},
                    f"models/ensemble_noweather_h{h}.pkl")
        print(f"h={h} k={k}: MAE {mm['MAE']:.1f} (2024 {m24:.1f} | 2025 "
              f"{m25:.1f}) MAPE {mm['MAPE']:.1f}% vs "
              f"{bb.model.replace('Baseline: ', '')} {bb.test_MAE:.1f} -> "
              f"{imp:+.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    print("[save] models/ensemble_noweather_h*.pkl, "
          "reports/metrics/backtest_noweather_h*.csv")


if __name__ == "__main__":
    main()
