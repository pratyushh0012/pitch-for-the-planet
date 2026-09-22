"""
Phase 3 -- Climate bridge.

DHM gauge rainfall ends 2023-12-31; the EWARS disease series runs to 2025.
This calibrates NASA POWER (satellite/reanalysis, Kathmandu point) against the
23-station DHM gauge average on the 2021-2023 OVERLAP, then uses that fit to
produce DHM-equivalent rainfall estimates for 2024-2025.

Every produced row carries `rain_source_is_dhm` (1 = real gauges, 0 = estimated)
so both the model and the pitch can tell ground truth from estimate.

Output: data/interim/rainfall_weekly_bridged.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

DHM = "data/interim/rainfall_weekly_dhm.csv"
NASA = "data/interim/nasa_power_weekly.csv"
OUT = "data/interim/rainfall_weekly_bridged.csv"
REPORT = "reports/metrics/climate_bridge_calibration.csv"

# Each DHM target and the NASA columns used to predict it.
TARGETS = {
    "rain_mean_mm":  ["nasa_rain_sum", "nasa_wet_days", "nasa_rain_max_daily"],
    "rain_max_mm":   ["nasa_rain_sum", "nasa_wet_days", "nasa_rain_max_daily"],
    "rain_std_mm":   ["nasa_rain_sum", "nasa_wet_days", "nasa_rain_max_daily"],
    "wet_days":      ["nasa_wet_days", "nasa_rain_sum"],
    "max_daily_mm":  ["nasa_rain_max_daily", "nasa_rain_sum"],
}


def main():
    dhm = pd.read_csv(DHM)
    nasa = pd.read_csv(NASA)
    print(f"[load] DHM {len(dhm)} weeks, NASA {len(nasa)} weeks")

    ov = dhm.merge(nasa, on=["year", "week"], how="inner")
    ov = ov[ov.year.between(2021, 2023)]
    print(f"[overlap] {len(ov)} weeks in 2021-2023 used for calibration")

    # Headline check the plan asks for: simple DHM ~ NASA weekly rainfall.
    simple = LinearRegression().fit(ov[["nasa_rain_sum"]], ov["rain_mean_mm"])
    simple_r2 = r2_score(ov["rain_mean_mm"],
                         simple.predict(ov[["nasa_rain_sum"]]))
    corr = ov["rain_mean_mm"].corr(ov["nasa_rain_sum"])
    print(f"\n[headline] rain_mean_mm ~ nasa_rain_sum")
    print(f"           slope  = {simple.coef_[0]:.3f}")
    print(f"           interc = {simple.intercept_:.3f}")
    print(f"           R2     = {simple_r2:.3f}   (plan threshold: 0.50)")
    print(f"           Pearson r = {corr:.3f}")
    if simple_r2 < 0.50:
        print("  !! R2 below 0.50 -- plan says fall back to 2021-2023 only")
    else:
        print("  OK R2 above threshold -- bridging 2024-2025 is defensible")

    # Fit each DHM feature. Held-out year check = train 21/22, test 23.
    rows, models = [], {}
    for tgt, preds in TARGETS.items():
        X, y = ov[preds], ov[tgt]
        m = LinearRegression().fit(X, y)
        r2_in = r2_score(y, m.predict(X))

        tr = ov[ov.year <= 2022]
        te = ov[ov.year == 2023]
        m_cv = LinearRegression().fit(tr[preds], tr[tgt])
        r2_out = r2_score(te[tgt], m_cv.predict(te[preds]))
        mae_out = mean_absolute_error(te[tgt], m_cv.predict(te[preds]))

        models[tgt] = (m, preds)
        rows.append({"dhm_feature": tgt, "predictors": "+".join(preds),
                     "r2_insample": round(r2_in, 3),
                     "r2_holdout_2023": round(r2_out, 3),
                     "mae_holdout_2023": round(mae_out, 2),
                     "dhm_mean": round(y.mean(), 2)})

    cal = pd.DataFrame(rows)
    print("\n[calibration] fit quality per DHM feature")
    print(cal.to_string(index=False))
    cal.to_csv(REPORT, index=False)

    # Build the bridged series: real DHM where we have it, estimate after.
    dhm_part = dhm[dhm.year.between(2021, 2023)].copy()
    dhm_part["rain_source_is_dhm"] = 1

    future = nasa[nasa.year.between(2024, 2025)].copy()
    est = future[["year", "week"]].copy()
    for tgt, (m, preds) in models.items():
        est[tgt] = np.clip(m.predict(future[preds]), 0, None)
    est["n_stations"] = np.nan
    est["rain_source_is_dhm"] = 0
    print(f"\n[bridge] estimated {len(est)} weeks for 2024-2025")

    bridged = pd.concat([dhm_part, est], ignore_index=True)
    bridged = bridged.merge(
        nasa[["year", "week", "temp_c_mean", "temp_c_max",
              "humidity_pct_mean", "nasa_rain_sum"]],
        on=["year", "week"], how="left")
    bridged = bridged.sort_values(["year", "week"]).reset_index(drop=True)

    print(f"[bridge] total {len(bridged)} weeks "
          f"({int(bridged.rain_source_is_dhm.sum())} gauge, "
          f"{int((1 - bridged.rain_source_is_dhm).sum())} estimated)")

    # Sanity: estimated years should look seasonally plausible, not flat.
    ann = bridged.groupby("year")["rain_mean_mm"].sum().round(0)
    print("\n[sanity] annual rainfall by year (mm):")
    print(ann.to_string())
    print("         (DHM known: 2021=1791, 2022=1830, 2023=1553)")

    bridged.to_csv(OUT, index=False)
    print(f"\n[save] {OUT}")


if __name__ == "__main__":
    main()
