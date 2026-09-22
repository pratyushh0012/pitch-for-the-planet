"""
National rainfall indices, and the question: does Kathmandu represent Nepal?

Builds four weekly climate indices from NASA POWER, all on the EDCD epi-week
calendar (source: data/interim/nasa_power_districts_weekly.csv):

  ktm     the Kathmandu grid cell -- what the project used until now
  nat     all of Nepal: mean over the 39 distinct grid cells (each cell once,
          so the four Kathmandu Valley districts do not count four times)
  terai   the southern plains only (mean over distinct Terai cells), where
          most of the population and most top-5 AGE districts are
  casew   districts weighted by their share of AGE cases in the EWARS top-5
          district lists. CAUTION: those lists only exist for 2024-2025, which
          is also the test period, so this index uses test-period knowledge of
          WHERE cases occur (not of weekly counts). Treat its score as a
          best-case upper bound, not a clean result.

For each index: weekly rain, 1-3 week lags, 3-week total, heaviest day,
humidity, temperature, and a rainfall ANOMALY (rain minus the 2021-2023 average
for that week of the year). The anomaly is the part of rainfall the calendar
cannot already tell the model -- "wetter or drier than normal for this week".
Climatology uses 2021-2023 only, so no test-year information leaks into it.

Outputs:
  data/interim/national_rain_weekly.csv
  reports/metrics/national_rain_representativeness.csv
  reports/metrics/national_rain_age_correlation.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

SRC = "data/interim/nasa_power_districts_weekly.csv"
KTM_ORIG = "data/interim/nasa_power_weekly.csv"
EWARS = "data/raw/ewars_national_weekly.csv"
DISTRICT_CASES = "data/raw/ewars_age_districts_long.csv"
OUT = "data/interim/national_rain_weekly.csv"
REP = "reports/metrics/national_rain_representativeness.csv"
COR = "reports/metrics/national_rain_age_correlation.csv"
VARS = ["rain_sum", "rain_max_daily", "humidity_pct_mean", "temp_c_mean"]
CLIM_YEARS = (2021, 2023)


def cell_mean(df):
    """Average over distinct grid cells, each cell counted once."""
    per_cell = df.groupby(["year", "week", "cell_id"])[VARS].mean()
    return per_cell.groupby(["year", "week"]).mean()


def case_weights(districts):
    c = pd.read_csv(DISTRICT_CASES)
    c["district"] = (c["district"].str.upper().str.strip()
                     .replace({"KAPILVASTU": "KAPILBASTU"}))
    w = c.groupby("district")["age_cases"].sum()
    missing = sorted(set(w.index) - set(districts))
    if missing:
        print(f"[casew] no coordinates for {missing} -- excluded")
    w = w[w.index.isin(districts)]
    print(f"[casew] {len(w)} districts carry weight; top 5: "
          + ", ".join(f"{k} {v / w.sum():.0%}" for k, v in
                      w.sort_values(ascending=False).head(5).items()))
    return w / w.sum()


def weighted_mean(df, w):
    d = df[df.district.isin(w.index)].copy()
    d["w"] = d.district.map(w)
    g = d.groupby(["year", "week"])
    return pd.DataFrame({v: g.apply(lambda x: np.average(x[v], weights=x.w),
                                    include_groups=False) for v in VARS})


def add_features(idx, tag):
    """Lags, rolling totals and anomalies on a complete epi-week grid."""
    idx = idx.sort_index()
    out = pd.DataFrame(index=idx.index)
    r = idx["rain_sum"]
    out[f"{tag}_rain"] = r
    for L in (1, 2, 3):
        out[f"{tag}_rain_lag{L}"] = r.shift(L)
    out[f"{tag}_rain_roll3"] = r.rolling(3).sum()
    out[f"{tag}_max_daily"] = idx["rain_max_daily"]
    out[f"{tag}_humidity"] = idx["humidity_pct_mean"]
    out[f"{tag}_temp"] = idx["temp_c_mean"]

    wk = idx.index.get_level_values("week")
    yr = idx.index.get_level_values("year")
    in_clim = (yr >= CLIM_YEARS[0]) & (yr <= CLIM_YEARS[1])
    # Smooth the 3-year climatology over +-1 week: 3 years is a small sample.
    clim = r[in_clim].groupby(wk[in_clim]).mean()
    clim = (pd.concat([clim.iloc[-1:], clim, clim.iloc[:1]])
              .rolling(3, center=True).mean().iloc[1:-1])
    anom = r - np.asarray(clim.reindex(wk).values)
    out[f"{tag}_rain_anom"] = anom
    for L in (1, 2, 3):
        out[f"{tag}_rain_anom_lag{L}"] = anom.shift(L)
    out[f"{tag}_rain_anom_roll3"] = anom.rolling(3).sum()
    return out


def main():
    df = pd.read_csv(SRC)
    print(f"[load] {len(df)} rows, {df.district.nunique()} districts, "
          f"{df.cell_id.nunique()} grid cells")

    ktm_cell = df.loc[df.district == "KATHMANDU", "cell_id"].iloc[0]
    indices = {
        "ktm": cell_mean(df[df.cell_id == ktm_cell]),
        "nat": cell_mean(df),
        "terai": cell_mean(df[df.is_terai == 1]),
        "casew": weighted_mean(df, case_weights(df.district.unique())),
    }

    # Sanity: our Kathmandu cell must reproduce the project's original fetch.
    orig = pd.read_csv(KTM_ORIG).set_index(["year", "week"])["nasa_rain_sum"]
    j = indices["ktm"]["rain_sum"].to_frame().join(orig, how="inner")
    diff = (j.rain_sum - j.nasa_rain_sum).abs().max()
    print(f"[check] Kathmandu cell vs original nasa_power_weekly.csv: "
          f"{len(j)} weeks, max abs diff {diff:.3f} mm "
          f"({'OK' if diff < 0.5 else 'MISMATCH'})")

    feats = pd.concat([add_features(v, k) for k, v in indices.items()],
                      axis=1).reset_index()
    feats.to_csv(OUT, index=False)
    print(f"[save] {feats.shape} -> {OUT}")

    # ------------------------------------------------ representativeness
    ktm = feats.set_index(["year", "week"])
    prov = {p: add_features(cell_mean(df[df.province == p]), "p")
            for p in df.province.unique()}
    rows = []

    def compare(name, rain, anom):
        ok = rain.notna() & ktm["ktm_rain"].notna()
        return {"region": name,
                "annual_mm_mean": round(rain.groupby(level="year").sum()
                                        .mean(), 0),
                "peak_week": int(rain.groupby(level="week").mean().idxmax()),
                "corr_with_ktm_weekly": round(
                    rain[ok].corr(ktm["ktm_rain"][ok]), 2),
                "corr_with_ktm_anomaly": round(
                    anom.corr(ktm["ktm_rain_anom"]), 2)}

    for tag, label in [("ktm", "Kathmandu cell (used so far)"),
                       ("nat", "All Nepal (39 cells)"),
                       ("terai", "Terai / southern plains"),
                       ("casew", "Case-weighted districts")]:
        rows.append(compare(label, ktm[f"{tag}_rain"],
                            ktm[f"{tag}_rain_anom"]))
    for p, f in prov.items():
        rows.append(compare(f"Province: {p}", f["p_rain"], f["p_rain_anom"]))
    rep = pd.DataFrame(rows)
    rep.to_csv(REP, index=False)
    print("\n[representativeness] 2021-2025")
    print(rep.to_string(index=False))

    # ------------------------------------------------ link to AGE cases
    age = pd.read_csv(EWARS)[["year", "week", "age_this_week"]]
    # Remove the reporting-growth trend: each week as a share of its year mean.
    age["age_rel"] = age.age_this_week / age.groupby("year").age_this_week \
        .transform("mean")
    # Remove the average seasonal shape too, leaving "more or fewer cases than
    # normal for this week" -- the part rainfall would need to explain.
    age["age_rel_anom"] = age.age_rel - age.groupby("week").age_rel \
        .transform("mean")
    grid = ktm.reset_index()[["year", "week"]]
    a = grid.merge(age, on=["year", "week"], how="left")
    crows = []
    for tag, label in [("ktm", "Kathmandu cell"), ("nat", "All Nepal"),
                       ("terai", "Terai"), ("casew", "Case-weighted")]:
        r = ktm[f"{tag}_rain"].values
        ra = ktm[f"{tag}_rain_anom"].values
        row = {"index": label}
        for L in range(0, 5):
            # rain L weeks BEFORE the case week
            rs = pd.Series(r).shift(L)
            ras = pd.Series(ra).shift(L)
            row[f"raw_lag{L}"] = round(a.age_rel.corr(rs), 3)
            row[f"anom_lag{L}"] = round(a.age_rel_anom.corr(ras), 3)
        crows.append(row)
    cor = pd.DataFrame(crows)
    cor.to_csv(COR, index=False)
    print("\n[AGE link] raw = cases vs rain (trend removed); "
          "anom = unusual cases vs unusual rain")
    print(cor.to_string(index=False))


if __name__ == "__main__":
    main()
