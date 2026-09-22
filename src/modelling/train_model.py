"""
Phase 5 -- Model training, framing comparison and selection.

=============================================================== THE CORE PROBLEM
The AGE series is strongly NON-STATIONARY. Weekly mean rises 116 (2021) -> 240
-> 335 -> 433 -> 495 (2025), and the year-on-year growth ratio DECAYS the whole
way: x2.06, x1.40, x1.29, x1.14. Most of that rise is very likely EWARS
expanding its reporting-site network, not a 4x real explosion in disease.

Two consequences drive every design choice here:
  1. Trees cannot extrapolate. A tree trained on 2021-2023 (max 592 cases) can
     never output 2025's 829-case peak, whatever the inputs say.
  2. A model frozen on the high-growth era memorises a growth factor that is
     already wrong by 2025.

=================================================================== THE SPLIT
  TRAIN       2021-2023   model fitting
  VALIDATION  TimeSeriesSplit folds INSIDE 2021-2023
              -- every choice (model family, target framing, training scheme,
                 blend weight, hyperparameters) is made here and ONLY here
  TEST        2024-2025   rolling origin, never consulted for any choice

This ordering matters. An earlier version of this script selected on 2024 and
tested on 2025, and it produced badly misleading numbers: 2024 (+29% YoY) and
2025 (+14% YoY) reward OPPOSITE forecasters, so tuning on one actively
anti-selects for the other. Choosing on pre-2024 only, then reporting the whole
of 2024-2025 as one untouched out-of-sample period, is the honest design.
Per-year test results are reported separately so the instability stays visible.

============================================================== TARGET FRAMINGS
Each predicts a quantity, then reconstructs a case count via an ANCHOR:
  level        y = age_{t+h}                         anchor 1 (direct)
  logratio     y = log(age_{t+h} / age_t)            anchor = current level
  ratio_ly     y = log(age_{t+h} / age_lastyear)     anchor = same week last yr
  ratio_blend  y = log(age_{t+h} / A), A = sqrt(age_t * age_lastyear)
               -- geometric mean: respects both the current epidemic level and
                  the position in the season.
Anchored framings make the learning target near-stationary AND let a forecast
exceed the training range, which defuses both consequences above.

============================================================= TRAINING SCHEMES
Because the series is non-stationary, how much history to keep is itself a
hyperparameter, chosen by CV like any other:
  expand   all history to date          decay   exponential recency weights
  window   most recent 104 weeks only

===================================================================== BLENDING
Forecast combination is a standard variance-reduction device. Blend weight w
(prediction = (1-w)*model + w*seasonal_naive) is searched over {0, 0.25, 0.5}
by the same CV, so a combination is adopted only if pre-2024 evidence supports
it -- never because it flattered the test years.
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

HORIZONS = [1, 2, 3, 4]
TRAIN_END = 2023          # selection may see target years <= this
TEST_YEARS = (2024, 2025)
FRAMINGS = ["level", "logratio", "ratio_ly", "ratio_blend"]
SCHEMES = ["expand", "decay", "window"]
BLEND_W = [0.0, 0.25, 0.5]
WINDOW, HALFLIFE, MIN_TRAIN = 104, 40, 40
CV_SPLITS, CV_TEST = 3, 25
DROP = ["year", "week", "date", "horizon", "target_year", "target_week",
        "target"]

# A compact, mechanism-led feature set. 39 raw columns on ~140 training rows
# overfits; these 20 keep one representative of each signal family.
CORE = ["age_now", "age_lag1", "age_lag2", "age_roll4_mean", "age_diff1",
        "age_target_week_last_year", "age_yoy_ratio", "age_now_last_year",
        "rain_mean_mm", "rain_mean_lag1", "rain_mean_lag2", "rain_mean_lag3",
        "rain_mean_roll3", "max_daily_mm", "humidity_pct_mean", "temp_c_mean",
        "tgt_sin", "tgt_cos", "tgt_is_monsoon", "tgt_week_of_year"]


# ------------------------------------------------------------ framing algebra
def anchor_of(d, fr):
    if fr == "level":
        return np.ones(len(d))
    if fr == "logratio":
        return d["age_now"].values.astype(float)
    if fr == "ratio_ly":
        return d["age_target_week_last_year"].values.astype(float)
    return np.sqrt(d["age_now"].values.astype(float)
                   * d["age_target_week_last_year"].values.astype(float))


def make_y(d, fr):
    if fr == "level":
        return d["target"].values.astype(float)
    return np.log(d["target"].values.astype(float) / anchor_of(d, fr))


def to_level(p, fr, anchor):
    if fr == "level":
        return np.clip(p, 0, None)
    return np.clip(anchor * np.exp(np.clip(p, -2.5, 2.5)), 0, None)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE": float(np.mean(np.abs((y - p) / y)) * 100)}


# -------------------------------------------------------------------- models
def model_specs():
    return {
        "XGBoost": (
            lambda p: xgb.XGBRegressor(objective="reg:squarederror",
                                       random_state=42, n_jobs=2, verbosity=0,
                                       **p),
            [{"max_depth": d, "n_estimators": 300, "learning_rate": lr,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 3, "reg_lambda": 3.0}
             for d in (2, 3) for lr in (0.03, 0.05)]),
        "LightGBM": (
            lambda p: lgb.LGBMRegressor(random_state=42, n_jobs=2, verbose=-1,
                                        **p),
            [{"max_depth": d, "n_estimators": 300, "learning_rate": lr,
              "num_leaves": 7, "min_child_samples": 10, "subsample": 0.8,
              "colsample_bytree": 0.8}
             for d in (2, 3) for lr in (0.03, 0.05)]),
        "RandomForest": (
            lambda p: RandomForestRegressor(n_estimators=300, random_state=42,
                                            n_jobs=2, **p),
            [{"max_depth": d, "min_samples_leaf": l}
             for d in (4, 6) for l in (2, 4)]),
        "ExtraTrees": (
            lambda p: ExtraTreesRegressor(n_estimators=300, random_state=42,
                                          n_jobs=2, **p),
            [{"max_depth": d, "min_samples_leaf": l}
             for d in (6, None) for l in (2, 4)]),
        "Ridge": (
            lambda p: make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler(), Ridge(**p)),
            [{"alpha": a} for a in (10.0, 50.0, 200.0)]),
    }


def fit_one(build, params, tr, fc, fr, scheme):
    """Fit with the chosen history scheme. Returns None if too little data."""
    tr = tr[np.isfinite(make_y(tr, fr))]
    if scheme == "window":
        tr = tr.iloc[-WINDOW:]
    if len(tr) < MIN_TRAIN:
        return None
    m = build(params)
    y = make_y(tr, fr)
    w = None
    if scheme == "decay":
        w = 0.5 ** (np.arange(len(tr))[::-1] / HALFLIFE)
    try:
        if w is None:
            m.fit(tr[fc], y)
        elif hasattr(m, "steps"):
            m.fit(tr[fc], y, ridge__sample_weight=w)
        else:
            m.fit(tr[fc], y, sample_weight=w)
    except Exception:
        return None
    return m


def predict_rows(m, rows, fc, fr, w_blend):
    p = to_level(m.predict(rows[fc]), fr, anchor_of(rows, fr))
    if w_blend > 0:
        sn = rows["age_target_week_last_year"].values.astype(float)
        p = np.where(np.isfinite(sn), (1 - w_blend) * p + w_blend * sn, p)
    return p


# ----------------------------------------------------- VALIDATION (pre-2024)
def cv_score(d_pre, fc, build, params, fr, scheme, w_blend):
    """TimeSeriesSplit CV strictly inside the 2021-2023 training era.

    Early folds are skipped rather than failed: the seasonal-anchor framings
    have no usable rows in 2021 (no previous year to compare against), so their
    first fold legitimately cannot be fitted. At least 2 scored folds are
    required for a candidate to be eligible at all.
    """
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_TEST)
    errs = []
    for i_tr, i_cv in tscv.split(d_pre):
        tr, cv = d_pre.iloc[i_tr], d_pre.iloc[i_cv]
        m = fit_one(build, params, tr, fc, fr, scheme)
        if m is None:
            continue
        anc = anchor_of(cv, fr)
        keep = np.isfinite(anc) & (anc > 0)
        if w_blend > 0:
            keep &= cv["age_target_week_last_year"].notna().values
        if keep.sum() == 0:
            continue
        cvk = cv[keep]
        p = predict_rows(m, cvk, fc, fr, w_blend)
        if not np.all(np.isfinite(p)):
            continue
        errs.append(mean_absolute_error(cvk["target"].values, p))
    return float(np.mean(errs)) if len(errs) >= 2 else np.inf


# --------------------------------------------------------- TEST (2024-2025)
def rolling_origin(d, fc, build, params, fr, scheme, w_blend, ev):
    out = np.full(len(ev), np.nan)
    for k, i in enumerate(ev):
        m = fit_one(build, params, d.iloc[:i], fc, fr, scheme)
        if m is None:
            continue
        out[k] = predict_rows(m, d.iloc[[i]], fc, fr, w_blend)[0]
    return out


def frozen(d, fc, build, params, fr, scheme, w_blend, ev):
    """Contrast case: train once on 2021-2023 and never update again."""
    m = fit_one(build, params, d[d.target_year <= TRAIN_END], fc, fr, scheme)
    if m is None:
        return np.full(len(ev), np.nan)
    return predict_rows(m, d.loc[ev], fc, fr, w_blend)


# ------------------------------------------------------------------- driver
def run_horizon(h, log):
    d = pd.read_csv(f"data/processed/features_h{h}.csv").reset_index(drop=True)
    fc = [c for c in CORE if c in d.columns]

    # Full 2021-2023 training era. Rows without a seasonal anchor are kept --
    # the level/logratio framings can use them; the ratio framings drop them
    # internally in fit_one.
    d_pre = d[d.target_year <= TRAIN_END].reset_index(drop=True)

    ev = d[(d.target_year >= TEST_YEARS[0])
           & d["age_target_week_last_year"].notna()
           & d["age_roll4_mean"].notna()].index.to_numpy()
    ev_d = d.loc[ev]
    y_ev = ev_d["target"].values
    y24 = (ev_d.target_year == 2024).values
    y25 = (ev_d.target_year == 2025).values

    log.append(f"\n{'=' * 78}\nHORIZON {h} WEEK(S) AHEAD\n{'=' * 78}")
    log.append(f"features {len(fc)} | selection pool (target year <= "
               f"{TRAIN_END}): {len(d_pre)} rows, CV = {CV_SPLITS} "
               f"TimeSeriesSplit folds of {CV_TEST} weeks")
    log.append(f"TEST rolling-origin weeks: {len(ev)} "
               f"({y24.sum()} in 2024, {y25.sum()} in 2025)")
    log.append(f"training-era target range {d_pre.target.min():.0f}-"
               f"{d_pre.target.max():.0f}  vs  test range "
               f"{y_ev.min():.0f}-{y_ev.max():.0f}"
               "   <- trees cannot exceed the training max")

    rows = {}

    # ---------------- baselines on the identical rows ----------------
    base = {
        "Baseline: persistence": ev_d["age_now"].values,
        "Baseline: seasonal naive":
            ev_d["age_target_week_last_year"].values,
        "Baseline: seasonal + drift":
            (ev_d["age_target_week_last_year"]
             * ev_d["age_yoy_ratio"].fillna(1.0)).values,
        "Baseline: 4-week rolling mean": ev_d["age_roll4_mean"].values,
    }
    recs = []
    for name, p in base.items():
        r = {"model": name, "framing": "-", "scheme": "-", "blend_w": "-",
             "params": "-", "cv_MAE": np.nan}
        r.update({f"test_{k}": v for k, v in metrics(y_ev, p).items()})
        r.update({f"test2024_{k}": v for k, v in
                  metrics(y_ev[y24], p[y24]).items()})
        r.update({f"test2025_{k}": v for k, v in
                  metrics(y_ev[y25], p[y25]).items()})
        recs.append(r)
        rows[name] = p

    # ---------------- search: selection uses cv_MAE ONLY ----------------
    specs = model_specs()
    cands, t0 = [], time.time()
    for name, (build, grid) in specs.items():
        for fr in FRAMINGS:
            for scheme in SCHEMES:
                for wb in BLEND_W:
                    best = (np.inf, None)
                    for params in grid:
                        s = cv_score(d_pre, fc, build, params, fr, scheme, wb)
                        if s < best[0]:
                            best = (s, params)
                    if best[1] is not None and np.isfinite(best[0]):
                        cands.append({"name": name, "build": build,
                                      "params": best[1], "framing": fr,
                                      "scheme": scheme, "blend_w": wb,
                                      "cv": best[0]})
        print(f"   h={h} {name} searched ({time.time() - t0:.0f}s)")

    # ---------------- score every candidate on TEST (report only) --------
    for c in cands:
        p = rolling_origin(d, fc, c["build"], c["params"], c["framing"],
                           c["scheme"], c["blend_w"], ev)
        ok = np.isfinite(p)
        if ok.sum() < len(ev) * 0.8:
            continue
        r = {"model": c["name"], "framing": c["framing"],
             "scheme": c["scheme"], "blend_w": c["blend_w"],
             "params": json.dumps(c["params"]), "cv_MAE": c["cv"]}
        r.update({f"test_{k}": v for k, v in
                  metrics(y_ev[ok], p[ok]).items()})
        r.update({f"test2024_{k}": v for k, v in
                  metrics(y_ev[y24 & ok], p[y24 & ok]).items()})
        r.update({f"test2025_{k}": v for k, v in
                  metrics(y_ev[y25 & ok], p[y25 & ok]).items()})
        fz = frozen(d, fc, c["build"], c["params"], c["framing"],
                    c["scheme"], c["blend_w"], ev)
        if np.all(np.isfinite(fz)):
            r["frozen_test_MAE"] = metrics(y_ev, fz)["MAE"]
        recs.append(r)
        rows[f"{c['name']}|{c['framing']}|{c['scheme']}|w{c['blend_w']}"] = p
        c["key"] = f"{c['name']}|{c['framing']}|{c['scheme']}|w{c['blend_w']}"

    res = pd.DataFrame(recs)
    res.insert(0, "horizon", h)

    # ---------------- THE selection: lowest CV MAE, pre-2024 only -------
    champ = min(cands, key=lambda c: c["cv"])
    ck = champ["key"]
    crow = res[(res.model == champ["name"]) & (res.framing == champ["framing"])
               & (res.scheme == champ["scheme"])
               & (res.blend_w == champ["blend_w"])].iloc[0]
    bb = res[res.framing == "-"].sort_values("test_MAE").iloc[0]
    imp = 100 * (bb.test_MAE - crow.test_MAE) / bb.test_MAE

    log.append(f"\nSELECTED by pre-2024 CV: {champ['name']} "
               f"[{champ['framing']}] scheme={champ['scheme']} "
               f"blend_w={champ['blend_w']}  (CV MAE {champ['cv']:.1f})")
    log.append(f"  params {json.dumps(champ['params'])}")
    log.append(f"  TEST 2024-2025 MAE {crow.test_MAE:.1f} "
               f"(2024 {crow.test2024_MAE:.1f} | 2025 {crow.test2025_MAE:.1f})")
    log.append(f"  best baseline: {bb.model} MAE {bb.test_MAE:.1f} "
               f"(2024 {bb.test2024_MAE:.1f} | 2025 {bb.test2025_MAE:.1f})")
    log.append(f"  >> improvement over best baseline: {imp:+.1f}%")

    show = ["model", "framing", "scheme", "blend_w", "cv_MAE", "test_MAE",
            "test_RMSE", "test_MAPE", "test2024_MAE", "test2025_MAE",
            "frozen_test_MAE"]
    top = res.sort_values("test_MAE").head(18)
    log.append("\nTop 18 by out-of-sample MAE (selection did NOT use these):")
    log.append(top[show].round(2).to_string(index=False))

    # ---------------- artefacts ----------------
    bt = ev_d[["year", "week", "date", "target_year", "target_week", "target",
               "age_now", "age_target_week_last_year", "age_roll4_mean",
               "rain_mean_mm", "rain_source_is_dhm"]].copy()
    bt["horizon"] = h
    bt["pred"] = rows[ck]
    bt["champion"] = ck
    for bn, bp in base.items():
        bt[bn.replace("Baseline: ", "base_").replace(" ", "_")
           .replace("-", "").replace("+", "plus")] = bp
    bt.to_csv(f"reports/metrics/backtest_h{h}.csv", index=False)

    full = d[np.isfinite(make_y(d, champ["framing"]))]
    m = fit_one(champ["build"], champ["params"], full, fc, champ["framing"],
                champ["scheme"])
    joblib.dump({"model": m, "features": fc, "framing": champ["framing"],
                 "scheme": champ["scheme"], "blend_w": champ["blend_w"],
                 "name": champ["name"], "params": champ["params"],
                 "horizon": h, "n_train": len(full),
                 "cv_MAE": champ["cv"],
                 "test_MAE": float(crow.test_MAE),
                 "test_RMSE": float(crow.test_RMSE),
                 "test_MAPE": float(crow.test_MAPE),
                 "test2024_MAE": float(crow.test2024_MAE),
                 "test2025_MAE": float(crow.test2025_MAE),
                 "best_baseline": bb.model,
                 "best_baseline_MAE": float(bb.test_MAE),
                 "improvement_pct": float(imp)},
                f"models/model_h{h}.pkl")
    return res, ck, float(imp), float(crow.test_MAE), float(bb.test_MAE)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    log, allr, champs = [], [], {}
    for h in HORIZONS:
        r, ck, imp, mae, bmae = run_horizon(h, log)
        allr.append(r)
        champs[h] = {"champion": ck, "test_MAE": round(mae, 1),
                     "best_baseline_MAE": round(bmae, 1),
                     "improvement_pct": round(imp, 1)}
    pd.concat(allr, ignore_index=True).to_csv(
        "reports/metrics/model_comparison.csv", index=False)

    log.append(f"\n{'=' * 78}\nCHAMPIONS  (chosen on pre-2024 CV; scored on "
               f"untouched 2024-2025)\n{'=' * 78}")
    for h, c in champs.items():
        log.append(f"  h={h}: {c['champion']:34s} MAE {c['test_MAE']:6.1f} "
                   f"vs baseline {c['best_baseline_MAE']:6.1f}  "
                   f"({c['improvement_pct']:+.1f}%)")
    txt = "\n".join(log)
    print(txt)
    open("reports/metrics/training_log.txt", "w").write(txt)
    json.dump(champs, open("models/champions.json", "w"), indent=2)


if __name__ == "__main__":
    main()
