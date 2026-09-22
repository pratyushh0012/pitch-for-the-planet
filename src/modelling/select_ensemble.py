"""
Phase 5b -- Choose and build the deployed forecaster: a top-k ensemble.

WHY AN ENSEMBLE AT ALL. Selecting a single "best" model turned out to be
unreliable on this dataset, and that was established from pre-2024 data alone:
the CV folds barely agree on which candidate is best (Kendall tau between fold
rankings is only +0.07 to +0.22). With three years of weekly data there is not
enough signal to identify a single winner, so any single pick is largely luck.
Averaging several good candidates is the standard remedy -- it cannot pick the
best model, but it reliably avoids picking a bad one.

HOW k IS CHOSEN -- without touching the test years. Leave-one-fold-out inside
2021-2023: for each CV fold f, candidates are ranked by their error on the
OTHER folds only, the top-k of that ranking are averaged, and the ensemble is
scored on fold f. The k with the best mean score wins. Fold f never
contributes to its own ranking, so k is selected honestly.

Outputs:
  models/ensemble_h{1..4}.pkl   the k refitted members + metadata
  reports/metrics/ensemble_selection.csv
"""

import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.modelling.train_model import (  # noqa: E402
    CORE, TRAIN_END, TEST_YEARS, CV_SPLITS, CV_TEST, model_specs, fit_one,
    predict_rows, anchor_of, make_y, metrics, rolling_origin)

KS = [1, 3, 5, 10, 15, 20, 30]
POOL = 40          # candidates considered, ranked by overall CV
HORIZONS = [1, 2, 3, 4]


def fold_predictions(d_pre, fc, cands):
    """Per-candidate predictions and errors on each CV fold (pre-2024 only)."""
    folds = list(TimeSeriesSplit(n_splits=CV_SPLITS,
                                 test_size=CV_TEST).split(d_pre))
    P, E, Y = [], [], []
    for i_tr, i_cv in folds:
        tr, cv = d_pre.iloc[i_tr], d_pre.iloc[i_cv]
        preds, errs = [], []
        keep_common = np.ones(len(cv), bool)
        for c in cands:
            m = fit_one(c["build"], c["params"], tr, fc, c["framing"],
                        c["scheme"])
            if m is None:
                preds.append(np.full(len(cv), np.nan))
                errs.append(np.nan)
                continue
            p = predict_rows(m, cv, fc, c["framing"], c["blend_w"])
            preds.append(p)
            ok = np.isfinite(p)
            errs.append(mean_absolute_error(cv["target"].values[ok], p[ok])
                        if ok.sum() else np.nan)
        P.append(np.vstack(preds))
        E.append(np.array(errs))
        Y.append(cv["target"].values)
    return P, np.vstack(E), Y      # E: folds x candidates


def choose_k(P, E, Y):
    """Leave-one-fold-out: rank on other folds, score the top-k on this one."""
    n_folds = len(P)
    rows = []
    for k in KS:
        scores = []
        for f in range(n_folds):
            others = [g for g in range(n_folds) if g != f]
            rank = np.nanmean(E[others, :], axis=0)
            order = np.argsort(np.where(np.isnan(rank), np.inf, rank))
            pick = order[:k]
            ens = np.nanmean(P[f][pick, :], axis=0)
            ok = np.isfinite(ens)
            if ok.sum() == 0:
                continue
            scores.append(mean_absolute_error(Y[f][ok], ens[ok]))
        if scores:
            rows.append({"k": k, "loo_cv_MAE": float(np.mean(scores))})
    return pd.DataFrame(rows)


def main():
    cmp = pd.read_csv("reports/metrics/model_comparison.csv")
    specs = model_specs()
    summary = []

    for h in HORIZONS:
        d = pd.read_csv(f"data/processed/features_h{h}.csv").reset_index(
            drop=True)
        fc = [c for c in CORE if c in d.columns]
        d_pre = d[d.target_year <= TRAIN_END].reset_index(drop=True)
        ev = d[(d.target_year >= TEST_YEARS[0])
               & d["age_target_week_last_year"].notna()
               & d["age_roll4_mean"].notna()].index.to_numpy()
        ev_d = d.loc[ev]
        y = ev_d["target"].values
        y24 = (ev_d.target_year == 2024).values
        y25 = (ev_d.target_year == 2025).values

        cc = (cmp[(cmp.horizon == h) & (cmp.framing != "-")]
              .sort_values("cv_MAE").head(POOL))
        cands = [{"name": r.model, "build": specs[r.model][0],
                  "params": json.loads(r.params), "framing": r.framing,
                  "scheme": r.scheme, "blend_w": float(r.blend_w),
                  "cv": float(r.cv_MAE)} for _, r in cc.iterrows()]

        P, E, Y = fold_predictions(d_pre, fc, cands)
        ktab = choose_k(P, E, Y)
        best_k = int(ktab.loc[ktab.loo_cv_MAE.idxmin(), "k"])
        print(f"\n=== h={h} ===")
        print(ktab.round(2).to_string(index=False))
        print(f"  chosen k = {best_k}  (leave-one-fold-out, pre-2024 only)")

        members = cands[:best_k]
        # Honest out-of-sample score for the chosen ensemble
        preds = [rolling_origin(d, fc, c["build"], c["params"], c["framing"],
                                c["scheme"], c["blend_w"], ev)
                 for c in members]
        ens = np.nanmean(np.vstack(preds), axis=0)
        ok = np.isfinite(ens)
        mm = metrics(y[ok], ens[ok])
        m24 = mean_absolute_error(y[y24 & ok], ens[y24 & ok])
        m25 = mean_absolute_error(y[y25 & ok], ens[y25 & ok])

        bases = cmp[(cmp.horizon == h) & (cmp.framing == "-")]
        bb = bases.loc[bases.test_MAE.idxmin()]
        imp = 100 * (bb.test_MAE - mm["MAE"]) / bb.test_MAE
        print(f"  TEST MAE {mm['MAE']:.1f} (2024 {m24:.1f} | 2025 {m25:.1f}) "
              f"vs best baseline {bb.model.replace('Baseline: ','')} "
              f"{bb.test_MAE:.1f} -> {imp:+.1f}%")

        # Refit members on all available data and save
        full = d.copy()
        fitted = []
        for c in members:
            m = fit_one(c["build"], c["params"], full, fc, c["framing"],
                        c["scheme"])
            if m is not None:
                fitted.append({"model": m, "framing": c["framing"],
                               "scheme": c["scheme"], "blend_w": c["blend_w"],
                               "name": c["name"], "params": c["params"]})
        joblib.dump({"members": fitted, "features": fc, "horizon": h, "k": best_k,
                     "test_MAE": mm["MAE"], "test_RMSE": mm["RMSE"],
                     "test_MAPE": mm["MAPE"], "test2024_MAE": float(m24),
                     "test2025_MAE": float(m25),
                     "best_baseline": bb.model,
                     "best_baseline_MAE": float(bb.test_MAE),
                     "improvement_pct": float(imp),
                     "loo_cv_MAE": float(ktab.loo_cv_MAE.min())},
                    f"models/ensemble_h{h}.pkl")

        bt = pd.read_csv(f"reports/metrics/backtest_h{h}.csv")
        bt["pred_single"] = bt["pred"]
        bt["pred"] = ens
        bt["champion"] = f"Ensemble top-{best_k}"
        bt.to_csv(f"reports/metrics/backtest_h{h}.csv", index=False)

        summary.append({"horizon": h, "k": best_k,
                        "members": "; ".join(
                            f"{c['name']}|{c['framing']}|{c['scheme']}"
                            for c in members[:6])
                        + ("; ..." if best_k > 6 else ""),
                        "loo_cv_MAE": round(ktab.loo_cv_MAE.min(), 1),
                        "test_MAE": round(mm["MAE"], 1),
                        "test_RMSE": round(mm["RMSE"], 1),
                        "test_MAPE": round(mm["MAPE"], 1),
                        "test2024_MAE": round(m24, 1),
                        "test2025_MAE": round(m25, 1),
                        "best_baseline": bb.model.replace("Baseline: ", ""),
                        "best_baseline_MAE": round(bb.test_MAE, 1),
                        "improvement_pct": round(imp, 1)})

    s = pd.DataFrame(summary)
    s.to_csv("reports/metrics/ensemble_selection.csv", index=False)
    print("\n" + "=" * 78)
    print(s[["horizon", "k", "test_MAE", "test2024_MAE", "test2025_MAE",
             "best_baseline", "best_baseline_MAE",
             "improvement_pct"]].to_string(index=False))
    print("[save] models/ensemble_h*.pkl, reports/metrics/ensemble_selection.csv")


if __name__ == "__main__":
    main()
