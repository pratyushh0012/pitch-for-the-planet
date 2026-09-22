"""
Is single-model selection reliable here, and does averaging fix it?

Motivation comes from pre-2024 evidence alone: the CV fold-to-fold ranking of
candidates is measured first (Kendall tau between folds). If the ranking is
unstable across folds, no single-pick rule can be trusted, and averaging the
top-k is the standard remedy.

Only after that is the top-k ensemble scored on the 2024-2025 hold-out.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.modelling.train_model import (  # noqa: E402
    CORE, TRAIN_END, TEST_YEARS, CV_SPLITS, CV_TEST, model_specs, fit_one,
    predict_rows, anchor_of, make_y, metrics, FRAMINGS, SCHEMES, BLEND_W,
    rolling_origin)
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

KS = [1, 3, 5, 10, 20]


def fold_stability(d_pre, fc, cands):
    """Per-fold CV error for each candidate -> rank agreement between folds.
    Uses pre-2024 data ONLY."""
    tscv = list(TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_TEST)
                .split(d_pre))
    per_fold = []
    for i_tr, i_cv in tscv:
        tr, cv = d_pre.iloc[i_tr], d_pre.iloc[i_cv]
        col = []
        for c in cands:
            m = fit_one(c["build"], c["params"], tr, fc, c["framing"],
                        c["scheme"])
            if m is None:
                col.append(np.nan)
                continue
            anc = anchor_of(cv, c["framing"])
            keep = np.isfinite(anc) & (anc > 0)
            if c["blend_w"] > 0:
                keep &= cv["age_target_week_last_year"].notna().values
            if keep.sum() == 0:
                col.append(np.nan)
                continue
            p = predict_rows(m, cv[keep], fc, c["framing"], c["blend_w"])
            col.append(mean_absolute_error(cv[keep]["target"].values, p))
        per_fold.append(col)
    F = pd.DataFrame(per_fold).T.dropna()
    taus = []
    for i in range(F.shape[1]):
        for j in range(i + 1, F.shape[1]):
            taus.append(kendalltau(F[i], F[j]).statistic)
    return float(np.nanmean(taus)), F


def main():
    out = {}
    for h in [1, 2, 3, 4]:
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

        cmp = pd.read_csv("reports/metrics/model_comparison.csv")
        cc = cmp[(cmp.horizon == h) & (cmp.framing != "-")].copy()
        cc = cc.sort_values("cv_MAE")
        specs = model_specs()
        cands = []
        for _, r in cc.iterrows():
            cands.append({"name": r.model, "build": specs[r.model][0],
                          "params": json.loads(r.params), "framing": r.framing,
                          "scheme": r.scheme, "blend_w": float(r.blend_w),
                          "cv": r.cv_MAE})

        tau, _ = fold_stability(d_pre, fc, cands[:30])
        print(f"\n=== h={h} ===")
        print(f"  CV fold-to-fold rank agreement (Kendall tau, top-30 by CV, "
              f"pre-2024 only): {tau:+.2f}")
        print("  tau near 0 => the folds disagree about which model is best "
              "=> single-pick selection is unreliable")

        preds = {}
        for k in range(max(KS)):
            c = cands[k]
            preds[k] = rolling_origin(d, fc, c["build"], c["params"],
                                      c["framing"], c["scheme"], c["blend_w"],
                                      ev)
        row = {"horizon": h, "fold_tau": round(tau, 3)}
        print(f"  {'rule':28s} {'MAE':>7s} {'2024':>7s} {'2025':>7s}")
        for k in KS:
            P = np.nanmean(np.vstack([preds[i] for i in range(k)]), axis=0)
            ok = np.isfinite(P)
            m_all = mean_absolute_error(y[ok], P[ok])
            m24 = mean_absolute_error(y[y24 & ok], P[y24 & ok])
            m25 = mean_absolute_error(y[y25 & ok], P[y25 & ok])
            tag = "single best-CV model" if k == 1 else f"mean of top-{k} by CV"
            print(f"  {tag:28s} {m_all:7.1f} {m24:7.1f} {m25:7.1f}")
            row[f"top{k}_MAE"] = round(m_all, 1)
            row[f"top{k}_MAE_2024"] = round(m24, 1)
            row[f"top{k}_MAE_2025"] = round(m25, 1)
        bases = cmp[(cmp.horizon == h) & (cmp.framing == "-")]
        bb = bases.loc[bases.test_MAE.idxmin()]
        row["best_baseline"] = bb.model.replace("Baseline: ", "")
        row["best_baseline_MAE"] = round(bb.test_MAE, 1)
        print(f"  {'best baseline (' + row['best_baseline'] + ')':28s} "
              f"{bb.test_MAE:7.1f} {bb.test2024_MAE:7.1f} "
              f"{bb.test2025_MAE:7.1f}")
        out[h] = row

    df = pd.DataFrame(out.values())
    df.to_csv("reports/metrics/ensemble_analysis.csv", index=False)
    print("\n[save] reports/metrics/ensemble_analysis.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
