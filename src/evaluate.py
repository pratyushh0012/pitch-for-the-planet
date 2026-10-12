#!/usr/bin/env python3
"""
evaluate.py -- Phase 5: rolling-origin back-test by season + ablation.

For each test season k (once >= 3 usable training seasons exist):
  * surge line, scaler, spline knots and the warm/wet cut-off are fitted on
    seasons < k only;
  * training rows whose target window (t+1 .. t+5) reaches into season k are
    dropped, so no test-season case count is ever a training label;
  * every model is run on feature Set A (no weather) and Set B (with weather).

Outputs:
  data/processed/backtest_predictions.csv   one row per (test week, model)
  reports/metrics_by_season.csv             one row per (season, model)
  reports/results.md, reports/figures/dengue/*.png
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import dengue_features as F  # noqa: E402
from src import models as M  # noqa: E402
from src import targets as T  # noqa: E402

TABLE = os.path.join(ROOT, "data/processed/weekly_table.csv")
PRED_OUT = os.path.join(ROOT, "data/processed/backtest_predictions.csv")
MET_OUT = os.path.join(ROOT, "reports/metrics_by_season.csv")
FIG = os.path.join(ROOT, "reports/figures/dengue")
GREEN, AMBER = 0.3, 0.6                 # traffic-light defaults (not tuned -- see results.md)
MIN_TRAIN = 3
SKILL_MARGIN = 0.05                     # complex model must beat simpler by > 0.05 skill


def usable_seasons(table):
    out = []
    for y, g in table.groupby("year"):
        have = set(g.loc[g.cases.notna(), "week"])
        if len(have & set(range(1, 53))) >= 40 and len(set(range(22, 49)) - have) <= 3:
            out.append(int(y))
    return out


def light(prob, p90, line4):
    if np.isnan(prob):
        return None
    if prob >= AMBER and p90 > 2 * line4:
        return "red"
    if prob >= GREEN:
        return "amber"
    return "green"


def fold_frame(table, train_years, method="meansd"):
    """Targets + features for one fold (everything learnt from train_years only)."""
    line = T.surge_line(table, train_years, method)
    t = T.add_targets(T.add_line(table, line))
    feats = F.build_features(table, warm_c=F.warm_cutoff(table, train_years))
    t = t.merge(feats[["year", "week"] + F.SET_B], on=["year", "week"])
    t["line_t4"] = t.line.shift(-T.HORIZON)
    t.loc[t.line_t4.isna(), "line_t4"] = line.set_index("woy").line.reindex(
        (T.woy(t.week[t.line_t4.isna()]) + T.HORIZON - 1) % 52 + 1).values
    return t, line


def train_rows(t, train_years, feats):
    last = t.index[t.year == max(train_years)].max()
    ok = t.year.isin(train_years) & (t.index + t.target_end <= last)
    ok &= t[feats].notna().all(axis=1) & t.y_reg.notna() & t.y_cls.notna()
    return t[ok]


def run_folds(table, method="meansd"):
    seasons = usable_seasons(table)
    preds, folds = [], []
    for i, k in enumerate(seasons):
        train_years = seasons[:i]
        if len(train_years) < MIN_TRAIN:
            continue
        t, line = fold_frame(table, train_years, method)
        test = t[(t.year == k)]
        base = M.baseline_predictions(t, test.index)
        info = {"test": k, "train": train_years, "warm_cutoff": round(F.warm_cutoff(table, train_years), 2)}
        for name, p in base.items():
            preds.append(pd.DataFrame({"model": name, "set": "-", **{c: p[c] for c in p}}, index=test.index)
                         .join(test[["year", "week", "cases", "line", "line_t4", "above", "onset", "y_cls", "y_reg"]]))
        for key in M.MODEL_KEYS:
            mdl = M.make_model(key)
            tr = train_rows(t, train_years, mdl.feats + ([mdl.spline_col] if getattr(mdl, "spline_col", None) else []))
            mdl.fit(tr)
            te = test[test[mdl.feats].notna().all(axis=1)]
            p = mdl.predict(te)
            p["model"], p["set"] = key.split("_")[0], key.split("_")[1]
            preds.append(p.join(te[["year", "week", "cases", "line", "line_t4", "above", "onset", "y_cls", "y_reg"]]))
            info[f"{key}_n_train"] = len(tr)
            info[f"{key}_train_pos"] = int(tr.y_cls.sum())
            if hasattr(mdl, "family_used"):
                info[f"{key}_family"] = mdl.family_used
                info[f"{key}_alpha"] = round(mdl.alpha, 3) if np.isfinite(mdl.alpha) else None
        folds.append(info)
        print(f"[{method}] fold test={k} train={train_years} train positives={info['M1_A_train_pos']}", flush=True)
    P = pd.concat(preds).reset_index(names="row")
    P["line_method"] = method
    P["light"] = [light(p, q, l) if pd.notna(p) and pd.notna(q) else None
                  for p, q, l in zip(P.prob, P.p90, P.line_t4)]
    return P, folds


def event_metrics(g):
    """Surge-onset metrics for one season x model. g: that season's rows (sorted)."""
    g = g.sort_values("row")
    rows = g.set_index("row")
    alert = (rows.prob >= GREEN)
    onsets = rows.index[rows.onset == 1]
    caught, leads = 0, []
    for s in onsets:
        win = [r for r in range(s - T.HORIZON, s) if r in alert.index and alert[r]]
        if win:
            caught += 1
            leads.append(s - min(win))
    # false alarms: runs of alert weeks with no onset in t+1..t+4, outside an active surge
    false_wk = alert & (rows.y_cls == 0) & (rows.above != 1)
    runs = (false_wk & ~false_wk.shift(1, fill_value=False)).sum()
    return len(onsets), caught, (np.mean(leads) if leads else np.nan), int(runs)


def season_metrics(P):
    out = []
    for (k, model, fset), g in P.groupby(["year", "model", "set"]):
        r = {"season": k, "model": model, "set": fset}
        reg = g.dropna(subset=["y_reg", "mean"])
        r["mae"] = (reg.y_reg - reg["mean"]).abs().mean() if len(reg) else np.nan
        r["n_reg"] = len(reg)
        cls = g.dropna(subset=["y_cls", "prob"])
        if len(cls):
            r["brier"] = brier_score_loss(cls.y_cls, cls.prob.clip(0, 1))
            r["pr_auc"] = average_precision_score(cls.y_cls, cls.prob) if cls.y_cls.sum() > 0 else np.nan
            r["positives"] = int(cls.y_cls.sum())
            r["surges"], r["caught"], r["lead_time"], r["false_alarms"] = event_metrics(cls)
        if len(reg) and "p10" in g and g.p10.notna().any():
            r["coverage_80"] = ((reg.y_reg >= reg.p10) & (reg.y_reg <= reg.p90)).mean()
        out.append(r)
    S = pd.DataFrame(out)
    # skill on common rows: every model's MAE is compared on the same weeks as seasonal naive
    sk = []
    for k, g in P.groupby("year"):
        sn = g[g.model == "seasonal_naive"].set_index("row")["mean"]
        pe = g[g.model == "persistence"].set_index("row")["mean"]
        for (model, fset), h in g.groupby(["model", "set"]):
            h = h.set_index("row").dropna(subset=["mean", "y_reg"])
            common = h.index.intersection(sn.dropna().index).intersection(pe.dropna().index)
            if not len(common):
                continue
            mae = (h.y_reg[common] - h["mean"][common]).abs().mean()
            mae_sn = (h.y_reg[common] - sn[common]).abs().mean()
            mae_pe = (h.y_reg[common] - pe[common]).abs().mean()
            sk.append({"season": k, "model": model, "set": fset, "mae_common": mae,
                       "skill": 1 - mae / mae_sn, "skill_vs_persistence": 1 - mae / mae_pe})
    return S.merge(pd.DataFrame(sk), on=["season", "model", "set"], how="left")


def choose(S):
    """Simplest-first: move to a more complex model only if it beats the current
    choice by > SKILL_MARGIN skill in a majority of test seasons. Weather variants
    (Set B) count as more complex than Set A."""
    order = ["M1_A", "M1_B", "M2_B", "M3_A", "M3_B"]
    sk = S.assign(key=S.model + "_" + S.set).pivot(index="season", columns="key", values="skill")
    br = S.assign(key=S.model + "_" + S.set).pivot(index="season", columns="key", values="brier")
    cur, log = order[0], []
    for cand in order[1:]:
        d = sk[cand] - sk[cur]
        wins = int((d > SKILL_MARGIN).sum())
        verdict = wins > len(d) / 2
        log.append(f"{cand} vs {cur}: beats by > {SKILL_MARGIN} skill in {wins}/{len(d)} seasons "
                   f"(median Δ {d.median():+.3f}) → {'switch' if verdict else 'keep ' + cur}")
        if verdict:
            cur = cand
    return cur, log, sk, br


def choose_classifier(S):
    """Same simplest-first rule for the classifier, on Brier score: switch only if the
    more complex model's Brier is > 5% lower (relative) in a majority of seasons."""
    order = ["M1_A", "M1_B", "M2_B", "M3_A", "M3_B"]
    br = S.assign(key=S.model + "_" + S.set).pivot(index="season", columns="key", values="brier")
    cur, log = order[0], []
    for cand in order[1:]:
        rel = 1 - br[cand] / br[cur]
        wins = int((rel > SKILL_MARGIN).sum())
        verdict = wins > len(rel) / 2
        log.append(f"{cand} vs {cur}: Brier > 5% lower in {wins}/{len(rel)} seasons "
                   f"(median {rel.median():+.1%}) → {'switch' if verdict else 'keep ' + cur}")
        if verdict:
            cur = cand
    return cur, log


def ablation(S):
    rows = []
    for fam in ("M1", "M3"):
        a = S[(S.model == fam) & (S.set == "A")].set_index("season")
        b = S[(S.model == fam) & (S.set == "B")].set_index("season")
        for k in a.index.intersection(b.index):
            rows.append({"model": fam, "season": k, "skill_A": a.skill[k], "skill_B": b.skill[k],
                         "delta_skill": b.skill[k] - a.skill[k],
                         "brier_A": a.brier[k], "brier_B": b.brier[k],
                         "delta_brier_improvement": a.brier[k] - b.brier[k]})
    return pd.DataFrame(rows)


def fmt(x, d=2):
    return "" if pd.isna(x) else (f"{x:.{d}f}" if isinstance(x, (float, np.floating)) else str(x))


MIN_POS = 2      # a surge line is operational only if every training fold has >= 2 positives


def main():
    table = pd.read_csv(TABLE).sort_values(["year", "week"]).reset_index(drop=True)
    runs = {}
    for method in T.METHODS:
        P, folds = run_folds(table, method)
        S = season_metrics(P)
        S["line_method"] = method
        runs[method] = (P, S, folds)
    # rule fixed before comparing results: spec line unless it starves a fold of positives
    pos = {m: min(f["M1_A_train_pos"] for f in runs[m][2]) for m in runs}
    line_method = "meansd" if pos["meansd"] >= MIN_POS else "q3"
    line_note = (f"min training positives per fold: mean+2SD = {pos['meansd']}, Q3 = {pos['q3']} "
                 f"→ operational surge line: {line_method}")
    print(line_note)
    P, S, folds = runs[line_method]
    os.makedirs(os.path.dirname(PRED_OUT), exist_ok=True)
    pd.concat([r[0] for r in runs.values()]).to_csv(PRED_OUT, index=False)
    pd.concat([r[1] for r in runs.values()]).to_csv(MET_OUT, index=False)
    chosen, sel_log, sk, br = choose(S[S.model.isin(["M1", "M2", "M3"])])
    chosen_cls, cls_log = choose_classifier(S[S.model.isin(["M1", "M2", "M3"])])
    print("classifier:", chosen_cls, *cls_log, sep="\n  ")
    A = ablation(S)
    json.dump({"chosen": chosen, "chosen_cls": chosen_cls, "line_method": line_method, "line_note": line_note,
               "selection_log": sel_log, "cls_selection_log": cls_log, "folds": {m: r[2] for m, r in runs.items()}},
              open(os.path.join(ROOT, "reports/evaluation_summary.json"), "w"), indent=2, default=str)
    print("chosen:", chosen)
    print("\n".join(sel_log))
    print(S[["season", "model", "set", "mae", "skill", "skill_vs_persistence", "brier", "surges",
             "caught", "false_alarms", "lead_time"]].round(3).to_string(index=False))
    print(A.round(3).to_string(index=False))
    from src import report_results
    report_results.write(runs, line_method, line_note, A, chosen, sel_log, chosen_cls, cls_log, table)
    return runs, line_method, A, chosen, sel_log


if __name__ == "__main__":
    main()
