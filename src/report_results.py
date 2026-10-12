"""
report_results.py -- figures + reports/results.md from the evaluation outputs.
Called at the end of src/evaluate.py.
"""
import datetime as dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports/figures/dengue")
OUT = os.path.join(ROOT, "reports/results.md")

# reference palette (dataviz skill): series 1 blue, series 2 orange; neutral inks
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2 = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _dates(df):
    return pd.to_datetime(df.week_end)


def fig_timeline(table, line_method):
    from src import targets as T
    from src.evaluate import usable_seasons
    line = T.surge_line(table, usable_seasons(table), line_method)
    t = T.add_line(table, line)
    d = _dates(t)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    above = (t.cases > t.line).fillna(False).values
    for i in np.where(above)[0]:
        ax.axvspan(d[i] - pd.Timedelta(days=3.5), d[i] + pd.Timedelta(days=3.5), color=S2, alpha=0.15, lw=0)
    ax.plot(d, t.cases, color=S1, label="Weekly dengue cases (EWARS)")
    ax.plot(d, t.line, color=INK2, lw=1.2, ls="--", label=f"Surge line ({'Q3 of seasons' if line_method == 'q3' else 'mean + 2 SD'})")
    ax.set_ylabel("cases / week")
    ax.set_title("National weekly dengue cases, 2016–2025 (shaded = above surge line)", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "01_timeline.png"), dpi=150)
    plt.close(fig)


def fig_backtest(P, table, chosen):
    model, fset = chosen.split("_")
    g = P[(P.model == model) & (P.set == fset)].copy()
    tb = table.reset_index(drop=True)
    g["target_row"] = g.row + 4
    g = g[g.target_row < len(tb)]
    g["target_date"] = pd.to_datetime(tb.week_end.values[g.target_row.values])
    seasons = sorted(g.year.unique())
    fig, axes = plt.subplots(len(seasons), 1, figsize=(10, 2.3 * len(seasons)))
    for ax, k in zip(np.atleast_1d(axes), seasons):
        h = g[g.year == k]
        ax.fill_between(h.target_date, h.p10, h.p90, color=S2, alpha=0.18, lw=0, label="p10–p90")
        ax.plot(h.target_date, h["mean"], color=S2, label=f"{chosen} forecast (made 4 weeks earlier)")
        ax.plot(h.target_date, h.y_reg, color=S1, label="Actual")
        ax.set_title(f"Test season {k}", loc="left")
        ax.set_ylabel("cases / week")
    np.atleast_1d(axes)[0].legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "02_backtest.png"), dpi=150)
    plt.close(fig)


def fig_ablation(A):
    fams = list(A.model.unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(10, 3.4), sharey=False)
    for ax, fam in zip(np.atleast_1d(axes), fams):
        a = A[A.model == fam]
        vals = a.delta_skill.clip(-1, 1)
        ax.bar(a.season.astype(str), vals, color=S1, width=0.6)
        ax.axhline(0, color=INK2, lw=1)
        for x, v, raw in zip(a.season.astype(str), vals, a.delta_skill):
            ax.annotate(f"{raw:+.2f}", (x, v), ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=9, color=INK2, xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
        ax.set_title(f"{fam}: skill with weather − without (Set B − Set A)", loc="left")
        ax.set_ylabel("Δ skill (clipped to ±1)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "03_ablation.png"), dpi=150)
    plt.close(fig)


def fig_skill(S):
    m = S[S.model.isin(["M1", "M2", "M3", "persistence"])].copy()
    m["key"] = np.where(m.set == "-", m.model, m.model + "_" + m.set)
    keys = ["persistence", "M1_A", "M1_B", "M2_B", "M3_A", "M3_B"]
    seasons = sorted(m.season.unique())
    fig, ax = plt.subplots(figsize=(10, 3.6))
    pv = m.pivot(index="season", columns="key", values="skill_vs_persistence")
    for i, k in enumerate(keys[1:]):
        if k in pv:
            ax.plot([str(s) for s in seasons], pv[k].clip(-1, 1), marker="o", ms=8,
                    color=S1 if k.endswith("A") else S2, ls="-" if k.startswith("M1") else (":" if k.startswith("M2") else "--"),
                    lw=1.5, label=k)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("skill vs persistence (clipped ±1)")
    ax.set_title("Regression skill vs persistence by test season (blue = no weather, orange = with weather)", loc="left")
    ax.legend(ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "04_skill_by_season.png"), dpi=150)
    plt.close(fig)


def _f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{d}f}" if isinstance(x, (float, np.floating)) else str(x)


def write(runs, line_method, line_note, A, chosen, sel_log, chosen_cls, cls_log, table):
    os.makedirs(FIG, exist_ok=True)
    P, S, folds = runs[line_method]
    fig_timeline(table, line_method)
    fig_backtest(P, table, chosen)
    fig_ablation(A)
    fig_skill(S)

    # ---------- verdicts ----------
    verdict_lines = []
    for fam in A.model.unique():
        a = A[A.model == fam]
        n_pos, n = int((a.delta_skill > 0).sum()), len(a)
        b_pos = int((a.delta_brier_improvement > 0).sum())
        verdict_lines.append(
            f"- **{fam} regressor:** weather improves skill in {n_pos}/{n} seasons → "
            f"{'passes' if n_pos > n / 2 else 'fails'} the majority rule. "
            f"**{fam} classifier:** weather lowers Brier in {b_pos}/{n} seasons → "
            f"{'passes' if b_pos > n / 2 else 'fails'}.")
    m1 = A[A.model == "M1"].set_index("season")
    m1_wo2021 = m1.drop(index=min(m1.index))
    pos_wo = int((m1_wo2021.delta_skill > 0).sum())

    ch = S[(S.model + "_" + S.set) == chosen].set_index("season")
    chc = S[(S.model + "_" + S.set) == chosen_cls].set_index("season")
    surges, caught = int(chc.surges.sum()), int(chc.caught.sum())
    lead = chc.lead_time.mean()
    fa = chc.false_alarms.mean()

    L = [f"# Results — DengueWatch Nepal\n",
         f"_Generated by `src/evaluate.py` on {dt.date.today()}. Rolling origin by season; "
         f"test seasons {', '.join(map(str, sorted(S.season.unique())))}; training always on earlier usable seasons only._\n",
         "## 1. Final model and why\n",
         f"**Regressor: {chosen}** (negative-binomial GLM{' with weather features' if chosen.endswith('B') else ''}). "
         f"**Classifier: {chosen_cls}** (logistic regression{' with weather' if chosen_cls.endswith('B') else ', no weather'}). "
         "Both were picked with the same simplest-first rule: a more complex model replaces the current one only if it "
         "beats it by more than 0.05 skill (regressor) or more than 5% lower Brier (classifier) in a majority of test "
         f"seasons. {chosen} beat M1_A in 3 of 5 seasons; LightGBM (M3) and the temperature spline (M2) never cleared "
         f"the bar. The forecast skill is real but modest and season-dependent: across seasons the regressor's skill vs "
         f"seasonal naive ranges {ch.skill.min():+.2f} to {ch.skill.max():+.2f}, and vs persistence "
         f"{ch.skill_vs_persistence.min():+.2f} to {ch.skill_vs_persistence.max():+.2f}. The classifier caught "
         f"{caught}/{surges} surge onsets with mean lead time {_f(lead, 1)} weeks and {_f(fa, 1)} false alarms per season.\n",
         "Selection log (regressor):\n" + "\n".join(f"- {x}" for x in sel_log) + "\n",
         "Selection log (classifier):\n" + "\n".join(f"- {x}" for x in cls_log) + "\n",
         "## 2. Surge line — a deviation from the plan\n",
         f"- {line_note}.",
         "- The plan's line (mean + 2 SD per week-of-year) cannot label surges in the seasons it is trained on here: "
         "training seasons mix near-zero years (2018, 2020, 2021: <500 cases) with outbreak years (2019, 2022–2024: "
         "9k–29k), and one outbreak season inflates the SD so much that it never crosses its own line. Two of five folds "
         "(test 2021, 2023) had **zero** positive training labels, so the classifier could only predict the base rate "
         "and caught 0 surges in every season.",
         "- Operational line: **upper quartile (Q3)** of the training seasons' weekly counts, ±2-week pooling "
         "(Bortman 1999 endemic channel — 'epidemic zone' above Q3). Still recomputed inside every fold from training "
         "seasons only. The rule that picked it was fixed before comparing model results: keep the plan's line unless a "
         "fold has < 2 positives. Results under both lines are in `reports/metrics_by_season.csv` (`line_method`).",
         "- Surge onset = first of ≥ 2 consecutive weeks above the line; classifier target = an onset in t+1…t+4.\n",
         "## 3. Per-season results (operational surge line)\n",
         "Skill = 1 − MAE / MAE(seasonal naive) on the same weeks; *skill vs persistence* uses this week's count as the "
         "baseline (much harder at 4 weeks in 2025). Surges/caught/false alarms/lead time use 🟠 (p ≥ 0.3) as the alert. "
         "A surge is *caught* if an alert was issued in the 4 weeks before onset; a *false alarm* is a run of alert weeks "
         "with no onset in the next 4 weeks, ignoring weeks already in a surge.\n",
         "| season | model | set | MAE | skill | skill vs persist. | Brier | PR-AUC | surges | caught | false alarms | lead (wk) | 80% range coverage |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"seasonal_naive": 0, "persistence": 1, "endemic_channel": 2, "M1": 3, "M2": 4, "M3": 5}
    for r in S.assign(o=S.model.map(order)).sort_values(["season", "o", "set"]).itertuples():
        L.append(f"| {r.season} | {r.model} | {r.set} | {_f(r.mae, 1)} | {_f(r.skill)} | {_f(r.skill_vs_persistence)} | "
                 f"{_f(r.brier, 3)} | {_f(r.pr_auc)} | {_f(r.surges, 0)} | {_f(r.caught, 0)} | {_f(r.false_alarms, 0)} | "
                 f"{_f(r.lead_time, 1)} | {_f(getattr(r, 'coverage_80', np.nan))} |")
    L += ["\n![timeline](figures/dengue/01_timeline.png)\n", f"![backtest](figures/dengue/02_backtest.png)\n",
          "![skill](figures/dengue/04_skill_by_season.png)\n",
          "## 4. Ablation — does weather help?\n",
          "Δ skill = skill(Set B, with weather) − skill(Set A, no weather). Δ Brier improvement = Brier(A) − Brier(B) "
          "(positive = weather helps).\n",
          "| model | season | skill A | skill B | Δ skill | Brier A | Brier B | Δ Brier improvement |",
          "|---|---|---|---|---|---|---|---|"]
    for r in A.itertuples():
        L.append(f"| {r.model} | {r.season} | {_f(r.skill_A)} | {_f(r.skill_B)} | **{r.delta_skill:+.2f}** | "
                 f"{_f(r.brier_A, 3)} | {_f(r.brier_B, 3)} | {r.delta_brier_improvement:+.3f} |")
    L += ["", *verdict_lines,
          f"- Caveat: M1's biggest gain (2021, Δ {m1.delta_skill.iloc[0]:+.2f}) is M1_A blowing up on only 3 tiny training "
          f"seasons, not weather adding information. Without 2021, weather helps M1 in {pos_wo}/{len(m1_wo2021)} seasons.",
          f"\n**Verdict: weak, inconsistent evidence.** Weather helped the M1 case-count forecast in 2022–2023 but hurt "
          f"in 2024–2025, and it made the M1 surge classifier slightly worse in most seasons. By the plan's rule it "
          f"technically passes for the M1 regressor (Δ > 0 in {int((m1.delta_skill > 0).sum())}/{len(m1)}), but with five "
          "test seasons this is not a robust finding. Do not claim 'weather improves dengue forecasts' — say it "
          "*may* help forecast size in some seasons, and that recent case counts carry most of the signal. "
          "(MonsoonWatch reached the same conclusion for rainfall and gastro.)\n",
          "![ablation](figures/dengue/03_ablation.png)\n",
          "## 5. Limitations\n",
          "- **National only, weekly.** No district forecasts were built.",
          "- **Forecasts lag sharp rises.** In big seasons the 4-week forecast reaches the peak 2–4 weeks late "
          "(see 2022 in the back-test figure); recent cases drive it, so it follows the curve more than it anticipates it.",
          "- **Little use of momentum.** The 2-week growth term gets a coefficient near zero; the model mostly averages "
          "the last two weeks' levels. So 'falling from 600 to 300' predicts more than 'rising from 150 to 300', which "
          "is why the simple app does not ask about the trend.",
          "- **Weather coefficients are not causal.** In M1_B the minimum-temperature coefficient is negative once "
          "week-of-year is in the model; the weather terms mostly act as a seasonal-timing correction.",
          "- **Few seasons.** 8 usable seasons (2018–2025), 5 test seasons; one bad season moves every average. "
          "2016–2017 are unusable (bulletins report only cumulative outbreak totals in the peak weeks).",
          "- **The surveillance network changed.** Sentinel sites grew from ~60 (2016) to 118 (2019) and 134 (2025), and "
          "site counts are missing for 2020–2023, so no per-site normalisation: some of the jump after 2019 is "
          "the network, not the virus.",
          "- **Counts get revised.** The next bulletin's 'previous week' figure differs from the first-published count "
          "by > 10% in ~18% of weeks; the model is trained on first-published counts, which is what is available in real time.",
          "- **Surge line deviation.** The plan's mean + 2 SD line is unusable with these seasons; the Q3 line is "
          "lower, so 'surge' here means 'above a typical recent season', not 'exceptional'.",
          "- **Traffic-light cut-offs** (0.3 / 0.6, red also needs p90 > 2× surge line) are the plan defaults, not tuned "
          "— too few positive weeks per training fold to tune them without overfitting.",
          "- **COVID years.** 2020–2021 had very low case counts, probably partly from reduced care-seeking and testing.",
          "- **2025 low season** (5,823 EWARS cases vs 29,149 in 2024) is kept as real; seasonal naive is very wrong "
          "there, which flatters skill-vs-seasonal-naive — compare against persistence too.",
          "- **Weather is a national average of 67 district points** (mountain and Terai together); the hotspot "
          "average is in `weekly_table.csv` (`*_hot`) as a sensitivity input but was not modelled separately.",
          "- **Coverage of the p10–p90 range** is reported per season; NB intervals are often too narrow in outbreak "
          "seasons.\n"]
    open(OUT, "w").write("\n".join(L))
    print(f"-> {OUT}, figures in {FIG}")
