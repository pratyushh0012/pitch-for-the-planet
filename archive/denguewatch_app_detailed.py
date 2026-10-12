"""
DengueWatch Nepal -- Streamlit app.

    streamlit run app/app.py

Loads the saved model bundle (models/dengue/denguewatch.joblib) and the
pre-computed back-test; nothing is retrained on start-up.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # the bundle pickles src.models classes

# reference palette (dataviz skill): series blue/orange, neutral inks, reserved status colours
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
STATUS = {"green": ("#0ca30c", "🟢 Normal"), "amber": ("#fab219", "🟠 Get ready"), "red": ("#d03b3b", "🔴 Stand by")}

st.set_page_config(page_title="DengueWatch Nepal", page_icon="🦟", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join(ROOT, "models/dengue/denguewatch.joblib"))


@st.cache_data
def load_data():
    table = pd.read_csv(os.path.join(ROOT, "data/processed/weekly_table.csv"), parse_dates=["week_end"])
    preds = pd.read_csv(os.path.join(ROOT, "data/processed/backtest_predictions.csv"))
    mets = pd.read_csv(os.path.join(ROOT, "reports/metrics_by_season.csv"))
    return table, preds, mets


def light(prob, p90, line4, th):
    if prob >= th["amber"] and p90 > th["red_p90_x_line"] * line4:
        return "red"
    return "amber" if prob >= th["green"] else "green"


def style(fig, h=380, ytitle="cases / week"):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title=ytitle)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


B = load_bundle()
table, preds, mets = load_data()
TH = B["thresholds"]
LINE = B["surge_line"].set_index("woy").line
REG, CLS = B["chosen_reg"], B["chosen_cls"]
METHOD = B["line_method"]
EV = B["evaluation"]

st.title("🦟 DengueWatch Nepal")
st.caption("4-week-ahead national dengue surge forecasts from EWARS sentinel surveillance and NASA POWER weather. "
           "National, weekly. A planning aid for beds, platelets and staff, not a diagnosis.")

tab_over, tab_bt, tab_try, tab_card = st.tabs(["📈 Overview", "🔁 Back-test", "🎛️ Try it (what-if)", "📋 Model card"])

# ------------------------------------------------------------------ Overview
with tab_over:
    t = table.copy()
    t["woy"] = np.minimum(t.week, 52)
    t["line"] = t.woy.map(LINE)
    t["above"] = t.cases > t.line
    lt = B["latest"]
    line4 = LINE[(min(lt["week"], 52) + 4 - 1) % 52 + 1]
    status = light(lt["prob"], lt["p90"], line4, TH)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest data week", f"{lt['year']}-W{lt['week']:02d}")
    c2.metric("Expected cases in 4 weeks", f"{lt['mean']:.0f}", help=f"80% range {lt['p10']:.0f}–{lt['p90']:.0f}")
    c3.metric("P(surge starts within 4 weeks)", f"{lt['prob']:.0%}")
    c4.metric("Traffic light", STATUS[status][1])

    fig = go.Figure()
    d = t.week_end
    on = t.above.fillna(False).values
    # shaded surge weeks
    starts = np.where(on & ~np.r_[False, on[:-1]])[0]
    ends = np.where(on & ~np.r_[on[1:], False])[0]
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=d[s] - pd.Timedelta(days=3.5), x1=d[e] + pd.Timedelta(days=3.5),
                      fillcolor=ORANGE, opacity=0.13, line_width=0)
    fig.add_trace(go.Scatter(x=d, y=t.cases, name="Weekly dengue cases (EWARS)", line=dict(color=BLUE, width=2),
                             hovertemplate="%{y:,.0f} cases"))
    fig.add_trace(go.Scatter(x=d, y=t.line, name="Surge line", line=dict(color=INK2, width=1.2, dash="dash"),
                             hovertemplate="line %{y:,.0f}"))
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption(f"Shaded = weeks above the surge line ({'upper quartile of usable seasons, ±2 weeks' if METHOD == 'q3' else 'mean + 2 SD'}). "
               "Gaps = weeks with no usable bulletin (late 2016 and 2017 report only cumulative outbreak totals). "
               "The sentinel network roughly doubled in 2019, so pre-2019 levels are not comparable.")

# ------------------------------------------------------------------ Back-test
with tab_bt:
    p = preds[preds.line_method == METHOD]
    seasons = sorted(p.year.unique())
    a, b = st.columns([1, 2])
    season = a.selectbox("Test season", seasons, index=len(seasons) - 1)
    model_opts = ["M1_A", "M1_B", "M2_B", "M3_A", "M3_B"]
    reg_key = b.selectbox("Regressor", model_opts, index=model_opts.index(REG),
                          help=f"Official choice: {REG}. Trained only on seasons before {season}.")
    cls_key = b.selectbox("Classifier", model_opts, index=model_opts.index(CLS), help=f"Official choice: {CLS}.")
    r = p[(p.year == season) & (p.model + "_" + p.set == reg_key)].copy()
    c = p[(p.year == season) & (p.model + "_" + p.set == cls_key)].copy()
    tb = table.reset_index(drop=True)
    r = r[r.row + 4 < len(tb)]
    r["target_date"] = tb.week_end.values[r.row.values + 4]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.target_date, y=r.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r.target_date, y=r.p10, fill="tonexty", fillcolor="rgba(235,104,52,0.18)",
                             line=dict(width=0), name="p10–p90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r.target_date, y=r["mean"], name="Forecast (made 4 weeks earlier)",
                             line=dict(color=ORANGE, width=2), hovertemplate="forecast %{y:,.0f}"))
    fig.add_trace(go.Scatter(x=r.target_date, y=r.y_reg, name="Actual", line=dict(color=BLUE, width=2),
                             hovertemplate="actual %{y:,.0f}"))
    st.subheader(f"Actual vs forecast cases, {season}")
    st.plotly_chart(style(fig), width="stretch")

    c = c.merge(r[["row", "p90"]].rename(columns={"p90": "reg_p90"}), on="row", how="left")
    c["light"] = [light(pr, q if pd.notna(q) else 0, l4, TH) for pr, q, l4 in zip(c.prob, c.reg_p90, c.line_t4)]
    c["date"] = tb.week_end.values[c.row.values]
    fig2 = go.Figure()
    for k, (col, lab) in STATUS.items():
        h = c[c.light == k]
        fig2.add_trace(go.Bar(x=h.date, y=h.prob, name=lab, marker_color=col, hovertemplate="%{y:.0%}"))
    onsets = c[c.onset == 1]
    for dte in onsets.date:
        fig2.add_vline(x=dte, line=dict(color=INK2, dash="dot", width=1.5))
    fig2.add_hline(y=TH["green"], line=dict(color=GRID, width=1))
    fig2.add_hline(y=TH["amber"], line=dict(color=GRID, width=1))
    fig2.update_yaxes(range=[0, 1], tickformat=".0%")
    st.subheader("Weekly probability of a surge starting in the next 4 weeks")
    st.plotly_chart(style(fig2, 300, "probability").update_layout(barmode="overlay", bargap=0.15),
                    width="stretch")
    st.caption("Dotted lines = surge onsets (first of ≥ 2 weeks above the surge line computed from earlier seasons). "
               "A surge counts as caught if 🟠 or 🔴 was shown in the 4 weeks before onset.")

    # lead time achieved
    rows = []
    alert = c.set_index("row").prob >= TH["green"]
    for s in onsets.row:
        win = [x for x in range(s - 4, s) if x in alert.index and alert[x]]
        rows.append({"onset week": f"{season}-W{int(c.set_index('row').week[s]):02d}",
                     "first alert": f"W{int(c.set_index('row').week[min(win)]):02d}" if win else "—",
                     "lead time (weeks)": s - min(win) if win else "missed"})
    m = mets[(mets.line_method == METHOD) & (mets.season == season)]
    mr = m[(m.model + "_" + m.set) == reg_key].iloc[0]
    mc = m[(m.model + "_" + m.set) == cls_key].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE (cases/week)", f"{mr.mae:,.0f}")
    k2.metric("Skill vs seasonal naive", f"{mr.skill:+.2f}")
    k3.metric("Skill vs persistence", f"{mr.skill_vs_persistence:+.2f}")
    k4.metric("Brier score", f"{mc.brier:.3f}")
    st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"note": ["no surge onset in this season"]}),
                 hide_index=True, width="stretch")
    with st.expander("Weekly traffic lights"):
        st.dataframe(c.assign(week=c.week.astype(int), light=c.light.map(lambda k: STATUS[k][1]),
                              prob=c.prob.round(2))[["week", "cases", "prob", "light"]],
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ What-if
with tab_try:
    st.markdown("Set recent conditions; the saved final models turn them into a 4-week-ahead forecast. "
                "Weather sliders are **relative to normal for the chosen week** (2017–2025 average) and assume the "
                "anomaly held for the last 8–12 weeks, which keeps inputs realistic.")
    l, rcol = st.columns([1, 1], gap="large")
    with l:
        week = st.slider("Current epi-week", 1, 52, 32)
        cases = st.slider("Dengue cases this week (national EWARS)", 0, 3000, 200, step=10)
        growth = st.slider("Change vs 2 weeks ago (%)", -80, 300, 30, step=5)
        last_total = st.slider("Last calendar year's total cases", 0, 35000, 6000, step=500)
        t_anom = st.slider("Temperature vs normal (°C)", -3.0, 3.0, 0.0, step=0.25)
        rain_pct = st.slider("Rainfall, % of normal", 0, 200, 100, step=10)
        rh_anom = st.slider("Relative humidity vs normal (percentage points)", -15, 15, 0)
        use_weather_cls = st.checkbox(f"Use the weather classifier (M1_B) instead of the official {CLS}",
                                      value=CLS.endswith("B"))
    c_prev2 = cases / (1 + growth / 100)
    c_prev1 = np.sqrt(max(cases, 0) * max(c_prev2, 0)) if cases > 0 and c_prev2 > 0 else (cases + c_prev2) / 2
    c_prev3 = c_prev2 * (c_prev2 / c_prev1) if c_prev1 > 0 else c_prev2
    w = min(week, 52)
    clim = table[table.year >= 2017].assign(woy=np.minimum(table.week, 52)).groupby("woy")[["tmean", "tmin", "rain", "rh"]].mean()
    past = lambda n: [((w - 1 - k) % 52) + 1 for k in range(n)]          # this week and n-1 before it
    lag4_12 = [((w - 1 - k) % 52) + 1 for k in range(4, 13)]
    tm8 = clim.tmean.loc[past(8)] + t_anom
    rh8 = (clim.rh.loc[past(8)] + rh_anom).clip(upper=100)
    with l:
        st.caption(f"→ this week: {clim.tmean[w] + t_anom:.1f} °C mean, {clim.rain[w] * rain_pct / 100:.0f} mm rain, "
                   f"{min(100, clim.rh[w] + rh_anom):.0f}% RH (national average of 67 districts)")
    x = pd.DataFrame([{
        "log_cases_0": np.log1p(cases), "log_cases_1": np.log1p(c_prev1), "log_cases_2": np.log1p(c_prev2),
        "log_cases_3": np.log1p(c_prev3), "growth_2": np.log1p(cases) - np.log1p(c_prev2),
        "log_last_season": np.log1p(last_total), "woy_sin": np.sin(2 * np.pi * w / 52),
        "woy_cos": np.cos(2 * np.pi * w / 52),
        "tmin_mean_4": clim.tmin.loc[past(4)].mean() + t_anom, "tmin_mean_8": clim.tmin.loc[past(8)].mean() + t_anom,
        "tmean_mean_lag4_12": clim.tmean.loc[lag4_12].mean() + t_anom,
        "heat_above_18_8": (tm8 - 18).clip(lower=0).sum(),
        "rain_sum_4": clim.rain.loc[past(4)].sum() * rain_pct / 100,
        "rain_sum_8": clim.rain.loc[past(8)].sum() * rain_pct / 100,
        "rh_mean_4": (clim.rh.loc[past(4)] + rh_anom).clip(upper=100).mean(),
        "warm_wet_weeks_8": float(((tm8.values >= B["warm_cutoff"]) & (rh8.values >= 70)).sum())}])
    rng = B["feature_range"]
    raw_x = x.copy()
    for col_ in x.columns:
        x[col_] = x[col_].clip(*rng[col_])
    used = set(B["features"][REG]) | set(B["features"]["M1_B" if use_weather_cls else CLS])
    clamped = [col_ for col_ in x.columns if col_ in used and not np.isclose(x[col_].iloc[0], raw_x[col_].iloc[0])]
    reg = B["models"][REG].predict(x).iloc[0]
    cls_model = B["models"]["M1_B" if use_weather_cls else CLS]
    prob = float(cls_model.predict(x).prob.iloc[0])
    line4 = LINE[(w + 4 - 1) % 52 + 1]
    lt = light(prob, reg.p90, line4, TH)
    with rcol:
        col, lab = STATUS[lt]
        st.markdown(f"<div style='font-size:2.2rem;font-weight:700'>{lab}</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("P(surge starts within 4 weeks)", f"{prob:.0%}")
        m2.metric("Expected cases in 4 weeks", f"{reg['mean']:,.0f}", help="negative-binomial mean")
        st.metric("80% range (p10–p90)", f"{reg.p10:,.0f} – {reg.p90:,.0f}")
        st.metric("Surge line in 4 weeks", f"{line4:,.0f}")
        st.caption(f"🟢 < {TH['green']:.0%} ≤ 🟠 < {TH['amber']:.0%} ≤ 🔴 (🔴 also needs p90 > "
                   f"{TH['red_p90_x_line']:.0f}× the surge line). Regressor: {REG} "
                   f"({'uses' if REG.endswith('B') else 'ignores'} weather). Classifier: "
                   f"{'M1_B (uses weather)' if use_weather_cls else CLS + (' (uses weather)' if CLS.endswith('B') else ' (no weather — weather sliders do not change the probability)')}.")
        if reg["mean"] > 1.5 * table.cases.max():
            st.warning(f"The expected count is above anything EWARS has recorded (max {table.cases.max():,.0f}/week): "
                       "this input combination is outside the data, so the number is not trustworthy.")
        if REG.endswith("B"):
            st.caption("Heads-up: in this model a *cooler* setting raises the forecast. That's a statistical artefact of "
                       "temperature overlapping with week-of-year, not biology (see Model card → Limitations).")
        if clamped:
            st.warning("Outside the training range, so clamped to the nearest value seen in 2018–2025: "
                       + ", ".join(clamped) + ". Combinations of inputs can still be unusual; treat with care.")

# ------------------------------------------------------------------ Model card
with tab_card:
    m = mets[mets.line_method == METHOD]
    st.subheader("What it predicts")
    st.markdown(f"""
Every week *t*: (1) **probability** that a surge starts in weeks t+1…t+4 (logistic regression, **{CLS}**);
(2) **expected cases at t+4** with an 80% range (negative-binomial GLM, **{REG}**); (3) a **traffic light** from both.
A surge starts when weekly cases are above the surge line for 2 consecutive weeks.
""")
    st.subheader("Data")
    st.markdown("""
- **Cases:** EDCD EWARS weekly bulletins, 2016–2025 (482 bulletins harvested from edcd.gov.np; dengue parsed from
  prose and tables). Sentinel hospitals only, so counts are lower than published national totals
  (~40–50% of the 2019/2022/2023 totals).
- **Weather:** NASA POWER daily T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, RH2M at 67 district headquarters, Sep 2016 onward,
  averaged to the EDCD week (Sunday-ending; calendar calibrated from bulletin header dates; 2020 has 53 weeks).
- **Usable seasons:** 2018–2025 (≥ 40 weeks with data and ≤ 3 missing peak weeks).
""")
    st.subheader("Features")
    st.markdown("""
| group | features |
|---|---|
| recent cases | log(1 + cases) this week and 1 week ago; 2-week growth |
| immunity proxy | log(1 + last calendar year's total) |
| season | sin / cos of week-of-year |
| weather (M1_B only) | min temperature, mean of last 8 weeks; relative humidity, mean of last 4 weeks |

M1_B's two weather features were picked from the literature (Chitwan: minimum temperature ~2 months earlier, and
humidity), not by searching this data.
""")
    st.subheader("Back-test (rolling origin by season, training on earlier seasons only)")
    show = m[m.model.isin(["M1", "M2", "M3", "persistence", "seasonal_naive"])].copy()
    show["model"] = np.where(show.set == "-", show.model, show.model + "_" + show.set)
    st.dataframe(show[["season", "model", "mae", "skill", "skill_vs_persistence", "brier", "surges", "caught",
                       "false_alarms", "lead_time"]].round(3), hide_index=True, width="stretch")
    st.subheader("Does weather help? (ablation)")
    abl = []
    for fam in ("M1", "M3"):
        a = m[(m.model == fam) & (m.set == "A")].set_index("season")
        b = m[(m.model == fam) & (m.set == "B")].set_index("season")
        for k in a.index:
            abl.append({"model": fam, "season": k, "Δ skill (B − A)": round(b.skill[k] - a.skill[k], 3),
                        "Δ Brier improvement (A − B)": round(a.brier[k] - b.brier[k], 3)})
    abl = pd.DataFrame(abl)
    st.dataframe(abl, hide_index=True, width="stretch")
    n_m1 = int((abl[abl.model == "M1"]["Δ skill (B − A)"] > 0).sum())
    st.info(f"**Weak, inconsistent.** Weather improved the M1 case-count forecast in {n_m1}/5 test seasons "
            "(2021–2023), but the 2021 gain is mostly the no-weather model being unstable on 3 tiny training "
            "seasons, and weather hurt in 2024–2025. It made the surge classifier slightly worse in most seasons. "
            "Recent case counts carry most of the signal.")
    st.subheader("Limitations")
    st.markdown(f"""
- **National only, weekly.** No district forecasts.
- **Few seasons:** 8 usable, 5 test seasons. One unusual season moves every average.
- **Surge line deviates from the plan:** the plan's mean + 2 SD line gave zero positive training weeks in 2 of 5
  folds (one outbreak season inflates the SD). The app uses the **upper quartile of past seasons** (Bortman endemic
  channel), so "surge" means *above a typical recent season*, not *exceptional*.
- **Forecasts lag sharp rises:** in big seasons the 4-week forecast reaches the peak 2–4 weeks late, because recent
  cases drive it.
- **The surveillance network changed** (≈60 → 118 → 134 sentinel sites), and site counts are missing for 2020–2023.
- **Counts are revised** in later bulletins (> 10% in ~18% of weeks). The model uses first-published counts.
- **Weather adds little, and inconsistently.** In M1_B the minimum-temperature coefficient is *negative* once
  week-of-year is in the model, which is not biologically meaningful. The weather terms mostly act as a
  seasonal-timing correction, so do not read them as causal effects.
- **Traffic-light cut-offs** are the defaults (0.3 / 0.6), not tuned.
""")
    with st.expander("Model-selection log"):
        st.write("Regressor:", EV["selection_log"])
        st.write("Classifier:", EV["cls_selection_log"])
        st.write(EV["line_note"])
    st.caption(f"Model trained {B['trained_on']} on seasons {B['train_years']}.")
