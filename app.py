"""
MonsoonWatch -- local Streamlit app.

Run with:   streamlit run app.py

Two tabs: About (background, method flowchart, key results) and Make a
forecast (enter the latest weekly case counts, get forecasts 1-4 weeks ahead).
Everything reads local files; no network calls at runtime.

Forecasts come from the weather-free ensemble built by
src/modelling/build_noweather_ensemble.py -- weather added no accuracy, so the
app does not ask for it. The earlier, more detailed app is kept in
archive/app_v1_detailed.py.
"""

import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common.epiweek import epiweek_to_end_date  # noqa: E402
from src.common.viz_style import C, ROLE, MUTED  # noqa: E402
from src.modelling.predict import (  # noqa: E402
    HORIZONS, load_history, load_models, origin_range, forecast_all, t_of,
    yw_of)

st.set_page_config(page_title="MonsoonWatch", page_icon="🌧", layout="wide")

RULES = {"persistence": "copy this week's count",
         "seasonal naive": "same week last year",
         "seasonal + drift": "same week last year, scaled by growth",
         "4-week rolling mean": "average of the last 4 weeks"}
LINE_LEGEND = alt.Legend(orient="top", symbolType="stroke",
                         symbolStrokeWidth=3, symbolDash=[1, 0])


# ------------------------------------------------------------------ loading
@st.cache_data
def history():
    return load_history()


@st.cache_data
def weekly_cases():
    b = pd.read_csv("data/processed/features_base.csv")[
        ["year", "week", "age_this_week"]]
    b["t"] = [t_of(y, w) for y, w in zip(b.year, b.week)]
    b["date"] = [week_end(t) for t in b.t]
    return b


@st.cache_resource
def models_cached():
    return load_models("noweather")


@st.cache_data
def backtest(h):
    bt = pd.read_csv(f"reports/metrics/backtest_noweather_h{h}.csv")
    bt["t"] = [t_of(y, w) for y, w in zip(bt.target_year, bt.target_week)]
    bt["date"] = [week_end(t) for t in bt.t]
    return bt


@st.cache_data
def ratio_band(h):
    """Range that 8 in 10 past forecasts fell within, as ratios of the
    forecast (from the 2024-2025 hold-out)."""
    bt = backtest(h)
    r = (bt.target / bt.pred).replace([np.inf, -np.inf], np.nan).dropna()
    return tuple(np.quantile(r, [0.1, 0.9]))


def week_end(t):
    return pd.Timestamp(epiweek_to_end_date(*yw_of(t)))


def segments(df, t_col="t"):
    """Label unbroken runs of weeks, so a line is drawn with a gap where
    bulletins are missing rather than a straight line across them."""
    df = df.sort_values(t_col).copy()
    df["segment"] = (df[t_col].diff() != 1).cumsum().astype(str)
    return df


def plural(h):
    return f"{h} week{'s' if h > 1 else ''}"


hist = history()
cases = weekly_cases()
models = models_cached()

# ------------------------------------------------------------------- header
st.title("🌧 MonsoonWatch")
st.caption("Forecasting diarrhoeal disease in Nepal 1–4 weeks ahead · "
           "Pitch for the Planet, Macquarie University")

tab_about, tab_fc = st.tabs(["📖 About", "🔮 Make a forecast"])

# ==================================================================== ABOUT
with tab_about:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("The problem")
        st.markdown("""
- Every monsoon, floods contaminate drinking water and **acute
  gastroenteritis (AGE)**, a diarrhoeal disease, surges across Nepal. Young
  children are hit hardest.
- Nepal's early-warning system, **EWARS**, counts cases every week, but only
  **after** they happen. It does not forecast.
- The weekly counts are published as **PDF bulletins**, not as usable data.
""")
    with right:
        st.subheader("My solution")
        st.markdown("""
- I turned **five years of EWARS PDF bulletins** into a clean weekly dataset.
- MonsoonWatch **forecasts national AGE cases 1–4 weeks ahead**.
- That lead time lets health teams put **oral rehydration salts, chlorine and
  staff in place before cases rise**, not after.
""")

    if models:
        mape = [b["test_MAPE"] for b in models.values()]
        imp = [b["improvement_pct"] for b in models.values()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Weekly bulletins used",
                  f"{int(cases.age_this_week.notna().sum())}",
                  help="EWARS bulletins, 2021–2025.", border=True)
        c2.metric("Forecast horizon", "1–4 weeks", border=True)
        c3.metric("Typical forecast error",
                  f"{min(mape):.0f}–{max(mape):.0f}%", border=True,
                  help="Average % miss on 2024–2025 weeks the model never "
                       "saw while it was being built.")
        c4.metric("Better than simple rules",
                  f"{min(imp):.0f}–{max(imp):.0f}%", border=True,
                  help="How much smaller the average miss is than the best "
                       "simple rule (e.g. 'copy this week's count') at each "
                       "horizon.")

    st.subheader("How it works")
    st.graphviz_chart("""
digraph {
  graph [rankdir=LR, bgcolor="transparent", nodesep=0.3, ranksep=0.4];
  node  [shape=box, style="rounded,filled", fontname="Helvetica",
         fontsize=12, color="#c9c8c2", fontcolor="#0b0b0b",
         margin="0.16,0.08"];
  edge  [color="#8a8984", arrowsize=0.7, penwidth=1.3];

  pdf  [fillcolor="#e3eefb", label=<<b>1. Collect</b><br/>EWARS weekly PDF<br/>bulletins, 2021–2025>];
  read [fillcolor="#e3eefb", label=<<b>2. Extract</b><br/>scraper reads 3 PDF<br/>layouts → 241 weeks>];
  feat [fillcolor="#e3eefb", label=<<b>3. Build inputs</b><br/>last 4 weeks' cases<br/>same week last year<br/>growth · time of year>];
  cand [fillcolor="#fde8df", label=<<b>4. Try 720 set-ups</b><br/>180 for each forecast<br/>length · XGBoost<br/>LightGBM · Random Forest<br/>Extra Trees · Ridge>];
  pick [fillcolor="#fde8df", label=<<b>5. Choose</b><br/>rank on 2021–2023 only<br/>keep the best 15–30>];
  avg  [fillcolor="#fde8df", label=<<b>6. Average</b><br/>their forecasts,<br/>retrained every week>];
  test [fillcolor="#fde8df", label=<<b>7. Test</b><br/>on unseen 2024–2025<br/>vs simple rules>];
  out  [fillcolor="#d9f2e6", label=<<b>Forecast</b><br/>AGE cases<br/>1–4 weeks ahead>];
  rain [fillcolor="#f1f0ec", style="rounded,filled,dashed", label=<<b>Rainfall</b><br/>23 valley gauges +<br/>NASA, all of Nepal>];

  pdf -> read -> feat -> cand;
  rain -> feat [style=dashed, label=<<font point-size="10" color="#8a8984"> tested: no gain, left out </font>>];
  cand -> pick;
  avg -> pick [dir=back];
  test -> avg [dir=back];
  out -> test [dir=back];
  {rank=same; pdf; rain; out}
  {rank=same; read; test}
  {rank=same; feat; avg}
  {rank=same; cand; pick}
}
""", width="stretch")

    st.subheader("What I found")
    st.markdown("**1 · Cases climb every monsoon**")
    g = segments(cases.dropna(subset=["age_this_week"]))
    years = sorted(cases.year.unique())
    bands = pd.DataFrame({
        "start": [week_end(t_of(y, 22)) - pd.Timedelta(days=6)
                  for y in years],
        "end": [week_end(t_of(y, 39)) for y in years]})
    chart = (
        alt.Chart(bands).mark_rect(color=C["aqua"], opacity=0.14)
        .encode(x="start:T", x2="end:T")
        + alt.Chart(g).mark_line(color=ROLE["age"], strokeWidth=2,
                                 point=alt.OverlayMarkDef(size=10))
        .encode(x=alt.X("date:T", title=None),
                y=alt.Y("age_this_week:Q", title="AGE cases per week"),
                detail="segment:N",
                tooltip=[alt.Tooltip("date:T", title="Week ending",
                                     format="%d %b %Y"),
                         alt.Tooltip("age_this_week:Q", title="Cases",
                                     format=",.0f")])
    ).properties(height=280)
    st.altair_chart(chart, width="stretch")
    ym = cases.groupby("year").age_this_week.mean()
    st.caption(f"Green bands = monsoon season. Gaps = missing bulletins. "
               f"Average weekly cases rose from {ym.iloc[0]:.0f} (2021) to "
               f"{ym.iloc[-1]:.0f} (2025), mostly because more health "
               f"facilities joined EWARS, not only because of more disease.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**2 · The forecast beats simple rules at every "
                    "horizon**")
        rows = []
        for h, b in models.items():
            rule = b["best_baseline"].replace("Baseline: ", "")
            rows += [{"Horizon": f"{plural(h)} ahead",
                      "Method": "Best simple rule", "Miss": b[
                          "best_baseline_MAE"],
                      "Detail": RULES.get(rule, rule)},
                     {"Horizon": f"{plural(h)} ahead",
                      "Method": "MonsoonWatch", "Miss": b["test_MAE"],
                      "Detail": f"{b['improvement_pct']:.0f}% smaller miss"}]
        df = pd.DataFrame(rows)
        enc = dict(
            x=alt.X("Horizon:N", title=None, sort=None,
                    axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Method:N", sort=["Best simple rule",
                                                  "MonsoonWatch"]),
            y=alt.Y("Miss:Q", title="Average miss (cases per week)"))
        bars = alt.Chart(df).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            color=alt.Color("Method:N", title=None,
                            scale=alt.Scale(
                                domain=["Best simple rule", "MonsoonWatch"],
                                range=[C["violet"], C["orange"]]),
                            legend=alt.Legend(orient="top")),
            tooltip=["Horizon", "Method",
                     alt.Tooltip("Miss:Q", format=".0f"), "Detail"], **enc)
        labels = alt.Chart(df).mark_text(dy=-7, fontSize=11,
                                         color=MUTED).encode(
            text=alt.Text("Miss:Q", format=".0f"), **enc)
        st.altair_chart((bars + labels).properties(height=300),
                        width="stretch")
        st.caption("Tested on 2024–2025 weeks the model never saw. Simple "
                   "rules: copy this week's count, or use the same week "
                   "last year.")

    with c2:
        st.markdown("**3 · Rainfall did not improve the forecast**")
        nr = pd.read_csv("reports/metrics/national_rain_test.csv")
        opts = [("No rainfall (this app)", "no weather"),
                ("Kathmandu Valley gauges", "valley gauges (deployed)"),
                ("All-Nepal rainfall", "national (NASA)"),
                ("All-Nepal, wetter/drier than normal",
                 "national anomaly (NASA)")]
        df = pd.DataFrame({"Input": [a for a, _ in opts],
                           "Miss": [nr[c].mean() for _, c in opts]})
        df["Kind"] = np.where(df.Input.str.startswith("No"), "none", "rain")
        enc = dict(y=alt.Y("Input:N", title=None, sort=None,
                           axis=alt.Axis(labelLimit=260)),
                   x=alt.X("Miss:Q", title="Average miss (cases per week)"))
        bars = alt.Chart(df).mark_bar(
            cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            color=alt.Color("Kind:N", legend=None,
                            scale=alt.Scale(domain=["none", "rain"],
                                            range=[C["orange"], C["aqua"]])),
            tooltip=["Input", alt.Tooltip("Miss:Q", format=".1f")], **enc)
        labels = alt.Chart(df).mark_text(align="left", dx=4, fontSize=11,
                                         color=MUTED).encode(
            text=alt.Text("Miss:Q", format=".1f"), **enc)
        st.altair_chart((bars + labels).properties(height=300),
                        width="stretch")
        st.caption("Average over the 1–4 week forecasts. All four are "
                   "within about 1 case of each other, so the app needs no "
                   "weather data. Knowing the time of year already captures "
                   "the monsoon.")

    st.subheader("Sustainable Development Goals")
    s1, s2, s3 = st.columns(3)
    s1.markdown("**SDG 3 · Good health**  \nEarly warning for a leading "
                "cause of child illness.")
    s2.markdown("**SDG 6 · Clean water**  \nAGE surges show when unsafe "
                "water needs treating.")
    s3.markdown("**SDG 13 · Climate action**  \nPrepares health services "
                "for the yearly monsoon.")

    with st.expander("Limitations"):
        st.markdown("""
- **National totals only.** District counts exist for 2024–2025 only, too
  short to forecast districts.
- **Reported, not true, cases.** Much of the rise since 2021 is more
  facilities reporting.
- **Big peaks are under-forecast**, and 18 weekly bulletins are missing.
- **Small dataset.** About five years of weekly counts; more years would
  help more than a different algorithm.
""")

# ================================================================= FORECAST
with tab_fc:
    st.subheader("Forecast the next 4 weeks")
    st.markdown("Pick the week of your latest EWARS bulletin and enter its "
                "AGE case count plus the three weeks before. MonsoonWatch "
                "looks up last year's numbers and the time of year itself. "
                "**No weather data needed.**")
    if len(models) < len(HORIZONS):
        st.error("Models not found. Run "
                 "`python src/modelling/build_noweather_ensemble.py` first.")
        st.stop()

    lo, hi = origin_range(hist)
    latest = int(hist.dropna().index.max())
    options = list(range(lo, hi + 1))

    def fmt(t):
        y, w = yw_of(t)
        return f"{y} · week {w} (ends {week_end(t):%d %b %Y})"

    T = st.selectbox("Week of your latest bulletin", options,
                     index=options.index(latest), format_func=fmt)
    year, week = yw_of(T)

    filled = hist.interpolate(limit_area="inside")
    defaults, on_record = [], True
    for k in range(4):
        v = hist.get(T - k, np.nan)
        if not np.isfinite(v):
            on_record = False
            v = filled.get(T - k - 52, np.nan)
        defaults.append(int(round(v)) if np.isfinite(v) else 0)

    labels = ["This week", "1 week before", "2 weeks before",
              "3 weeks before"]
    recent = []
    for k, col in enumerate(st.columns(4)):
        recent.append(col.number_input(
            f"{labels[k]} (week {yw_of(T - k)[1]})", min_value=0,
            max_value=100_000, value=defaults[k], step=1,
            key=f"cases_{T}_{k}"))
    if on_record:
        st.caption("Filled in from the EWARS bulletins on record. Change any "
                   "number to see how the forecast responds.")
    else:
        st.info("Some of these weeks have no bulletin on record, so those "
                "boxes start with **last year's** counts. Replace them with "
                "this year's numbers for a real forecast.")

    res = forecast_all(hist, models, year, week, recent)

    st.markdown("#### Forecast")
    for h, col in zip(HORIZONS, st.columns(4)):
        r = res[h]
        q10, q90 = ratio_band(h)
        r["lo"], r["hi"] = r["pred"] * q10, r["pred"] * q90
        col.metric(f"{plural(h)} ahead · week {r['target_week']}",
                   f"{r['pred']:,.0f} cases",
                   f"{r['pred'] - recent[0]:+,.0f} vs this week",
                   delta_color="inverse", border=True,
                   help=f"Week ending {r['target_date']:%d %b %Y}. 8 in 10 "
                        f"past forecasts at this horizon landed between "
                        f"{r['lo']:,.0f} and {r['hi']:,.0f}.")
        col.caption(f"Likely range **{r['lo']:,.0f}–{r['hi']:,.0f}** · "
                    f"ends {r['target_date']:%d %b %Y}")

    if recent[0] > 0:
        pct = 100 * (res[4]["pred"] - recent[0]) / recent[0]
        trend = ("rise" if pct > 10 else "fall" if pct < -10
                 else "stay about the same")
        st.markdown(f"**Outlook:** cases are expected to **{trend}** over "
                    f"the next 4 weeks ({recent[0]:,} → "
                    f"{res[4]['pred']:,.0f}, {pct:+.0f}%).")

    # ---- forecast in context
    SER = ["Reported cases", "Forecast", "Same weeks last year"]
    past = pd.DataFrame({"t": range(T - 11, T + 1)})
    past["cases"] = [recent[T - t] if T - t < 4 else hist.get(t, np.nan)
                     for t in past.t]
    past["series"] = SER[0]
    fut = pd.DataFrame({"t": [T] + [T + h for h in HORIZONS],
                        "cases": [recent[0]] + [res[h]["pred"]
                                                for h in HORIZONS],
                        "lo": [recent[0]] + [res[h]["lo"] for h in HORIZONS],
                        "hi": [recent[0]] + [res[h]["hi"] for h in HORIZONS]})
    fut["series"] = SER[1]
    ly = pd.DataFrame({"t": range(T - 11, T + max(HORIZONS) + 1)})
    ly["cases"] = [hist.get(t - 52, np.nan) for t in ly.t]
    ly["series"] = SER[2]
    lines = pd.concat([segments(d.dropna(subset=["cases"]))
                       .assign(segment=lambda x, s=s: s + x.segment)
                       for d, s in ((past, "p"), (fut, "f"), (ly, "l"))])
    lines["date"] = [week_end(t) for t in lines.t]
    fut["date"] = [week_end(t) for t in fut.t]

    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b %Y"))
    band = alt.Chart(fut).mark_area(color=C["orange"], opacity=0.15).encode(
        x=x, y="lo:Q", y2="hi:Q")
    line = alt.Chart(lines).mark_line(point=True, strokeWidth=2.2).encode(
        x=x,
        y=alt.Y("cases:Q", title="AGE cases per week",
                scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=SER, range=[
                            ROLE["actual"], ROLE["pred"], MUTED]),
                        legend=LINE_LEGEND),
        strokeDash=alt.StrokeDash("series:N", legend=None,
                                  scale=alt.Scale(domain=SER, range=[
                                      [1, 0], [6, 3], [2, 3]])),
        detail="segment:N",
        tooltip=[alt.Tooltip("date:T", title="Week ending",
                             format="%d %b %Y"),
                 "series:N", alt.Tooltip("cases:Q", title="Cases",
                                         format=",.0f")])
    st.altair_chart((band + line).properties(height=320), width="stretch")
    st.caption("Shaded band = likely range (8 in 10 past forecasts fell "
               "inside it). The dotted grey line shows the same weeks last "
               "year for comparison.")

    with st.expander("How is this forecast made?"):
        ks = [b["k"] for b in models.values()]
        st.markdown(f"""
- From your 4 numbers and the stored EWARS history, MonsoonWatch builds a few
  simple inputs: the recent level and trend, the same week last year, how much
  higher this year is running than last year, and the time of year.
- **{min(ks)}–{max(ks)} machine-learning models** (XGBoost, LightGBM, Random
  Forest, Extra Trees and Ridge) each make a forecast, and MonsoonWatch reports
  their **average**. Averaging was more reliable than trusting any one model.
- The likely range comes from how far past forecasts actually missed.
""")

    # ---- track record
    st.divider()
    st.subheader("How accurate has it been?")
    st.markdown("Every forecast below was made using **only the data "
                "available at the time**, on 2024–2025 weeks held back for "
                "testing.")
    hsel = st.radio("Forecast horizon", HORIZONS, horizontal=True,
                    key="acc_h", format_func=lambda h: f"{plural(h)} ahead")
    b = models[hsel]
    rule = b["best_baseline"].replace("Baseline: ", "")
    m1, m2, m3 = st.columns(3)
    m1.metric("Average miss", f"{b['test_MAE']:.0f} cases", border=True,
              help="Mean absolute error per week.")
    m2.metric("Average miss (%)", f"{b['test_MAPE']:.0f}%", border=True)
    m3.metric("Better than best simple rule", f"{b['improvement_pct']:.0f}%",
              border=True,
              help=f"Best simple rule at this horizon: "
                   f"{RULES.get(rule, rule)} (average miss "
                   f"{b['best_baseline_MAE']:.0f} cases).")

    bt = backtest(hsel)
    SER2 = ["Actual cases", "MonsoonWatch forecast"]
    long = pd.concat([
        segments(bt[["t", "date", "target"]].rename(
            columns={"target": "cases"})).assign(series=SER2[0]),
        segments(bt[["t", "date", "pred"]].rename(
            columns={"pred": "cases"})).assign(series=SER2[1])])
    chart = alt.Chart(long).mark_line(
        point=alt.OverlayMarkDef(size=18), strokeWidth=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
        y=alt.Y("cases:Q", title="AGE cases per week"),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=SER2, range=[
                            ROLE["actual"], ROLE["pred"]]),
                        legend=LINE_LEGEND),
        strokeDash=alt.StrokeDash("series:N", legend=None,
                                  scale=alt.Scale(domain=SER2,
                                                  range=[[1, 0], [6, 3]])),
        detail="segment:N",
        tooltip=[alt.Tooltip("date:T", title="Week ending",
                             format="%d %b %Y"),
                 "series:N", alt.Tooltip("cases:Q", title="Cases",
                                         format=",.0f")])
    st.altair_chart(chart.properties(height=320), width="stretch")
    st.caption("Gaps are weeks that could not be scored because a bulletin "
               "was missing.")
