# ARCHIVED 2026-09-16: the earlier five-page app, kept for reference only.
# It imports INPUT_SPEC/predict_one/random_test_case from the old
# src/modelling/predict.py, which was replaced, so it no longer runs.
"""
MonsoonWatch -- local Streamlit demo.

Run with:   streamlit run app.py
Everything reads from local files; no network calls at runtime.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common.viz_style import (  # noqa: E402
    apply_style, ROLE, C, MUTED, GRID, INK2, break_gaps)
from src.modelling.predict import (  # noqa: E402
    INPUT_SPEC, predict_one, load_model, random_test_case, case_from_history)
from src.common.epiweek import epiweek_to_end_date  # noqa: E402

apply_style()
st.set_page_config(page_title="MonsoonWatch", page_icon="🌧", layout="wide")

HORIZONS = [1, 2, 3, 4]


# ------------------------------------------------------------------ loading
@st.cache_data
def load_base():
    return pd.read_csv("data/processed/features_base.csv")


@st.cache_data
def load_backtests():
    out = {}
    for h in HORIZONS:
        p = f"reports/metrics/backtest_h{h}.csv"
        if os.path.exists(p):
            out[h] = pd.read_csv(p)
    return out


@st.cache_data
def load_comparison():
    p = "reports/metrics/model_comparison.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_resource
def load_models():
    """Prefer the deployed top-k ensemble; fall back to the single model."""
    out = {}
    for h in HORIZONS:
        for p in (f"models/ensemble_h{h}.pkl", f"models/model_h{h}.pkl"):
            if os.path.exists(p):
                out[h] = joblib.load(p)
                break
    return out


base = load_base()
backtests = load_backtests()
models = load_models()
comparison = load_comparison()


def wk_label(y, w):
    d = epiweek_to_end_date(int(y), int(w))
    return f"{int(y)} wk {int(w)} (ends {d:%d %b %Y})" if d else \
        f"{int(y)} wk {int(w)}"


# ------------------------------------------------------------------- layout
st.title("🌧 MonsoonWatch")
st.caption("Forecasting monsoon diarrhoeal-disease risk in Nepal, 1-4 weeks "
           "ahead · Macquarie University *Pitch for the Planet*")

page = st.sidebar.radio(
    "Section",
    ["Overview", "Data explorer", "Hindcast: predicted vs actual",
     "Make a prediction", "Model performance"])
st.sidebar.markdown("---")
if models:
    st.sidebar.markdown("**Deployed forecasters**")
    for h, b in models.items():
        tag = (f"ensemble of {b['k']}" if "members" in b
               else f"{b.get('name')} · {b.get('framing')}")
        st.sidebar.caption(
            f"{h} wk: {tag} · MAE {b['test_MAE']:.0f} "
            f"({b['improvement_pct']:+.0f}% vs baseline)")
else:
    st.sidebar.warning("No trained models found. Run `python src/modelling/train_model.py`.")


# =============================================================== 1. OVERVIEW
if page == "Overview":
    st.subheader("The problem")
    st.markdown("""
Nepal's monsoon (June–September) floods contaminate drinking water, and cases
of **acute gastroenteritis (AGE)** climb every year — a burden that falls
hardest on children under five. Nepal's existing early-warning system, **EWARS**,
publishes a weekly bulletin that *counts cases after they have already
happened*. It does not forecast.

**MonsoonWatch** adds the missing half: a forecast of national weekly AGE cases
**1–4 weeks ahead**, built from the disease history plus real ground rainfall.
That lead time is what turns a count into a decision — pre-positioning oral
rehydration salts, chlorine and staff *before* the spike rather than during it.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Weeks of disease data", f"{base.age_this_week.notna().sum()}",
              "2021–2025, EWARS bulletins")
    c2.metric("Rain gauges used", "23", "DHM, Kathmandu Valley")
    if models:
        best_h = min(models, key=lambda h: models[h]["test_MAE"])
        b = models[best_h]
        c3.metric(f"Out-of-sample MAE ({best_h} wk ahead)",
                  f"{b['test_MAE']:.0f} cases",
                  f"{b['improvement_pct']:+.0f}% vs best baseline")

    st.subheader("Sustainable Development Goals")
    a, bcol, c = st.columns(3)
    a.markdown("**SDG 3 — Good Health**\n\nTarget 3.d is explicitly about "
               "*early-warning capacity* for health risks. This is that "
               "capacity, for the disease that drives Nepal's monsoon "
               "child-mortality burden.")
    bcol.markdown("**SDG 6 — Clean Water**\n\nAGE is the measurable *symptom* "
                  "of unsafe water. Forecasting the spike tells water and "
                  "sanitation teams where and when treatment matters most.")
    c.markdown("**SDG 13 — Climate Action** *(supporting)*\n\nThe model's "
               "inputs are rainfall and temperature, so it quantifies a "
               "direct climate-to-health transmission channel.")

    st.subheader("How it works")
    st.markdown("""
1. **Disease data** — a custom scraper parses EDCD's weekly EWARS PDF bulletins
   (three different layout eras, 2021–2025) into a clean national weekly series.
2. **Rainfall** — 23 DHM ground rain gauges, daily 2021–2023, aggregated to
   valley-level weekly features.
3. **Climate bridge** — DHM data ends Dec 2023, so NASA POWER satellite data is
   *calibrated against the gauges* on the 2021–2023 overlap (R² = 0.87) and used
   to extend coverage through 2025.
4. **Model** — gradient-boosted trees and regularised linear models compete;
   the winner is chosen by cross-validation on 2021–2023 only, then tested on an
   untouched 2024–2025 hold-out.
    """)
    st.info("**On novelty, stated honestly:** the modelling approach follows "
            "published work (a 2024 UMD-led study forecast diarrhoeal disease "
            "in Nepal, Vietnam and Taiwan). The contribution here is newer "
            "data (2021–2025), a disease series recovered from PDF bulletins "
            "that exist in no machine-readable form, and a pipeline that "
            "actually runs end to end.")
    st.warning("**A negative result we report rather than bury:** the 23 "
               "ground rain gauges were meant to be this project's main "
               "advantage over satellite data. They do not improve forecast "
               "accuracy at the national weekly scale — removing all weather "
               "features changes error by under 3% (see *Model performance*). "
               "The forecasts still beat every baseline at every horizon, but "
               "they do so on disease history and seasonality. The likely "
               "cause is geography: valley rainfall against a national case "
               "count.")


# ========================================================== 2. DATA EXPLORER
elif page == "Data explorer":
    st.subheader("Rainfall and disease, week by week")
    years = sorted(base.year.unique())
    sel = st.multiselect("Years to show", years, default=years)
    d = base[base.year.isin(sel)].dropna(subset=["age_this_week"])

    if d.empty:
        st.warning("Pick at least one year.")
    else:
        x = (d["year"] - 2021) * 52 + d["week"]
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                 gridspec_kw={"hspace": 0.16})
        for y in sel:
            axes[0].axvspan((y - 2021) * 52 + 22, (y - 2021) * 52 + 39,
                            color=C["aqua"], alpha=0.07, lw=0)
            axes[1].axvspan((y - 2021) * 52 + 22, (y - 2021) * 52 + 39,
                            color=C["aqua"], alpha=0.07, lw=0)
        axes[0].plot(x, d["age_this_week"], color=ROLE["age"], lw=2)
        axes[0].set_ylabel("AGE cases / week")
        axes[0].set_title("Reported acute gastroenteritis")
        dd = d.dropna(subset=["rain_mean_mm"])
        xr = (dd["year"] - 2021) * 52 + dd["week"]
        axes[1].fill_between(xr, dd["rain_mean_mm"], color=ROLE["rain"],
                             alpha=0.8, lw=0)
        axes[1].set_ylabel("Rainfall mm / week")
        axes[1].set_title("Valley rainfall (shaded band = monsoon, wk 22–39)")
        axes[1].set_xticks([(y - 2021) * 52 + 1 for y in sel])
        axes[1].set_xticklabels([str(y) for y in sel])
        st.pyplot(fig, use_container_width=True)
        st.caption("Two stacked panels, not a dual-axis chart: with two "
                   "y-scales on one frame you can manufacture almost any "
                   "apparent lead–lag relationship just by rescaling.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Seasonal shape** (each year scaled to its own mean, so "
                    "the long-run level rise doesn't hide the season)")
        dd = base.dropna(subset=["age_this_week"]).copy()
        dd["rel"] = dd.groupby("year")["age_this_week"].transform(
            lambda s: s / s.mean())
        a = dd.groupby("week")["rel"].mean()
        r = dd.groupby("week")["rain_mean_mm"].mean()
        fig, ax = plt.subplots(2, 1, figsize=(5.4, 4.2), sharex=True)
        ax[0].plot(a.index, a.values, color=ROLE["age"], lw=2)
        ax[0].axhline(1, color=GRID, lw=1)
        ax[0].set_ylabel("AGE (relative)")
        ax[1].fill_between(r.index, r.values, color=ROLE["rain"], alpha=0.8,
                           lw=0)
        ax[1].set_ylabel("Rain mm")
        ax[1].set_xlabel("Epi-week")
        st.pyplot(fig, use_container_width=True)
        st.caption(f"AGE peaks around week {int(a.idxmax())}; rainfall peaks "
                   f"around week {int(r.idxmax())}. Cases rise *before* peak "
                   "rain — monsoon onset appears to matter more than total "
                   "volume.")
    with c2:
        st.markdown("**The 23 DHM gauges**")
        if os.path.exists("data/interim/rainfall_stations.csv"):
            st_df = pd.read_csv("data/interim/rainfall_stations.csv")
            st.map(st_df.rename(columns={"Latitude": "lat",
                                         "Longitude": "lon"})[["lat", "lon"]],
                   size=400)
            st.caption(f"{len(st_df)} stations, elevation "
                       f"{st_df.Elevation.min():.0f}–"
                       f"{st_df.Elevation.max():.0f} m. Using the spread across "
                       "gauges catches localised cloudbursts that a single "
                       "satellite grid point averages away.")

    with st.expander("Show the weekly data table"):
        st.dataframe(base[["year", "week", "age_this_week", "rain_mean_mm",
                           "rain_max_mm", "max_daily_mm", "wet_days",
                           "temp_c_mean", "humidity_pct_mean",
                           "rain_source_is_dhm"]].round(2),
                     use_container_width=True, height=320)


# ======================================================= 3. HINDCAST EXPLORER
elif page == "Hindcast: predicted vs actual":
    st.subheader("What the model predicted vs what actually happened")
    st.markdown("""
Every point below is a **genuine out-of-sample forecast**. For each week the
model was retrained on data available *only up to that week*, then asked to
forecast forward — it never saw the answer, or any later week, at fit time.
The whole of 2024–2025 was held out of model selection entirely.
    """)
    if not backtests:
        st.warning("No backtest files. Run `python src/modelling/train_model.py` first.")
    else:
        h = st.selectbox("Forecast horizon", HORIZONS,
                         format_func=lambda x: f"{x} week(s) ahead")
        bt = backtests[h].copy()
        bt["x"] = (bt["target_year"] - 2021) * 52 + bt["target_week"]
        bt = bt.sort_values("x")

        lo, hi = st.select_slider(
            "Drag to move through the hold-out period",
            options=list(bt["x"]),
            value=(bt["x"].iloc[0], bt["x"].iloc[-1]),
            format_func=lambda v: wk_label(2021 + (v - 1) // 52,
                                           ((v - 1) % 52) + 1))
        w = bt[(bt.x >= lo) & (bt.x <= hi)]

        m1, m2, m3, m4 = st.columns(4)
        err = (w["pred"] - w["target"]).abs()
        m1.metric("Weeks shown", len(w))
        m2.metric("Mean absolute error", f"{err.mean():.0f} cases")
        m3.metric("Mean error %",
                  f"{(err / w['target']).mean() * 100:.1f}%")
        base_err = (w["base_seasonal_naive"] - w["target"]).abs().mean()
        m4.metric("vs seasonal-naive baseline", f"{base_err:.0f} cases",
                  f"{100 * (base_err - err.mean()) / base_err:+.0f}%")

        fig, ax = plt.subplots(figsize=(11, 4.0))
        gx, gt, gp, gb = break_gaps(w["x"].values, w["target"].values,
                                    w["pred"].values,
                                    w["base_seasonal_naive"].values)
        ax.plot(gx, gt, color=ROLE["actual"], lw=2.2, label="Actual cases",
                zorder=3)
        ax.plot(gx, gp, color=ROLE["pred"], lw=2.2, ls=(0, (5, 2)),
                label=f"MonsoonWatch forecast ({h} wk ahead)", zorder=4)
        ax.fill_between(gx, gt, gp, color=ROLE["pred"], alpha=0.13, lw=0)
        ax.plot(gx, gb, color=ROLE["baseline"], lw=1.4, alpha=0.75,
                label="Baseline: same week last year")
        yrs = sorted(w["target_year"].unique())
        ax.set_xticks([(y - 2021) * 52 + 1 for y in yrs]
                      + [(y - 2021) * 52 + 26 for y in yrs])
        ax.set_xticklabels([f"{int(y)}" for y in yrs]
                           + [f"{int(y)} mid" for y in yrs], fontsize=8)
        ax.set_ylabel("AGE cases / week")
        ax.set_title("Forecast vs reality on held-out weeks")
        ax.legend(loc="upper left", ncol=3)
        st.pyplot(fig, use_container_width=True)

        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(4.6, 4.4))
            ax.scatter(w["target"], w["pred"], s=34, color=ROLE["pred"],
                       alpha=0.75, edgecolor="white", linewidth=0.7)
            lim = [0, max(w["target"].max(), w["pred"].max()) * 1.08]
            ax.plot(lim, lim, color=MUTED, lw=1.3, ls=(0, (4, 3)),
                    label="perfect forecast")
            ax.set_xlabel("Actual cases")
            ax.set_ylabel("Predicted cases")
            ax.set_title("Calibration")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
        with c2:
            st.markdown("**Week-by-week detail**")
            show = w[["target_year", "target_week", "target", "pred",
                      "base_seasonal_naive"]].copy()
            show.columns = ["Year", "Week", "Actual", "Predicted",
                            "Last year"]
            show["Error"] = (show["Predicted"] - show["Actual"]).round(0)
            show["Predicted"] = show["Predicted"].round(0)
            st.dataframe(show.set_index(["Year", "Week"]),
                         use_container_width=True, height=360)


# ======================================================= 4. MAKE A PREDICTION
elif page == "Make a prediction":
    st.subheader("Forecast a week")
    st.markdown("""
Enter the situation as of **one week** and the model forecasts AGE cases
1–4 weeks later. You do not need to supply engineered features — the ones the
model actually uses (4-week averages, week-on-week change, rolling rainfall
totals, seasonal sine/cosine terms) are derived automatically from what you
type below.
    """)
    if not models:
        st.warning("No trained models found. Run `python src/modelling/train_model.py`.")
        st.stop()

    spec = {k: (lab, typ, hlp) for k, lab, typ, hlp in INPUT_SPEC}

    def apply_inputs(d):
        """Write straight into the widget keys.

        Streamlit ignores a widget's `value=` argument once that key already
        exists in session state, so loading a preset has to set the keys
        themselves — otherwise the buttons appear to do nothing.
        """
        for k, v in d.items():
            if k not in spec:
                continue
            st.session_state[f"in_{k}"] = (int(v) if spec[k][1] == "int"
                                           else float(v))

    if "inputs_ready" not in st.session_state:
        apply_inputs(random_test_case(seed=7))
        st.session_state.inputs_ready = True

    st.markdown("##### Start from")
    c1, c2, c3 = st.columns([1, 1, 1.3])
    with c1:
        if st.button("🎲 Random test case", use_container_width=True):
            apply_inputs(random_test_case(seed=int(np.random.randint(1e6))))
            st.rerun()
    with c2:
        if st.button("↺ Reset to example", use_container_width=True):
            apply_inputs(random_test_case(seed=7))
            st.rerun()
    with c3:
        hist = base.dropna(subset=["age_this_week", "rain_mean_mm",
                                   "age_lag3"])
        opts = [(int(r.year), int(r.week)) for _, r in hist.iterrows()]
        pick = st.selectbox("…or load a real past week", opts,
                            index=max(0, len(opts) - 30),
                            format_func=lambda t: wk_label(*t))
        if st.button("Load this week", use_container_width=True):
            got, actual = case_from_history(pick[0], pick[1], horizon=1)
            if got:
                apply_inputs(got)
                st.session_state.loaded_actual = (pick, actual)
                st.rerun()
            else:
                st.warning("That week has incomplete history.")

    st.caption("The random generator samples case counts and rainfall from "
               "the ranges actually observed near that point in the calendar, "
               "so the test row is something the model could really be asked — "
               "not uniform noise.")
    if st.session_state.get("loaded_actual"):
        (py, pw), act = st.session_state["loaded_actual"]
        if act is not None:
            st.info(f"Loaded **{wk_label(py, pw)}**. What actually happened "
                    f"one week later: **{act:.0f} cases** — compare it with "
                    f"the 1-week forecast below.")

    st.markdown("##### Inputs")
    groups = {"When": ["year", "week"],
              "Recent disease counts": ["age_now", "age_lag1", "age_lag2",
                                        "age_lag3", "age_now_last_year",
                                        "age_target_week_last_year"],
              "Recent weather": ["rain_mean_mm", "rain_mean_lag1",
                                 "rain_mean_lag2", "rain_mean_lag3",
                                 "max_daily_mm", "temp_c_mean",
                                 "humidity_pct_mean"]}
    new = {}
    for gname, keys in groups.items():
        st.markdown(f"**{gname}**")
        cols = st.columns(min(3, len(keys)))
        for i, k in enumerate(keys):
            lab, typ, hlp = spec[k]
            with cols[i % len(cols)]:
                if typ == "int":
                    new[k] = st.number_input(
                        lab, min_value=1,
                        max_value=2100 if k == "year" else 52,
                        step=1, help=hlp, key=f"in_{k}")
                else:
                    new[k] = st.number_input(
                        lab, step=1.0, min_value=0.0, help=hlp,
                        key=f"in_{k}")

    st.markdown("---")
    st.markdown("##### Forecast")
    cols = st.columns(4)
    preds = {}
    for i, h in enumerate(HORIZONS):
        if h not in models:
            continue
        p, det = predict_one(new, h, models[h])
        preds[h] = (p, det)
        d = det["target_date"]
        cols[i].metric(
            f"{h} week(s) ahead",
            f"{p:,.0f}",
            f"± {det['mae']:.0f} typical error" if det["mae"] else None)
        cols[i].caption(
            f"week {det['target_week']}, {det['target_year']}"
            + (f" · ends {d:%d %b %Y}" if d else ""))

    if preds:
        fig, ax = plt.subplots(figsize=(9, 3.4))
        hist_x = [-3, -2, -1, 0]
        hist_y = [new["age_lag3"], new["age_lag2"], new["age_lag1"],
                  new["age_now"]]
        ax.plot(hist_x, hist_y, color=ROLE["age"], lw=2.2, marker="o",
                label="Recent observed cases")
        fx = [0] + list(preds)
        fy = [new["age_now"]] + [preds[h][0] for h in preds]
        ax.plot(fx, fy, color=ROLE["pred"], lw=2.2, ls=(0, (5, 2)), marker="o",
                label="Forecast")
        maes = [0] + [preds[h][1]["mae"] or 0 for h in preds]
        ax.fill_between(fx, np.array(fy) - np.array(maes),
                        np.array(fy) + np.array(maes), color=ROLE["pred"],
                        alpha=0.15, lw=0, label="± typical error")
        ax.axvline(0, color=GRID, lw=1.2)
        ax.annotate("now", xy=(0, ax.get_ylim()[1]), xytext=(3, -12),
                    textcoords="offset points", fontsize=9, color=MUTED,
                    va="top")
        ax.set_xticks(hist_x + list(preds))
        ax.set_xticklabels([f"{v:+d}" for v in hist_x]
                           + [f"+{h}" for h in preds])
        ax.set_xlabel("Weeks relative to now")
        ax.set_ylabel("AGE cases / week")
        ax.set_title("Forecast in context")
        ax.legend(loc="upper left", ncol=3)
        st.pyplot(fig, use_container_width=True)

        with st.expander("What the model did with these numbers"):
            rows = []
            for h, (p, det) in preds.items():
                rows.append({
                    "Horizon": f"{h} wk",
                    "Forecaster": det["model_name"],
                    "Target framings used": det["framing"],
                    "Members": det["n_members"],
                    "Spread across members": round(det["member_spread"], 1),
                    "Final forecast": round(p, 1)})
            st.dataframe(pd.DataFrame(rows).set_index("Horizon"),
                         use_container_width=True)
            st.markdown("""
The deployed forecaster is an **ensemble**: several models, each with its own
target framing, averaged together. *Target framing* is what a model actually
learned to predict. `level` predicts the case count directly; `logratio` the
**ratio** to this week's count; `ratio_ly` the ratio to the same week last
year; `ratio_blend` the ratio to the geometric mean of both. Ratio framings
matter because the reported series has quadrupled since 2021 — predicting a
*relative change* stays valid across that shift, and lets a forecast exceed any
level seen in training, which a tree predicting levels directly can never do.

*Spread across members* is how much the individual models disagree. A wide
spread is a signal to treat that particular forecast with more caution.
            """)

    with st.expander("What each input means"):
        st.table(pd.DataFrame(
            [{"Input": lab, "Meaning": hlp}
             for k, lab, typ, hlp in INPUT_SPEC if hlp]))


# ====================================================== 5. MODEL PERFORMANCE
elif page == "Model performance":
    st.subheader("How well does it actually work?")
    if not models:
        st.warning("No trained models found.")
        st.stop()

    st.markdown("""
**Protocol.** Everything — model family, target framing, how much history to
keep, hyperparameters — was chosen by cross-validation *inside 2021–2023 only*.
The whole of **2024–2025 was never consulted for any decision** and is reported
here as a single untouched hold-out, evaluated by rolling origin: retrain on all
data up to week *t*, forecast week *t+h*, step forward one week, repeat.
    """)

    rows = []
    for h, b in models.items():
        rows.append({
            "Horizon": f"{h} week(s)",
            "Model": (f"Ensemble of {b['k']}" if "members" in b
                      else b.get("name")),
            "Framing": ("mixed" if "members" in b else b.get("framing")),
            "History": ("mixed" if "members" in b else b.get("scheme", "-")),
            "MAE": round(b["test_MAE"], 1),
            "RMSE": round(b["test_RMSE"], 1),
            "MAPE %": round(b["test_MAPE"], 1),
            "MAE 2024": round(b.get("test2024_MAE", np.nan), 1),
            "MAE 2025": round(b.get("test2025_MAE", np.nan), 1),
            "Best baseline": b["best_baseline"].replace("Baseline: ", ""),
            "Baseline MAE": round(b["best_baseline_MAE"], 1),
            "Improvement %": round(b["improvement_pct"], 1)})
    perf = pd.DataFrame(rows).set_index("Horizon")
    st.dataframe(perf, use_container_width=True)

    if backtests:
        h = st.selectbox("Show forecast-vs-actual for horizon", HORIZONS,
                         format_func=lambda x: f"{x} week(s) ahead")
        bt = backtests[h].copy()
        bt["x"] = (bt["target_year"] - 2021) * 52 + bt["target_week"]
        bt = bt.sort_values("x")
        fig, ax = plt.subplots(figsize=(11, 3.8))
        gx, gt, gp, gb = break_gaps(bt["x"].values, bt["target"].values,
                                    bt["pred"].values,
                                    bt["base_seasonal_naive"].values)
        ax.plot(gx, gt, color=ROLE["actual"], lw=2.2, label="Actual")
        ax.plot(gx, gp, color=ROLE["pred"], lw=2.0, ls=(0, (5, 2)),
                label="MonsoonWatch")
        ax.plot(gx, gb, color=ROLE["baseline"], lw=1.3, alpha=0.7,
                label="Seasonal naive")
        ax.set_ylabel("AGE cases / week")
        ax.set_title(f"Hold-out period, {h} week(s) ahead")
        ax.legend(ncol=3, loc="upper left")
        yrs = sorted(bt["target_year"].unique())
        ax.set_xticks([(y - 2021) * 52 + 1 for y in yrs])
        ax.set_xticklabels([str(int(y)) for y in yrs])
        st.pyplot(fig, use_container_width=True)

    st.markdown("##### Why an ensemble rather than one best model")
    st.markdown("""
Single-model selection proved unreliable here, and that was established from
**2021–2023 data alone**: the cross-validation folds barely agree on which
candidate is best (Kendall rank correlation between folds is only +0.07 to
+0.22). With three years of weekly data there is not enough signal to identify
a single winner, so any single pick is largely luck. Averaging the top-*k*
candidates is the standard remedy — it cannot pick the best model, but it
reliably avoids picking a bad one. The value of *k* was chosen by
leave-one-fold-out inside 2021–2023, so the test years played no part.
    """)
    if os.path.exists("reports/metrics/ensemble_selection.csv"):
        es = pd.read_csv("reports/metrics/ensemble_selection.csv")
        st.dataframe(es[["horizon", "k", "loo_cv_MAE", "test_MAE",
                         "test2024_MAE", "test2025_MAE", "best_baseline",
                         "best_baseline_MAE", "improvement_pct"]],
                     use_container_width=True)

    if comparison is not None:
        st.markdown("##### Every candidate that was tried")
        st.caption("Selection used the `cv_MAE` column only. The test columns "
                   "are shown for transparency and played no part in choosing "
                   "the winner.")
        h2 = st.selectbox("Horizon", HORIZONS, key="cmp",
                          format_func=lambda x: f"{x} week(s) ahead")
        c = comparison[comparison.horizon == h2].copy()
        cols = [x for x in ["model", "framing", "scheme", "blend_w", "cv_MAE",
                            "test_MAE", "test_RMSE", "test_MAPE",
                            "test2024_MAE", "test2025_MAE", "frozen_test_MAE"]
                if x in c.columns]
        sort_col = next((x for x in ("test_MAE", "roll_test_MAE", "cv_MAE")
                         if x in cols), None)
        view = c[cols].sort_values(sort_col) if sort_col else c[cols]
        st.dataframe(view.round(2), use_container_width=True, height=380)

    st.markdown("##### Does rainfall actually help?")
    st.markdown("""
The premise of the project is that rainfall carries predictive signal, so it
was tested directly: the same ensemble, the same hold-out weeks, the same
protocol — just different feature sets.
    """)
    if os.path.exists("reports/metrics/ablation.csv"):
        ab = pd.read_csv("reports/metrics/ablation.csv")
        ab2 = ab.rename(columns={
            "full (deployed)": "Full model MAE",
            "no weather": "Disease + season only",
            "weather + season only": "Weather + season only",
            "rainfall contribution": "Rainfall contribution %"})
        st.dataframe(ab2.set_index("horizon"), use_container_width=True)
    st.error("**Removing every rainfall and climate feature changes accuracy "
             "by less than 3% at every horizon — sometimes slightly for the "
             "better.** At the national weekly scale, with seasonal terms "
             "already in the model, this rainfall series adds no measurable "
             "predictive value.")
    st.markdown("""
Three explanations fit the evidence, and they point at different fixes:

1. **Spatial mismatch — most likely.** Rainfall is measured in the Kathmandu
   Valley; cases are counted nationally. This is a data-coverage problem, not
   proof that rainfall is epidemiologically irrelevant.
2. **Seasonality already encodes the monsoon.** The model knows the week of the
   year, so the ablation measures what rainfall adds *on top of knowing the
   date* — and that increment is near zero.
3. **The underlying correlation is weak anyway** (0.26, peaking at lag 0 rather
   than at the multi-week lag a contamination mechanism would imply).

What this does not undermine: the forecasting system works and beats every
operational baseline at every horizon. What it changes is the honest claim —
this is an early-warning system built on surveillance history and seasonality,
which rainfall does not currently improve at this scale.
    """)

    st.markdown("##### Honest limitations")
    st.markdown("""
- **National only.** EWARS district columns are populated for 2024–2025 only,
  which is far too short for a model that needs multi-year seasonality. The
  bulletins name a top-5 district list; this project does not forecast a
  district grid, and the pitch should not claim it does.
- **The reported series quadrupled since 2021** (116 → 495 cases/week). Much of
  that is almost certainly EWARS expanding its reporting network rather than a
  real fourfold rise in disease. Every model here therefore forecasts the
  *reported* count, which is what an operational responder acts on, but the
  trend must not be read as pure epidemiology.
- **Rainfall for 2024–2025 is estimated**, not measured: DHM gauge data ends
  Dec 2023, and satellite data calibrated against the gauges (R² = 0.87) covers
  the rest. Weeks using estimated rainfall are flagged in the data.
- **Rainfall is Kathmandu Valley only, and adds no measurable accuracy** — see
  the ablation above. Treat the climate–health link as motivation and future
  work, not as a demonstrated driver of this model's performance.
- **Model selection is noisy.** Cross-validated error on 2021–2023 correlates
  only weakly with hold-out error (Kendall τ between CV folds is +0.07 to
  +0.22). That is why the deployed forecaster averages many models rather than
  trusting a single "best" one.
- **A strong seasonal baseline is genuinely hard to beat** in some years. In
  2025 specifically, "same week last year" was an unusually good forecast
  because growth had flattened. This is reported per-year rather than hidden in
  an average.
    """)
