"""
Figures for the report and the web app.

Usage:  python src/reporting/make_figures.py eda    # data only, no models
        python src/reporting/make_figures.py model  # needs backtest_h*.csv
        python src/reporting/make_figures.py all

Charting rules followed throughout (see src/common/viz_style.py):
  * NO dual-axis charts. Rainfall and cases are stacked panels on a shared
    x-axis instead -- two y-scales can manufacture any apparent lead/lag just
    by rescaling, which would undermine the exact claim being tested.
  * Colour marks an ENTITY (AGE, rainfall, model, baseline), never a rank.
  * A legend whenever two or more series share a frame.
"""

import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from src.common.viz_style import (  # noqa
    apply_style, ROLE, SERIES, C, INK2, MUTED, GRID, break_gaps)

FIG = "reports/figures"
apply_style()
os.makedirs(FIG, exist_ok=True)

MONSOON = (22, 39)


def _idx(df):
    """Continuous week index for plotting on a real timeline."""
    return (df["year"] - 2021) * 52 + df["week"]


def _year_ticks(ax, df):
    ticks, labels = [], []
    for y in sorted(df["year"].unique()):
        ticks.append((y - 2021) * 52 + 1)
        labels.append(str(int(y)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _shade_monsoon(ax, df, label=True):
    done = False
    for y in sorted(df["year"].unique()):
        x0 = (y - 2021) * 52 + MONSOON[0]
        x1 = (y - 2021) * 52 + MONSOON[1]
        ax.axvspan(x0, x1, color=C["aqua"], alpha=0.07, lw=0,
                   label="Monsoon (wk 22-39)" if (label and not done) else None)
        done = True


# =============================================================== EDA FIGURES
def fig_age_series(base):
    d = base.dropna(subset=["age_this_week"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _shade_monsoon(ax, d)
    ax.plot(_idx(d), d["age_this_week"], color=ROLE["age"], lw=2,
            label="Weekly AGE cases")
    for y in sorted(d.year.unique()):
        m = d[d.year == y]["age_this_week"].mean()
        x0, x1 = (y - 2021) * 52 + 1, (y - 2021) * 52 + 52
        ax.plot([x0, x1], [m, m], color=MUTED, lw=1.4, ls=(0, (4, 3)),
                label="Annual mean" if y == 2021 else None)
        ax.annotate(f"{m:.0f}", xy=(x1 - 2, m), fontsize=8.5, color=MUTED,
                    va="bottom", ha="right")
    _year_ticks(ax, d)
    ax.set_ylabel("AGE cases per week")
    ax.set_title("Nepal weekly acute gastroenteritis, 2021-2025 (EWARS national)")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/01_age_series.png")
    plt.close(fig)


def fig_stacked_rain_age(base):
    d = base.dropna(subset=["age_this_week"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    _shade_monsoon(axes[0], d)
    axes[0].plot(_idx(d), d["age_this_week"], color=ROLE["age"], lw=2)
    axes[0].set_ylabel("AGE cases / week")
    axes[0].set_title("Disease and rainfall share a season - but not a peak")
    axes[0].legend(handles=[Line2D([], [], color=ROLE["age"], lw=2,
                                   label="AGE cases")], loc="upper left")

    _shade_monsoon(axes[1], d, label=False)
    dd = d.dropna(subset=["rain_mean_mm"])
    gauge = dd[dd.rain_source_is_dhm == 1]
    est = dd[dd.rain_source_is_dhm == 0]
    axes[1].fill_between(_idx(gauge), gauge["rain_mean_mm"], color=ROLE["rain"],
                         alpha=0.85, lw=0)
    axes[1].fill_between(_idx(est), est["rain_mean_mm"], color=ROLE["rain"],
                         alpha=0.32, lw=0, hatch="////", edgecolor=ROLE["rain"])
    axes[1].set_ylabel("Rainfall mm / week")
    axes[1].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=ROLE["rain"], alpha=0.85,
                      label="DHM gauges (23 stations)"),
        plt.Rectangle((0, 0), 1, 1, fc=ROLE["rain"], alpha=0.32, hatch="////",
                      ec=ROLE["rain"], label="NASA POWER, calibrated estimate")],
        loc="upper left", ncol=2)
    _year_ticks(axes[1], d)
    fig.tight_layout()
    fig.savefig(f"{FIG}/02_rain_vs_age_panels.png")
    plt.close(fig)


def fig_seasonal_profile(base):
    d = base.dropna(subset=["age_this_week"])
    # Normalise each year to its own mean so the 4x level trend doesn't swamp
    # the seasonal shape we actually want to see.
    d = d.copy()
    d["age_rel"] = d.groupby("year")["age_this_week"].transform(
        lambda s: s / s.mean())
    a = d.groupby("week")["age_rel"].mean()
    r = d.groupby("week")["rain_mean_mm"].mean()

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    axes[0].axvspan(*MONSOON, color=C["aqua"], alpha=0.07, lw=0)
    axes[0].plot(a.index, a.values, color=ROLE["age"], lw=2)
    pa = int(a.idxmax())
    axes[0].scatter([pa], [a.max()], color=ROLE["age"], zorder=5, s=38)
    axes[0].annotate(f"AGE peaks wk {pa}", xy=(pa, a.max()),
                     xytext=(pa + 2, a.max()), color=ROLE["age"], fontsize=9)
    axes[0].axhline(1.0, color=GRID, lw=1)
    axes[0].set_ylabel("AGE, relative to\nthat year's mean")
    axes[0].set_title("AGE rises BEFORE peak rainfall - onset matters more "
                      "than volume")

    axes[1].axvspan(*MONSOON, color=C["aqua"], alpha=0.07, lw=0)
    axes[1].fill_between(r.index, r.values, color=ROLE["rain"], alpha=0.75,
                         lw=0)
    pr = int(r.idxmax())
    axes[1].annotate(f"rain peaks wk {pr}", xy=(pr, r.max()),
                     xytext=(pr + 2, r.max() * 0.92), color=C["aqua"],
                     fontsize=9)
    axes[1].set_ylabel("Mean rainfall\nmm / week")
    axes[1].set_xlabel("EDCD epidemiological week")
    axes[1].annotate(f"{pr - pa}-week offset", xy=((pa + pr) / 2, r.max() * 0.46),
                     ha="center", fontsize=9, color=MUTED)
    axes[1].annotate("", xy=(pa, r.max() * 0.36), xytext=(pr, r.max() * 0.36),
                     arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
    fig.tight_layout()
    fig.savefig(f"{FIG}/03_seasonal_profile.png")
    plt.close(fig)


def fig_crosscorr(base):
    d = base.dropna(subset=["age_this_week", "rain_mean_mm"]).copy()
    lags = range(0, 9)
    vals = []
    for L in lags:
        vals.append(d["age_this_week"].corr(d["rain_mean_mm"].shift(L)))
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    cols = [ROLE["rain"] if v == max(vals) else "#cfd6da" for v in vals]
    bars = ax.bar(list(lags), vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK2)
    ax.set_xlabel("Rainfall lag (weeks before the case count)")
    ax.set_ylabel("Correlation with AGE")
    ax.set_title("Rainfall correlates with AGE most strongly at lag 0")
    ax.axhline(0, color=GRID, lw=1)
    fig.tight_layout()
    fig.savefig(f"{FIG}/04_rain_age_crosscorr.png")
    plt.close(fig)


def fig_calibration():
    cal = pd.read_csv("reports/metrics/climate_bridge_calibration.csv")
    dhm = pd.read_csv("data/interim/rainfall_weekly_dhm.csv")
    nasa = pd.read_csv("data/interim/nasa_power_weekly.csv")
    ov = dhm.merge(nasa, on=["year", "week"])
    ov = ov[ov.year.between(2021, 2023)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.scatter(ov["nasa_rain_sum"], ov["rain_mean_mm"], s=22,
               color=ROLE["dhm"], alpha=0.7, edgecolor="white", linewidth=0.6)
    lim = max(ov["nasa_rain_sum"].max(), ov["rain_mean_mm"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls=(0, (4, 3)),
            label="1:1 line")
    z = np.polyfit(ov["nasa_rain_sum"], ov["rain_mean_mm"], 1)
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, np.polyval(z, xs), color=ROLE["pred"], lw=2,
            label=f"fit: y = {z[0]:.2f}x + {z[1]:.1f}")
    r = ov["rain_mean_mm"].corr(ov["nasa_rain_sum"])
    ax.set_xlabel("NASA POWER weekly rainfall (mm)")
    ax.set_ylabel("DHM 23-station mean (mm)")
    ax.set_title(f"Bridge calibration, 2021-2023  (r = {r:.2f}, "
                 f"R² = {r ** 2:.2f})")
    ax.legend(loc="upper left")

    ax = axes[1]
    cal = cal.sort_values("r2_holdout_2023")
    ypos = np.arange(len(cal))
    ax.barh(ypos, cal["r2_holdout_2023"], color=ROLE["dhm"], height=0.55)
    for i, v in enumerate(cal["r2_holdout_2023"]):
        ax.annotate(f"{v:.2f}", xy=(v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(cal["dhm_feature"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("R² on held-out 2023")
    ax.set_title("Each rainfall feature, reconstructed from satellite")
    fig.tight_layout()
    fig.savefig(f"{FIG}/05_climate_bridge.png")
    plt.close(fig)


def fig_stations():
    st = pd.read_csv("data/interim/rainfall_stations.csv")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sc = ax.scatter(st["Longitude"], st["Latitude"], c=st["Elevation"],
                    cmap="YlGnBu", s=110, edgecolor="white", linewidth=1.1,
                    zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82)
    cb.set_label("Elevation (m)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{len(st)} DHM rain gauges, Kathmandu Valley")
    ax.scatter([85.3240], [27.7172], marker="*", s=260, color=ROLE["pred"],
               edgecolor="white", linewidth=1, zorder=4)
    ax.annotate("NASA POWER\ngrid point", xy=(85.3240, 27.7172),
                xytext=(85.30, 27.66), fontsize=8.5, color=ROLE["pred"])
    fig.tight_layout()
    fig.savefig(f"{FIG}/06_station_map.png")
    plt.close(fig)


def fig_level_shift(base):
    d = base.dropna(subset=["age_this_week"])
    m = d.groupby("year")["age_this_week"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax = axes[0]
    bars = ax.bar([str(int(y)) for y in m.index], m.values, color=ROLE["age"],
                  width=0.6)
    for b, v in zip(bars, m.values):
        ax.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, color=INK2)
    ax.set_ylabel("Mean AGE cases / week")
    ax.set_title("Reported level rises 4x in five years")

    ax = axes[1]
    ratio = (m / m.shift(1)).dropna()
    ax.plot([str(int(y)) for y in ratio.index], ratio.values,
            color=ROLE["pred"], marker="o", lw=2)
    for x, v in zip([str(int(y)) for y in ratio.index], ratio.values):
        ax.annotate(f"{v:.2f}x", xy=(x, v), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK2)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.set_ylim(0.8, 2.35)
    ax.set_ylabel("Year-on-year ratio")
    ax.set_title("...but the growth rate is decaying toward 1.0")
    fig.tight_layout()
    fig.savefig(f"{FIG}/07_level_shift.png")
    plt.close(fig)


def fig_coverage(base):
    d = base.copy()
    fig, ax = plt.subplots(figsize=(10, 2.2))
    have = d["age_this_week"].notna()
    ax.scatter(_idx(d)[have], np.ones(have.sum()), marker="|", s=180,
               color=ROLE["age"], linewidth=1.6, label="EWARS bulletin present")
    ax.scatter(_idx(d)[~have], np.ones((~have).sum()), marker="|", s=180,
               color=C["red"], linewidth=1.6, label="bulletin missing")
    g = d["rain_source_is_dhm"] == 1
    ax.scatter(_idx(d)[g], np.zeros(g.sum()), marker="|", s=180,
               color=ROLE["dhm"], linewidth=1.6, label="DHM gauge rainfall")
    e = d["rain_source_is_dhm"] == 0
    ax.scatter(_idx(d)[e], np.zeros(e.sum()), marker="|", s=180,
               color=ROLE["nasa"], linewidth=1.6, label="estimated rainfall")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Rainfall", "Disease"])
    ax.set_ylim(-0.6, 1.6)
    _year_ticks(ax, d)
    ax.grid(axis="y", visible=False)
    ax.set_title("Data coverage by week")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/08_coverage.png")
    plt.close(fig)


def run_eda():
    base = pd.read_csv("data/processed/features_base.csv")
    fig_age_series(base)
    fig_stacked_rain_age(base)
    fig_seasonal_profile(base)
    fig_crosscorr(base)
    fig_calibration()
    fig_stations()
    fig_level_shift(base)
    fig_coverage(base)
    print("[figures] EDA figures written to", FIG)




# ============================================================ MODEL FIGURES
def fig_forecast_vs_actual():
    hs = [h for h in (1, 2, 3, 4)
          if os.path.exists(f"reports/metrics/backtest_h{h}.csv")]
    if not hs:
        return
    fig, axes = plt.subplots(len(hs), 1, figsize=(10, 2.5 * len(hs)),
                             sharex=True, gridspec_kw={"hspace": 0.28})
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs):
        bt = pd.read_csv(f"reports/metrics/backtest_h{h}.csv")
        bt["x"] = (bt["target_year"] - 2021) * 52 + bt["target_week"]
        bt = bt.sort_values("x")
        gx, gt, gp = break_gaps(bt["x"].values, bt["target"].values,
                                bt["pred"].values)
        ax.plot(gx, gt, color=ROLE["actual"], lw=2, label="Actual", zorder=3)
        ax.plot(gx, gp, color=ROLE["pred"], lw=2, ls=(0, (5, 2)),
                label="MonsoonWatch forecast", zorder=4)
        ax.fill_between(gx, gt, gp, color=ROLE["pred"], alpha=0.12, lw=0)
        mae = (bt["pred"] - bt["target"]).abs().mean()
        ax.set_title(f"{h} week(s) ahead   ·   MAE {mae:.0f} cases/week")
        ax.set_ylabel("AGE cases")
        if h == hs[0]:
            ax.legend(loc="upper left", ncol=2)
    yrs = sorted(bt["target_year"].unique())
    axes[-1].set_xticks([(y - 2021) * 52 + 1 for y in yrs]
                        + [(y - 2021) * 52 + 27 for y in yrs])
    axes[-1].set_xticklabels([f"{int(y)}" for y in yrs]
                             + [f"mid {int(y)}" for y in yrs], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/09_forecast_vs_actual.png")
    plt.close(fig)


def fig_model_comparison():
    p = "reports/metrics/model_comparison.csv"
    if not os.path.exists(p):
        return
    cmp = pd.read_csv(p)
    ens = (pd.read_csv("reports/metrics/ensemble_selection.csv")
           if os.path.exists("reports/metrics/ensemble_selection.csv") else None)
    hs = sorted(cmp.horizon.unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(3.1 * len(hs), 4.6),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs):
        c = cmp[cmp.horizon == h].copy()
        base = c[c.framing == "-"][["model", "test_MAE"]]
        mods = c[c.framing != "-"]
        best = (mods.groupby("model")["test_MAE"].min()
                .sort_values().reset_index())
        labels = (list(best["model"])
                  + [m.replace("Baseline: ", "") for m in base["model"]])
        vals = list(best["test_MAE"]) + list(base["test_MAE"])
        cols = [C["blue"]] * len(best) + [ROLE["baseline"]] * len(base)
        if ens is not None and h in set(ens.horizon):
            e = ens[ens.horizon == h].iloc[0]
            labels.append(f"ENSEMBLE (deployed, k={int(e.k)})")
            vals.append(float(e.test_MAE))
            cols.append(ROLE["pred"])
        order = np.argsort(vals)
        labels = [labels[i] for i in order]
        vals = [vals[i] for i in order]
        cols = [cols[i] for i in order]
        y = np.arange(len(vals))
        ax.barh(y, vals, color=cols, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f}", xy=(v, i), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=INK2)
        ax.set_title(f"{h} week(s) ahead", fontsize=10)
        ax.set_xlabel("Hold-out MAE" if h == hs[-1] else "")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=ROLE["pred"],
                             label="Deployed ensemble"),
               plt.Rectangle((0, 0), 1, 1, fc=C["blue"],
                             label="Best single model per family (optimistic: "
                                   "best-of-many on the hold-out)"),
               plt.Rectangle((0, 0), 1, 1, fc=ROLE["baseline"],
                             label="Baselines")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("What was tried, and what is deployed (lower is better)",
                 x=0.01, ha="left", fontsize=12, weight="600")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(f"{FIG}/10_model_comparison.png")
    plt.close(fig)


def fig_feature_importance():
    def _imp(m):
        if hasattr(m, "feature_importances_"):
            return np.asarray(m.feature_importances_, float)
        if hasattr(m, "steps"):
            est = m.steps[-1][1]
            if hasattr(est, "coef_"):
                return np.abs(np.asarray(est.coef_, float).ravel())
        return None

    rows = []
    for h in (1, 2, 3, 4):
        p = (f"models/ensemble_h{h}.pkl"
             if os.path.exists(f"models/ensemble_h{h}.pkl")
             else f"models/model_h{h}.pkl")
        if not os.path.exists(p):
            continue
        b = joblib.load(p)
        feats = b["features"]
        # Average importance across every ensemble member, each normalised
        # first so a model with large raw coefficients cannot dominate.
        members = b.get("members") or [{"model": b["model"]}]
        acc = []
        for mem in members:
            imp = _imp(mem["model"])
            if imp is None or not np.isfinite(imp).all() or imp.sum() == 0:
                continue
            acc.append(imp / imp.sum())
        if not acc:
            continue
        imp = np.mean(np.vstack(acc), axis=0)
        for f, v in zip(feats, imp):
            rows.append({"horizon": h, "feature": f, "importance": v})
    if not rows:
        return
    df = pd.DataFrame(rows)
    agg = (df.groupby("feature")["importance"].mean()
           .sort_values(ascending=False).head(14)[::-1])

    def fam(f):
        if f.startswith("age"):
            return ROLE["age"]
        if f.startswith("rain") or f in ("max_daily_mm",):
            return ROLE["rain"]
        if f.startswith("tgt"):
            return C["yellow"]
        return C["magenta"]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(agg))
    ax.barh(y, agg.values, color=[fam(f) for f in agg.index], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=9)
    for i, v in enumerate(agg.values):
        ax.annotate(f"{v * 100:.1f}%", xy=(v, i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
    ax.set_xlabel("Mean relative importance, averaged over all\nensemble members and all four horizons")
    ax.set_title("What the models actually lean on")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=ROLE["age"], label="Disease history"),
               plt.Rectangle((0, 0), 1, 1, fc=ROLE["rain"], label="Rainfall"),
               plt.Rectangle((0, 0), 1, 1, fc=C["yellow"], label="Seasonality"),
               plt.Rectangle((0, 0), 1, 1, fc=C["magenta"], label="Other climate")]
    ax.legend(handles=handles, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/11_feature_importance.png")
    plt.close(fig)


def fig_error_by_season():
    hs = [h for h in (1, 2, 3, 4)
          if os.path.exists(f"reports/metrics/backtest_h{h}.csv")]
    if not hs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels = None
    for i, h in enumerate(hs):
        bt = pd.read_csv(f"reports/metrics/backtest_h{h}.csv")
        bt["err"] = (bt["pred"] - bt["target"]).abs()
        bt["bin"] = pd.cut(bt["target_week"], bins=[0, 13, 21, 39, 52],
                           labels=["Wk 1-13\nwinter", "Wk 14-21\npre-monsoon",
                                   "Wk 22-39\nmonsoon", "Wk 40-52\npost"])
        g = bt.groupby("bin", observed=True)["err"].mean()
        n = bt.groupby("bin", observed=True)["err"].size()
        ax.plot(range(len(g)), g.values, marker="o", lw=2,
                color=SERIES[i], label=f"{h} wk ahead")
        if labels is None:
            labels = [f"{ix}\n(n={v})" for ix, v in zip(g.index, n.values)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean absolute error (cases)")
    ax.set_title("Error is largest in the monsoon build-up")
    ax.legend(ncol=2, fontsize=8.5)

    # Right: model vs best baseline, as horizon grows
    ax = axes[1]
    maes, bmaes = [], []
    cmp = pd.read_csv("reports/metrics/model_comparison.csv")
    for h in hs:
        bt = pd.read_csv(f"reports/metrics/backtest_h{h}.csv")
        maes.append((bt["pred"] - bt["target"]).abs().mean())
        b = cmp[(cmp.horizon == h) & (cmp.framing == "-")]
        bmaes.append(b["test_MAE"].min())
    x = np.arange(len(hs))
    ax.plot(x, maes, marker="o", lw=2.2, color=ROLE["pred"],
            label="MonsoonWatch")
    ax.plot(x, bmaes, marker="o", lw=2.2, color=ROLE["baseline"],
            label="Best baseline at that horizon")
    for xi, (m, b) in enumerate(zip(maes, bmaes)):
        ax.annotate(f"{m:.0f}", xy=(xi, m), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK2)
        ax.annotate(f"{b:.0f}", xy=(xi, b), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} wk" for h in hs])
    ax.set_ylabel("Hold-out MAE (cases)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("The advantage grows with lead time")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, max(bmaes) * 1.22)
    fig.tight_layout()
    fig.savefig(f"{FIG}/12_error_by_season.png")
    plt.close(fig)


def run_model():
    fig_forecast_vs_actual()
    fig_model_comparison()
    fig_feature_importance()
    fig_error_by_season()
    print("[figures] model figures written to", FIG)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("eda", "all"):
        run_eda()
    if what in ("model", "all"):
        run_model()
