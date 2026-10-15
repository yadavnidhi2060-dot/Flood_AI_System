"""
Exploratory data analysis + visualisation (Pandas, Matplotlib, Seaborn).

Every plot function takes a DataFrame and returns a Matplotlib Figure, so the same code is used
by the command line (saves PNGs to reports/figures) and by the Streamlit dashboard.

Usage:  python -m src.eda
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.evaluate import plot_confusion_matrix

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
PALETTE = ["#0b6e99", "#e07a1f", "#4c9f70", "#8e5ea2", "#c0392b"]
RISK_PAL = config.RISK_COLORS
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fig(w=9, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rainfall_mm", "river_level_m", "temperature_c", "humidity_pct"]
    return df[cols].describe().T.round(2)


def plot_missing_values(raw: pd.DataFrame):
    """Share of missing values per column in the RAW data (before cleaning)."""
    cols = ["rainfall_mm", "river_level_m", "temperature_c", "humidity_pct"]
    pct = raw[cols].isna().mean().mul(100).sort_values()
    fig, ax = _fig(7, 3)
    ax.barh(pct.index, pct.values, color=PALETTE[0])
    for i, v in enumerate(pct.values):
        ax.text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing data in raw dataset", fontweight="bold")
    ax.set_xlim(0, max(pct.max() * 1.25, 1))
    fig.tight_layout()
    return fig


def plot_rainfall_trend(df: pd.DataFrame):
    """Monthly rainfall totals per station with a 12-month rolling mean."""
    m = (df.set_index("date").groupby("station")["rainfall_mm"].resample("ME").sum()
         .reset_index())
    fig, ax = _fig(10, 4.4)
    for i, (st, g) in enumerate(m.groupby("station")):
        ax.plot(g["date"], g["rainfall_mm"], lw=1.1, alpha=0.75, color=PALETTE[i % 5], label=st)
    tot = m.groupby("date")["rainfall_mm"].mean()
    ax.plot(tot.index, tot.rolling(12, min_periods=6).mean(), color="black", lw=2.2,
            label="12-month rolling mean (all stations)")
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.set_xlabel("")
    ax.set_title("Rainfall trend", fontweight="bold")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def plot_monthly_rainfall(df: pd.DataFrame):
    """Average rainfall per calendar month (seasonality)."""
    d = df.copy()
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year
    monthly = d.groupby(["station", "year", "month"])["rainfall_mm"].sum().reset_index()
    avg = monthly.groupby("month")["rainfall_mm"].mean().reindex(range(1, 13)).fillna(0)
    fig, ax = _fig(8, 3.8)
    colors = [PALETTE[0] if v < avg.max() * 0.6 else PALETTE[1] for v in avg.values]
    ax.bar(MONTHS, avg.values, color=colors)
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax.set_title("Seasonal rainfall pattern", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_water_levels(df: pd.DataFrame):
    """River level time-series with warning / danger lines (one panel per station)."""
    stations = sorted(df["station"].unique())
    fig, axes = plt.subplots(len(stations), 1, figsize=(10, 2.9 * len(stations)), sharex=True,
                             squeeze=False)
    for ax, st in zip(axes[:, 0], stations):
        g = df[df["station"] == st]
        ax.plot(g["date"], g["river_level_m"], color=PALETTE[0], lw=0.9)
        w, dgr = g["warning_level_m"].iloc[0], g["danger_level_m"].iloc[0]
        ax.axhline(w, color=config.RISK_COLORS["Medium"], ls="--", lw=1.3, label=f"Warning ({w:.1f} m)")
        ax.axhline(dgr, color=config.RISK_COLORS["High"], ls="--", lw=1.3, label=f"Danger ({dgr:.1f} m)")
        fl = g[g["flood_occurred"] == 1]
        ax.scatter(fl["date"], fl["river_level_m"], s=9, color=config.RISK_COLORS["High"], zorder=3,
                   label="Flood recorded")
        ax.set_ylabel("Level (m)")
        ax.set_title(st, fontsize=10, loc="left", fontweight="bold")
        ax.legend(fontsize=7.5, ncol=3, loc="upper left", frameon=False)
    fig.suptitle("River water levels", fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


def plot_flood_frequency(df: pd.DataFrame):
    """Flood days per year (stacked by station) and flood days per month."""
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    by_year = d.pivot_table(index="year", columns="station", values="flood_occurred", aggfunc="sum").fillna(0)
    by_month = d.groupby("month")["flood_occurred"].sum().reindex(range(1, 13)).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    by_year.plot(kind="bar", stacked=True, ax=axes[0], color=PALETTE[: by_year.shape[1]], width=0.8)
    axes[0].set_title("Flood days per year", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Flood days")
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].bar(MONTHS, by_month.values, color=PALETTE[4])
    axes[1].set_title("Flood days by month", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_risk_distribution(df: pd.DataFrame):
    """Number of days in each risk class (class imbalance check)."""
    counts = df["risk_today"].value_counts().reindex(config.RISK_LABELS).fillna(0)
    fig, ax = _fig(5.5, 3.6)
    bars = ax.bar(counts.index, counts.values, color=[RISK_PAL[k] for k in counts.index])
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}\n({v / total:.1%})", ha="center",
                va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Days")
    ax.set_title("Flood-risk class distribution", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame):
    cols = ["rainfall_mm", "rain_7d_mm", "river_level_m", "temperature_c", "humidity_pct", "flood_occurred"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = _fig(6.5, 5)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.8, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation between variables", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_rain_vs_level(df: pd.DataFrame):
    """Rainfall of the last 7 days vs river level, coloured by risk."""
    d = df.sample(min(len(df), 4000), random_state=1)
    fig, ax = _fig(7.5, 4.4)
    sns.scatterplot(data=d, x="rain_7d_mm", y="river_level_m", hue="risk_today",
                    hue_order=config.RISK_LABELS, palette=RISK_PAL, s=16, alpha=0.7,
                    linewidth=0, ax=ax)
    ax.set_xlabel("Rainfall over previous 7 days (mm)")
    ax.set_ylabel("River level (m)")
    ax.set_title("Rainfall vs river level", fontweight="bold")
    ax.legend(title="Risk", frameon=False)
    fig.tight_layout()
    return fig


def plot_weather_by_risk(df: pd.DataFrame):
    """Box-plots of temperature and humidity for each risk level."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, lab in zip(axes, ["temperature_c", "humidity_pct"], ["Temperature (°C)", "Humidity (%)"]):
        sns.boxplot(data=df, x="risk_today", y=col, order=config.RISK_LABELS, hue="risk_today",
                    palette=RISK_PAL, legend=False, ax=ax, fliersize=2, linewidth=1)
        ax.set_xlabel("Risk level")
        ax.set_ylabel(lab)
    fig.suptitle("Weather conditions by risk level", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(imp: pd.DataFrame, column="rf_importance", title="Feature importance"):
    d = imp.sort_values(column, ascending=True).copy()
    d["label"] = d["feature"].map(config.FEATURE_LABELS).fillna(d["feature"])
    fig, ax = _fig(8, 4.6)
    ax.barh(d["label"], d[column], color=PALETTE[0])
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics: dict):
    rows = []
    for name, m in metrics["models"].items():
        for k, lab in [("accuracy", "Accuracy"), ("precision_macro", "Precision"),
                       ("recall_macro", "Recall"), ("f1_macro", "F1-score")]:
            rows.append({"Model": name, "Metric": lab, "Score": m[k]})
    d = pd.DataFrame(rows)
    fig, ax = _fig(9, 4)
    sns.barplot(data=d, x="Metric", y="Score", hue="Model", palette=PALETTE[:3], ax=ax)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("")
    ax.set_title("Model comparison on the held-out test period (macro-averaged)", fontweight="bold")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7.5, padding=2)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.tight_layout()
    return fig


def save_all() -> None:
    """Create every figure and save it to reports/figures/."""
    from src.data_loader import load_raw
    from src.preprocessing import add_features

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw, tag = load_raw()
    clean = pd.read_csv(config.CLEAN_FILE, parse_dates=["date"])
    feat = add_features(clean)
    figs = {
        "01_missing_values": plot_missing_values(raw),
        "02_rainfall_trend": plot_rainfall_trend(clean),
        "03_seasonal_rainfall": plot_monthly_rainfall(clean),
        "04_river_levels": plot_water_levels(clean),
        "05_flood_frequency": plot_flood_frequency(clean),
        "06_risk_distribution": plot_risk_distribution(clean),
        "07_correlation": plot_correlation(feat),
        "08_rain_vs_level": plot_rain_vs_level(feat),
        "09_weather_by_risk": plot_weather_by_risk(clean),
    }
    if config.IMPORTANCE_FILE.exists():
        imp = pd.read_csv(config.IMPORTANCE_FILE)
        figs["10_feature_importance"] = plot_feature_importance(imp, "rf_importance",
                                                                 "Random Forest feature importance")
    if config.METRICS_FILE.exists():
        with open(config.METRICS_FILE) as f:
            metrics = json.load(f)
        figs["11_model_comparison"] = plot_model_comparison(metrics)
        for name, m in metrics["models"].items():
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_confusion_matrix(m["confusion_matrix"], f"Confusion matrix – {name}", ax=ax)
            fig.tight_layout()
            figs["12_confusion_" + name.split()[0].lower().strip("()")] = fig
    for name, fig in figs.items():
        fig.savefig(config.FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Data source: {tag}\nSaved {len(figs)} figures -> {config.FIG_DIR}")
    print(summary_table(clean).to_string())


if __name__ == "__main__":
    save_all()
