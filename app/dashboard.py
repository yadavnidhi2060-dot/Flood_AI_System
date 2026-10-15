"""
Flood Analysis & Prediction System – Streamlit dashboard.

Run from the project root:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config, eda, predict as pred
from src.data_loader import REAL_TAG
from src.evaluate import plot_confusion_matrix
from src.preprocessing import add_features

st.set_page_config(page_title="Flood Analysis & Prediction", page_icon="🌊", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
h1, h2, h3 {color:#0a3d62; letter-spacing:-0.01em;}
.hero {background: linear-gradient(120deg,#0a3d62 0%,#0b6e99 60%,#2a9bb8 100%); color:white;
       padding: 1.5rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:1.9rem;}
.hero p {margin:.3rem 0 0 0; opacity:.92; font-size:1rem;}
.badge {display:inline-block; padding:.28rem .7rem; border-radius:999px; font-size:.8rem; font-weight:600;}
.badge-sample {background:#fff3cd; color:#7a5b00; border:1px solid #f0c24b;}
.badge-real {background:#d9f2e3; color:#14633a; border:1px solid #7fcf9f;}
.kpi {background:white; border:1px solid #dbe6ef; border-radius:12px; padding:.9rem 1rem;
      box-shadow:0 1px 3px rgba(10,61,98,.06);}
.kpi .v {font-size:1.65rem; font-weight:700; color:#0a3d62; line-height:1.15;}
.kpi .l {font-size:.78rem; color:#5b7083; text-transform:uppercase; letter-spacing:.05em;}
.kpi .s {font-size:.78rem; color:#7b8b99;}
.risk-card {border-radius:16px; padding:1.4rem 1.6rem; color:white; text-align:center;
            box-shadow:0 4px 14px rgba(0,0,0,.15);}
.risk-card .t {font-size:.85rem; letter-spacing:.12em; text-transform:uppercase; opacity:.9;}
.risk-card .r {font-size:2.8rem; font-weight:800; line-height:1.1; margin:.2rem 0;}
.risk-card .a {font-size:.92rem; opacity:.95;}
.meter {position:relative; height:16px; border-radius:8px; margin:.6rem 0 .2rem 0;
        background:linear-gradient(90deg,#2e9e5b 0%,#f0a30a 50%,#d64545 100%);}
.meter .pin {position:absolute; top:-5px; width:6px; height:26px; background:#102a43;
             border-radius:3px; border:2px solid white;}
.pbar-row {display:flex; align-items:center; margin:.35rem 0; font-size:.9rem;}
.pbar-row .n {width:70px; font-weight:600;}
.pbar-row .b {flex:1; background:#e6edf3; border-radius:6px; height:14px; overflow:hidden;}
.pbar-row .b div {height:100%; border-radius:6px;}
.pbar-row .p {width:60px; text-align:right; font-variant-numeric:tabular-nums;}
.note {background:#eef5fb; border-left:4px solid #0b6e99; padding:.7rem 1rem; border-radius:6px;
       font-size:.9rem; color:#1d3b53;}
.warn {background:#fff6e5; border-left:4px solid #f0a30a; padding:.7rem 1rem; border-radius:6px;
       font-size:.9rem; color:#5c4300;}
[data-testid="stSidebar"] {background:#eaf1f8;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def get_data():
    from src.data_loader import load_raw
    raw, tag = load_raw()
    clean = pd.read_csv(config.CLEAN_FILE, parse_dates=["date"])
    return raw, clean, add_features(clean), tag


@st.cache_data(show_spinner=False)
def get_metrics():
    with open(config.METRICS_FILE) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_importance():
    return pd.read_csv(config.IMPORTANCE_FILE)


@st.cache_data(show_spinner=False)
def get_test_predictions():
    return pd.read_csv(config.TEST_PRED_FILE, parse_dates=["date"])


@st.cache_resource(show_spinner=False)
def get_model():
    return pred.load_model()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def source_badge(tag):
    cls = "badge-real" if tag == REAL_TAG else "badge-sample"
    icon = "✅" if tag == REAL_TAG else "⚠️"
    return f'<span class="badge {cls}">{icon} Data source: {tag}</span>'


# ----------------------------------------------------------------------------- guard
needed = [config.CLEAN_FILE, config.MODEL_FILE, config.METRICS_FILE, config.IMPORTANCE_FILE]
if not all(p.exists() for p in needed):
    st.error("The project has not been prepared yet. Run these commands in the terminal, then refresh:")
    st.code("python -m src.generate_sample_data\npython -m src.preprocessing\npython -m src.train\n"
            "streamlit run app/dashboard.py", language="bash")
    st.stop()

raw, clean, feat, SOURCE = get_data()
metrics = get_metrics()
bundle = get_model()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🌊 Flood AI")
    page = st.radio("Navigate", ["🏠 Overview", "📊 Data Explorer", "🔮 Predict Flood Risk",
                                 "🧪 Model Performance", "📘 About & Method"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(source_badge(SOURCE), unsafe_allow_html=True)
    st.caption(f"Model: **{bundle['model_name']}**  \nTrained: {bundle['trained_at'].replace('T', ' ')}")

    flt = {}
    if page == "📊 Data Explorer":
        st.markdown("### Filters")
        stations = sorted(feat["station"].unique())
        flt["stations"] = st.multiselect("Station", stations, default=stations)
        dmin, dmax = feat["date"].min().date(), feat["date"].max().date()
        flt["dates"] = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
        mnames = eda.MONTHS
        flt["months"] = st.multiselect("Months", mnames, default=mnames)
        flt["risk"] = st.multiselect("Risk level (today)", config.RISK_LABELS, default=config.RISK_LABELS)

# ----------------------------------------------------------------------------- header
st.markdown(f"""
<div class="hero"><h1>🌊 Flood Analysis & Prediction System</h1>
<p>Machine-learning estimate of next-day flood risk (Low / Medium / High) from rainfall, river level,
temperature and humidity.</p></div>""", unsafe_allow_html=True)
if SOURCE != REAL_TAG:
    st.markdown('<div class="warn"><b>Demo mode:</b> the dataset shown here is <b>synthetic sample data</b> '
                'generated by a simulator, not real measurements. All figures and metrics describe the '
                'simulated data. See <i>About & Method</i> for how to load real observations.</div>',
                unsafe_allow_html=True)
    st.write("")

# ============================================================================= OVERVIEW
if page == "🏠 Overview":
    best = metrics["models"][metrics["best_model"]]
    c = st.columns(5)
    kpi(c[0], "Records", f"{len(clean):,}", f"{clean['station'].nunique()} stations")
    kpi(c[1], "Period", f"{clean['date'].dt.year.min()}–{clean['date'].dt.year.max()}", "daily data")
    kpi(c[2], "Flood days", f"{int(clean['flood_occurred'].sum()):,}", "recorded in dataset")
    kpi(c[3], "Best model", metrics["best_model"], "chosen by cross-validation")
    kpi(c[4], "Test macro-F1", f"{best['f1_macro']:.2f}", f"accuracy {best['accuracy']:.0%}")
    st.write("")

    l, r = st.columns([1.15, 1])
    with l:
        st.subheader("How the system works")
        st.markdown("""
1. **Collect** – daily rainfall, river level, temperature, humidity and flood records per gauge station.
2. **Clean** – remove duplicates and impossible values, fill gaps (interpolation / seasonal median).
3. **Label** – each day is *Low*, *Medium* (≥ warning level) or *High* (≥ danger level or flood recorded).
4. **Engineer features** – rainfall over 3/7/14 days, 24 h level change, distance to warning level, season.
5. **Train** – Logistic Regression and Random Forest, validated on a *chronological* hold-out period.
6. **Predict** – estimate tomorrow's risk class with probabilities for any weather / river-level input.
""")
        st.markdown('<div class="note">Predictions are probabilistic estimates for education and '
                    'research. They do <b>not</b> guarantee that a flood will or will not happen.</div>',
                    unsafe_allow_html=True)
    with r:
        show(eda.plot_risk_distribution(clean))
    show(eda.plot_rainfall_trend(clean))

# ============================================================================= DATA EXPLORER
elif page == "📊 Data Explorer":
    st.subheader("Exploratory data analysis")
    d = feat.copy()
    if flt["stations"]:
        d = d[d["station"].isin(flt["stations"])]
    if isinstance(flt["dates"], (tuple, list)) and len(flt["dates"]) == 2:
        d = d[(d["date"].dt.date >= flt["dates"][0]) & (d["date"].dt.date <= flt["dates"][1])]
    mnum = [eda.MONTHS.index(m) + 1 for m in flt["months"]]
    d = d[d["date"].dt.month.isin(mnum)]
    d = d[d["risk_today"].isin(flt["risk"])]

    if d.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    c = st.columns(5)
    kpi(c[0], "Days selected", f"{len(d):,}")
    kpi(c[1], "Total rainfall", f"{d['rainfall_mm'].sum():,.0f} mm", "sum over all selected rows")
    kpi(c[2], "Peak river level", f"{d['river_level_m'].max():.1f} m")
    kpi(c[3], "Flood days", f"{int(d['flood_occurred'].sum()):,}")
    kpi(c[4], "High-risk days", f"{(d['risk_today'] == 'High').mean():.1%}")
    st.write("")

    t = st.tabs(["🌧️ Rainfall", "🌊 Water levels", "📈 Flood frequency", "🔗 Relationships",
                 "🧹 Data quality", "🗂️ Data table"])
    with t[0]:
        show(eda.plot_rainfall_trend(d))
        show(eda.plot_monthly_rainfall(d))
    with t[1]:
        show(eda.plot_water_levels(d))
    with t[2]:
        a, b = st.columns([1.6, 1])
        with a:
            show(eda.plot_flood_frequency(d))
        with b:
            show(eda.plot_risk_distribution(d))
    with t[3]:
        a, b = st.columns(2)
        with a:
            show(eda.plot_correlation(d))
        with b:
            show(eda.plot_rain_vs_level(d))
        show(eda.plot_weather_by_risk(d))
    with t[4]:
        st.markdown("Share of missing values in the **raw** file, and how they were treated:")
        a, b = st.columns([1, 1])
        with a:
            show(eda.plot_missing_values(raw))
        with b:
            st.markdown("""
- Duplicate (date, station) rows removed
- Sentinel / impossible values (e.g. −999 rainfall, humidity > 100 %) set to missing
- River level, temperature, humidity: time interpolation (≤ 3 days), then day-of-year median
- Rainfall: day-of-year median (interpolating rain would invent rainfall)
""")
        st.markdown("**Descriptive statistics (cleaned data, selected rows)**")
        st.dataframe(eda.summary_table(d))
    with t[5]:
        cols = ["date", "station", "rainfall_mm", "river_level_m", "temperature_c", "humidity_pct",
                "warning_level_m", "danger_level_m", "flood_occurred", "risk_today"]
        view = d[cols].sort_values("date", ascending=False)
        st.dataframe(view, hide_index=True)
        st.download_button("⬇️ Download filtered data (CSV)", view.to_csv(index=False).encode(),
                           "filtered_flood_data.csv", "text/csv")

# ============================================================================= PREDICT
elif page == "🔮 Predict Flood Risk":
    st.subheader("Estimate next-day flood risk")
    st.caption("Enter today's conditions at a gauge station. The model returns probabilities for "
               "Low, Medium and High risk tomorrow.")

    thresholds = clean.groupby("station")[["warning_level_m", "danger_level_m"]].first()
    presets = {
        "Custom": dict(rain=10.0, r3=25.0, r7=50.0, r14=90.0, lvl=4.0, chg=0.1, temp=27.0, hum=70, mon=7),
        "Dry-season day": dict(rain=0.0, r3=1.0, r7=3.0, r14=6.0, lvl=2.5, chg=0.0, temp=31.0, hum=42, mon=2),
        "Moderate monsoon day": dict(rain=18.0, r3=45.0, r7=95.0, r14=170.0, lvl=6.0, chg=0.4, temp=26.0, hum=82, mon=7),
        "Heavy monsoon spell": dict(rain=70.0, r3=170.0, r7=260.0, r14=330.0, lvl=10.5, chg=1.2, temp=25.0, hum=92, mon=8),
        "Extreme downpour": dict(rain=150.0, r3=280.0, r7=380.0, r14=450.0, lvl=13.0, chg=2.5, temp=24.0, hum=96, mon=8),
    }
    left, right = st.columns([1.05, 1])
    with left:
        station = st.selectbox("Gauge station", thresholds.index.tolist())
        warn_lvl = float(thresholds.loc[station, "warning_level_m"])
        dang_lvl = float(thresholds.loc[station, "danger_level_m"])
        st.caption(f"Warning level **{warn_lvl:.1f} m** · Danger level **{dang_lvl:.1f} m**")
        preset = st.selectbox("Quick scenario (optional)", list(presets))
        p = presets[preset]
        k = preset  # widget keys change with the preset, so the inputs reset to its values
        st.markdown("**🌧️ Rainfall**")
        c1, c2 = st.columns(2)
        rain = c1.number_input("Today (mm)", 0.0, 500.0, p["rain"], 1.0, key=f"rain{k}")
        r3 = c2.number_input("Last 3 days total (mm)", 0.0, 1000.0, p["r3"], 5.0, key=f"r3{k}")
        c1, c2 = st.columns(2)
        r7 = c1.number_input("Last 7 days total (mm)", 0.0, 1500.0, p["r7"], 5.0, key=f"r7{k}")
        r14 = c2.number_input("Last 14 days total (mm)", 0.0, 2000.0, p["r14"], 5.0, key=f"r14{k}")
        st.markdown("**🌊 River**")
        c1, c2 = st.columns(2)
        lvl = c1.number_input("Current river level (m)", 0.0, 60.0, p["lvl"], 0.1, key=f"lvl{k}")
        chg = c2.number_input("Change in last 24 h (m)", -10.0, 10.0, p["chg"], 0.1, key=f"chg{k}")
        st.markdown("**🌡️ Weather**")
        c1, c2, c3 = st.columns(3)
        temp = c1.number_input("Temperature (°C)", -10.0, 50.0, p["temp"], 0.5, key=f"temp{k}")
        hum = c2.number_input("Humidity (%)", 0, 100, int(p["hum"]), 1, key=f"hum{k}")
        mon = c3.selectbox("Month", list(range(1, 13)), index=p["mon"] - 1,
                           format_func=lambda m: eda.MONTHS[m - 1], key=f"mon{k}")
        st.caption("The estimate on the right updates automatically when you change any input.")

    with right:
        X = pred.build_feature_row(rain, r3, r7, r14, lvl, chg, warn_lvl, temp, hum, mon)
        res = pred.predict(bundle, X)
        color = config.RISK_COLORS[res.label]
        st.markdown(f"""
<div class="risk-card" style="background:{color}">
  <div class="t">Estimated next-day flood risk</div>
  <div class="r">{res.label.upper()}</div>
  <div class="a">{res.advice}</div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="margin-top:1rem;font-weight:600;color:#0a3d62">Risk index: {res.risk_score:.0f} / 100</div>
<div class="meter"><div class="pin" style="left:calc({min(res.risk_score, 100):.1f}% - 3px)"></div></div>
<div style="display:flex;justify-content:space-between;font-size:.75rem;color:#5b7083">
<span>Low</span><span>Medium</span><span>High</span></div>""", unsafe_allow_html=True)

        st.markdown("**Class probabilities**")
        rows = "".join(
            f'<div class="pbar-row"><div class="n">{lab}</div><div class="b"><div style="width:'
            f'{res.probabilities[lab] * 100:.1f}%;background:{config.RISK_COLORS[lab]}"></div></div>'
            f'<div class="p">{res.probabilities[lab]:.1%}</div></div>' for lab in config.RISK_LABELS)
        st.markdown(rows, unsafe_allow_html=True)

        # position of the river relative to the thresholds
        st.markdown(f"**River level vs thresholds:** {lvl:.1f} m  → "
                    f"{'above danger level' if lvl >= dang_lvl else 'above warning level' if lvl >= warn_lvl else 'below warning level'}")

        # out-of-range warning (extrapolation)
        rng = feat[config.FEATURES]
        out = [config.FEATURE_LABELS[c] for c in config.FEATURES
               if X[c].iloc[0] > rng[c].max() * 1.05 or X[c].iloc[0] < rng[c].min() - abs(rng[c].min()) * 0.05 - 1e-9]
        if out:
            st.markdown('<div class="warn">⚠️ Some inputs are outside the range seen in training data ('
                        + ", ".join(out[:4]) + '). The estimate is less reliable.</div>', unsafe_allow_html=True)
        if SOURCE != REAL_TAG:
            st.markdown('<div class="warn" style="margin-top:.6rem">The model was trained on '
                        '<b>synthetic sample data</b>; treat this as a demonstration only.</div>',
                        unsafe_allow_html=True)
        st.caption(config.DISCLAIMER)

# ============================================================================= MODEL PERFORMANCE
elif page == "🧪 Model Performance":
    st.subheader("Model evaluation")
    sp = metrics["split"]
    st.markdown(f"""<div class="note"><b>Validation design:</b> chronological split – trained on
{sp['train_start']} → {sp['train_end']} ({sp['train_rows']:,} rows), tested on the unseen period
{sp['test_start']} → {sp['test_end']} ({sp['test_rows']:,} rows). Test class counts:
{', '.join(f'{k}: {v}' for k, v in metrics['class_counts_test'].items())}.
<b>Macro</b> averages treat Low, Medium and High equally, so rare High events matter as much as common Low days.
</div>""", unsafe_allow_html=True)
    st.write("")

    rows = []
    for name, m in metrics["models"].items():
        rows.append({"Model": name + (" ⭐" if name == metrics["best_model"] else ""),
                     "Accuracy": m["accuracy"], "Precision (macro)": m["precision_macro"],
                     "Recall (macro)": m["recall_macro"], "F1 (macro)": m["f1_macro"],
                     "F1 (weighted)": m["f1_weighted"], "Balanced acc.": m["balanced_accuracy"],
                     "CV F1 (train)": m["cv_f1_macro_train"]})
    tbl = pd.DataFrame(rows).set_index("Model")
    st.dataframe(tbl.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None))
    st.caption("⭐ = selected by cross-validated macro-F1 on the training period only. The baseline always "
               "predicts 'Low': note its high accuracy but near-zero usefulness – why accuracy alone is misleading.")
    show(eda.plot_model_comparison(metrics))

    st.markdown("### Confusion matrix & per-class scores")
    names = list(metrics["models"])
    sel = st.selectbox("Model", names, index=names.index(metrics["best_model"]))
    m = metrics["models"][sel]
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_confusion_matrix(m["confusion_matrix"], "Counts", ax=ax)
        fig.tight_layout()
        show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_confusion_matrix(m["confusion_matrix"], "Row-normalised (recall per actual class)",
                              normalize=True, ax=ax, cmap="Greens")
        fig.tight_layout()
        show(fig)
    pc = pd.DataFrame(m["per_class"]).T
    pc.columns = ["Precision", "Recall", "F1-score", "Support"]
    st.dataframe(pc.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}",
                                  "Support": "{:.0f}"}))
    if m.get("best_params"):
        st.caption(f"Best hyper-parameters: `{m['best_params']}`")

    st.markdown("### Feature importance")
    imp = get_importance()
    choice = st.radio("Method", ["Random Forest (impurity)", "Permutation (best model, test set)",
                                 "Logistic Regression |coefficient|"], horizontal=True)
    col, ttl = {"Random Forest (impurity)": ("rf_importance", "Random Forest feature importance"),
                "Permutation (best model, test set)": ("permutation_importance",
                                                       "Permutation importance (drop in macro-F1)"),
                "Logistic Regression |coefficient|": ("lr_abs_coef", "Mean |coefficient| (standardised)")}[choice]
    show(eda.plot_feature_importance(imp, col, ttl))

    st.markdown("### Test-period timeline")
    tp = get_test_predictions()
    st_sel = st.selectbox("Station", sorted(tp["station"].unique()))
    g = tp[tp["station"] == st_sel]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(g["date"], g["p_High"], color=config.RISK_COLORS["High"], alpha=.35, label="P(High)")
    ax.plot(g["date"], g["p_Medium"], color=config.RISK_COLORS["Medium"], lw=.9, label="P(Medium)")
    hi = g[g["actual"] == "High"]
    ax.scatter(hi["date"], [1.02] * len(hi), marker="|", s=90, color="black", label="Actual High (next day)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Predicted probability")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.set_title(f"Predicted risk vs actual High-risk days – {st_sel}", fontweight="bold")
    fig.tight_layout()
    show(fig)

# ============================================================================= ABOUT
else:
    st.subheader("About & Method")
    a, b = st.columns(2)
    with a:
        st.markdown("""
#### Problem
Floods are among the most damaging natural hazards. This project explores whether simple, openly
available observations – rainfall, river level, temperature and humidity – can be turned into a
**next-day flood-risk class** using machine learning.

#### Risk classes
| Class | Rule applied to the next day |
|---|---|
| 🟢 Low | River level below the gauge's warning level |
| 🟠 Medium | Level ≥ warning level |
| 🔴 High | Level ≥ danger level **or** a flood was recorded |

#### Models
* **Logistic Regression** – interpretable linear baseline (standardised inputs, class-balanced).
* **Random Forest** – non-linear ensemble, tuned with time-series cross-validation.
* **Majority-class baseline** – reference to expose misleading accuracy.
""")
    with b:
        st.markdown("""
#### Data provenance
* The bundled file is **synthetic**, produced by `src/generate_sample_data.py`; every row is tagged
  *SAMPLE (synthetic)*.
* To use **real observations**, place `data/raw/observed_data.csv` (or build it with
  `src/build_observed_dataset.py` from NASA POWER weather + CWC / India-WRIS river levels + official
  flood records). The dashboard then shows *OBSERVED (real)*.

#### Limitations
* Metrics from synthetic data say nothing about real-world accuracy.
* Gauge-level based labels depend on local warning / danger thresholds.
* No rainfall forecast, soil moisture, upstream gauges, dams or topography are used.
* Flood events are rare, so recall on *High* has wide uncertainty.
* A model trained on one basin should not be applied to another without retraining.
""")
        st.markdown(f'<div class="warn"><b>Disclaimer.</b> {config.DISCLAIMER}</div>', unsafe_allow_html=True)
