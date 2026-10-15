# 🌊 Flood Analysis & Prediction System using AI/ML

A final-year B.Tech (CSE – Data Science) project that analyses rainfall, river water level,
temperature, humidity and flood records, and trains machine-learning models to estimate the
**next-day flood risk (Low / Medium / High)**. Results are explored in an interactive Streamlit dashboard.

> ⚠️ **Data honesty notice.** The dataset bundled with this repository is **synthetic sample data**
> produced by `src/generate_sample_data.py` (every row is tagged `SAMPLE (synthetic)`, and the dashboard
> shows an amber "Demo mode" banner). Metrics obtained on it demonstrate that the *pipeline works*; they
> are **not** evidence of real-world accuracy. Section 6 explains how to plug in real observations.
> Model output is a probabilistic estimate and **never a guarantee** that a flood will or will not occur.

---

## 1. Features

| Requirement | Where |
|---|---|
| Data cleaning, missing values, preprocessing, EDA | `src/preprocessing.py`, `src/eda.py` |
| Logistic Regression & Random Forest (Low/Medium/High) + majority-class baseline | `src/train.py` |
| Rainfall trend, water-level, flood-frequency, feature-importance plots (Pandas/Matplotlib/Seaborn) | `src/eda.py` → `reports/figures/` |
| Streamlit dashboard: filters, charts, predictions, risk indicators | `app/dashboard.py` |
| Manual prediction form (rainfall, river level, temperature, humidity …) | dashboard → *Predict Flood Risk*, or `python -m src.predict` |
| Accuracy, precision, recall, F1, confusion matrix | `src/evaluate.py`, dashboard → *Model Performance*, `models/metrics.json` |
| Real-data path (NASA POWER + CWC/India-WRIS) | `src/build_observed_dataset.py` |
| Automated tests | `tests/test_pipeline.py` |

## 2. Folder structure

```
flood-ai-system/
├── README.md                    <- this guide
├── requirements.txt             <- Python dependencies
├── run_all.py                   <- one command: clean -> train -> figures
├── .streamlit/config.toml       <- dashboard theme
├── app/
│   └── dashboard.py             <- Streamlit dashboard (5 pages)
├── src/
│   ├── config.py                <- paths, feature list, class labels, disclaimer
│   ├── generate_sample_data.py  <- SYNTHETIC data simulator (clearly labelled)
│   ├── build_observed_dataset.py<- builds a REAL dataset (NASA POWER + your gauge/flood files)
│   ├── data_loader.py           <- loads data, tags source (real vs sample)
│   ├── preprocessing.py         <- cleaning, imputation, risk labels, feature engineering
│   ├── eda.py                   <- all plots (used by CLI and dashboard)
│   ├── evaluate.py              <- metrics + confusion-matrix plot
│   ├── train.py                 <- model training, tuning, selection, saving
│   └── predict.py               <- prediction API + CLI
├── data/
│   ├── raw/                     <- sample_flood_data.csv (synthetic) / observed_data.csv (real, optional)
│   └── processed/               <- clean_data.csv, model_features.csv
├── models/                      <- best_model.joblib, metrics.json, feature_importance.csv, test_predictions.csv
├── reports/figures/             <- PNG charts created by `python -m src.eda`
├── tests/test_pipeline.py       <- pytest unit tests
└── docs/PROJECT_REPORT_OUTLINE.md
```

## 3. Setup in VS Code (step by step)

**Prerequisites:** Python 3.10+ ([python.org](https://www.python.org/downloads/)) and VS Code with the *Python* extension.

1. **Open the project.** VS Code → *File ▸ Open Folder…* → select `flood-ai-system`.
2. **Open a terminal.** *Terminal ▸ New Terminal* (it opens in the project root).
3. **Create a virtual environment.**
   * Windows: `python -m venv .venv` then `.venv\Scripts\activate`
     (if PowerShell blocks it: `Set-ExecutionPolicy -Scope Process RemoteSigned`)
   * macOS / Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. **Select the interpreter.** `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv`.
5. **Install dependencies.** `pip install -r requirements.txt`
6. **Generate the sample dataset** (skip if `data/raw/sample_flood_data.csv` already exists):
   `python -m src.generate_sample_data`
7. **Run the whole pipeline (clean → train → evaluate → figures):** `python run_all.py`
   Or step by step:
   ```
   python -m src.preprocessing     # cleaning + features   -> data/processed/
   python -m src.train             # training + evaluation  -> models/
   python -m src.eda               # charts                 -> reports/figures/
   ```
   Training takes about 30–60 seconds.
8. **Launch the dashboard.** `streamlit run app/dashboard.py` → opens http://localhost:8501.
9. **(Optional) Run the tests.** `pytest -q`
10. **(Optional) Predict from the command line.**
    ```
    python -m src.predict --rain-today 85 --rain-3d 190 --rain-7d 260 --rain-14d 320 \
        --level 11.2 --level-change 1.3 --warning 10.5 --temp 25 --humidity 92 --month 8
    ```

**Troubleshooting**

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run commands from the project root using `python -m src.<module>` |
| Dashboard says "project has not been prepared" | Run `python run_all.py` first |
| `streamlit` not recognised | Activate the venv, or use `python -m streamlit run app/dashboard.py` |
| Port busy | `streamlit run app/dashboard.py --server.port 8502` |

## 4. Workflow explained

```
raw CSV ─► validate ─► clean ─► label risk ─► engineer features ─► chronological split
                                                                       │
                     dashboard ◄── saved model + metrics ◄── train / tune / evaluate
```

1. **Data schema** – one row per *station × day*: `date, station, rainfall_mm, river_level_m,
   temperature_c, humidity_pct, warning_level_m, danger_level_m, flood_occurred`.
2. **Cleaning** (`preprocessing.clean`) – parse dates; drop duplicate (date, station) rows; convert
   impossible values (rain < 0 or the −999 code, humidity > 100 %, temperature outside −10…50 °C) to missing;
   fill missing values per station – river level / temperature / humidity by time interpolation (≤ 3 days)
   then day-of-year median, rainfall by day-of-year median (interpolating rainfall would invent rain).
3. **Risk labels** – *High*: level ≥ danger level **or** flood recorded; *Medium*: level ≥ warning level;
   *Low*: otherwise.
4. **Prediction target** – the risk class of the **next day**. Using the same day's level to predict the
   same day's class would be trivial leakage because the class is *defined* from the level.
5. **Features** – rainfall today and cumulative over 3/7/14 days, river level, 24 h level change, level
   minus warning level, temperature, humidity, month encoded as sin/cos.
6. **Validation** – weather is a time series, so a random split would leak the future. The last 20 % of the
   timeline is the test set; hyper-parameters are tuned with forward-chaining `TimeSeriesSplit`; the best model
   is chosen by cross-validated macro-F1 on the training period only.
7. **Imbalance** – High events are rare, so both models use balanced class weights and are judged with
   macro-averaged precision/recall/F1. A majority-class baseline is included to show that high accuracy alone
   can be meaningless.
8. **Explainability** – Random-Forest impurity importance, permutation importance on the test set, and
   standardised logistic-regression coefficients.

### Example results (on the SYNTHETIC sample; yours will be close but not identical)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Majority-class baseline | 0.883 | 0.294 | 0.333 | 0.313 |
| Logistic Regression | 0.854 | 0.603 | 0.716 | 0.632 |
| Random Forest (selected) | 0.932 | 0.791 | 0.686 | 0.695 |

These numbers describe simulated data only. The test period contains just ~116 "High" days, so per-class
scores for High carry sizeable uncertainty.

## 5. Dashboard guide

* **Overview** – KPIs, workflow, class distribution, rainfall trend.
* **Data Explorer** – sidebar filters (station, date range, months, risk level) drive tabs for rainfall, water
  levels, flood frequency, relationships, data-quality summary and a downloadable table.
* **Predict Flood Risk** – choose a gauge, optionally load a scenario preset, enter rainfall / river /
  weather values; a colour-coded card (green/amber/red), 0–100 risk index and class probabilities update live.
  Inputs outside the training range trigger a reliability warning.
* **Model Performance** – metric table, model comparison chart, confusion matrices (counts and normalised),
  per-class report, feature importance (3 methods) and a test-period timeline.
* **About & Method** – rules, limitations, data provenance, disclaimer.

## 6. Using real-world data

The sample is synthetic because river-gauge and flood-event records are issued by agencies under their own
terms and cannot be redistributed here. To run the project on real observations:

| Variable | Public source |
|---|---|
| Rainfall, temperature, humidity | **NASA POWER** daily API (`PRECTOTCORR`, `T2M`, `RH2M`) – https://power.larc.nasa.gov/ (gridded satellite/reanalysis estimates, not rain-gauge readings). IMD gridded data (https://www.imdpune.gov.in/) is an alternative for India. |
| River water level, warning/danger levels | **India-WRIS** https://indiawris.gov.in/ and **CWC** https://cwc.gov.in/ (daily gauge data, flood bulletins) |
| Flood event dates | NDMA / State Disaster Management Authority reports, **Dartmouth Flood Observatory** https://floodobservatory.colorado.edu/, **EM-DAT** https://www.emdat.be/ |

Steps:

1. Download the daily river levels as `data/raw/my_levels.csv` with columns `date,river_level_m`.
2. (Optional) save flood dates as `data/raw/my_flood_dates.csv` with one column `date`.
3. Build the dataset (needs internet for NASA POWER):
   ```
   python -m src.build_observed_dataset --station "My Gauge" --lat 17.25 --lon 81.65 \
       --start 2010-01-01 --end 2023-12-31 --levels data/raw/my_levels.csv \
       --floods data/raw/my_flood_dates.csv --warning 10.5 --danger 12.0
   ```
   Repeat with `--append` for more stations.
4. Retrain: `python run_all.py` – when `data/raw/observed_data.csv` exists it takes priority and the dashboard
   badge switches to **OBSERVED (real)**.

> `build_observed_dataset.py` was written against the documented NASA POWER API but could not be run end-to-end
> in the environment where this project was authored (no external internet). Test it on your machine and
> check the output CSV before training.

## 7. Limitations & future work

* Labels are derived from gauge thresholds, so quality depends on correct local warning/danger levels.
* No forecast rainfall, soil moisture, upstream gauges, reservoir releases, terrain or land use.
* Gridded weather ≠ point rainfall at the gauge; basin-average rainfall would be better.
* Future work: LSTM / gradient boosting, multi-day lead times, SHAP explanations, upstream-station features,
  satellite flood extent (Sentinel-1) validation, probability calibration, live data feeds, deployment on
  Streamlit Community Cloud.

## 8. Disclaimer

This is an educational project. Predictions are statistical estimates and must not be used for emergency
decisions. Follow official warnings from CWC, IMD, NDMA and your State Disaster Management Authority.
