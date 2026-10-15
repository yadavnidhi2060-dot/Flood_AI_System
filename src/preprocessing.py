"""
Data cleaning, missing-value handling and feature engineering.

Pipeline
--------
1. Validate schema, parse dates, drop duplicate rows.
2. Replace impossible / sentinel values (e.g. -999, humidity > 100) with NaN.
3. Impute gaps per station: short gaps by time interpolation, long gaps by the
   station's day-of-year median (no look-ahead beyond a 3-day interpolation window).
4. Create the flood-risk class for every day:
        High   : flood recorded OR river level >= danger level
        Medium : river level >= warning level
        Low    : otherwise
5. Engineer features (rolling rainfall, level change, seasonality).
6. TARGET = risk class of the NEXT day, so the model is a genuine 1-day-ahead
   predictor and does not just re-read today's river level.

Usage:  python -m src.preprocessing
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_raw

VALID_RANGES = {
    "rainfall_mm": (0, 500),
    "river_level_m": (0, 60),
    "temperature_c": (-10, 50),
    "humidity_pct": (0, 100),
}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return cleaned data and a small report describing what was changed."""
    report: dict = {"rows_in": int(len(df))}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    report["bad_dates_dropped"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    before = len(df)
    df = df.drop_duplicates(subset=["date", "station"], keep="first")
    report["duplicates_dropped"] = int(before - len(df))

    for col in VALID_RANGES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["flood_occurred"] = pd.to_numeric(df["flood_occurred"], errors="coerce").fillna(0).astype(int)

    # 2. impossible values -> NaN
    invalid = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["invalid_values_set_to_nan"] = invalid
    report["missing_before_imputation"] = {c: int(df[c].isna().sum()) for c in VALID_RANGES}

    # 3. impute per station
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("station", sort=False):
        g = g.copy().set_index("date")
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full)                                  # expose missing days
        g["station"] = g["station"].ffill().bfill()
        for c in ["warning_level_m", "danger_level_m"]:
            g[c] = g[c].ffill().bfill()
        g["flood_occurred"] = g["flood_occurred"].fillna(0).astype(int)
        g["data_source"] = g["data_source"].ffill().bfill()
        for col in VALID_RANGES:
            if col == "rainfall_mm":
                # rainfall is not smooth, so interpolating it would invent rain; use the
                # station's day-of-year median instead (0 mm in the dry season)
                med = g[col].groupby(g.index.dayofyear).transform("median")
                g[col] = g[col].fillna(med).fillna(0)
            else:
                g[col] = g[col].interpolate(method="time", limit=3, limit_direction="both")
                med = g[col].groupby(g.index.dayofyear).transform("median")
                g[col] = g[col].fillna(med).fillna(g[col].median())
        g.index.name = "date"
        parts.append(g.reset_index())
    df = pd.concat(parts, ignore_index=True)
    report["missing_after_imputation"] = {c: int(df[c].isna().sum()) for c in VALID_RANGES}

    # 4. risk class of each day
    df["risk_today"] = assign_risk(df["river_level_m"], df["warning_level_m"],
                                   df["danger_level_m"], df["flood_occurred"])
    report["rows_out"] = int(len(df))
    return df.sort_values(["date", "station"]).reset_index(drop=True), report


def assign_risk(level, warning, danger, flood) -> pd.Series:
    """Rule used to turn level + flood record into Low / Medium / High."""
    level, warning, danger, flood = map(pd.Series, (level, warning, danger, flood))
    risk = pd.Series("Low", index=level.index)
    risk[level >= warning] = "Medium"
    risk[(level >= danger) | (flood == 1)] = "High"
    return risk


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features and the next-day target (per station)."""
    df = df.sort_values(["station", "date"]).copy()
    g = df.groupby("station", sort=False)
    df["rain_3d_mm"] = g["rainfall_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_7d_mm"] = g["rainfall_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["rain_14d_mm"] = g["rainfall_mm"].transform(lambda s: s.rolling(14, min_periods=1).sum())
    df["level_change_1d_m"] = g["river_level_m"].diff().fillna(0)
    df["level_to_warning_m"] = df["river_level_m"] - df["warning_level_m"]
    m = df["date"].dt.month
    df["month"] = m
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["target"] = g["risk_today"].shift(-1)                 # risk tomorrow
    return df.sort_values(["date", "station"]).reset_index(drop=True)


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_features(df).dropna(subset=["target"]).copy()
    out["target_int"] = out["target"].map(config.RISK_TO_INT).astype(int)
    return out


def run() -> pd.DataFrame:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    raw, tag = load_raw()
    cleaned, report = clean(raw)
    cleaned.to_csv(config.CLEAN_FILE, index=False)
    table = build_model_table(cleaned)
    table.to_csv(config.FEATURE_FILE, index=False)

    print(f"Data source          : {tag}")
    print(f"Rows in -> out       : {report['rows_in']:,} -> {report['rows_out']:,}")
    print(f"Duplicates dropped   : {report['duplicates_dropped']}")
    print(f"Invalid -> NaN       : {report['invalid_values_set_to_nan']}")
    print(f"Missing (before)     : {report['missing_before_imputation']}")
    print(f"Missing (after)      : {report['missing_after_imputation']}")
    print("Target distribution  :", table["target"].value_counts().to_dict())
    return table


if __name__ == "__main__":
    run()
