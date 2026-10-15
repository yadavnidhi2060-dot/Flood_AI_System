"""Unit tests for the data pipeline and prediction helpers.  Run:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.generate_sample_data import generate
from src.predict import build_feature_row
from src.preprocessing import assign_risk, build_model_table, clean
from src.train import time_split


@pytest.fixture(scope="module")
def raw():
    d = generate(years=3, end_year=2020, seed=1)
    d["data_source"] = "SAMPLE (synthetic)"
    return d


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean(raw)


def test_sample_data_has_required_columns_and_flag(raw):
    assert set(config.REQUIRED_COLUMNS) <= set(raw.columns)
    assert (raw["data_source"] == "SAMPLE (synthetic)").all()


def test_cleaning_removes_gaps_duplicates_and_impossible_values(raw, cleaned):
    df, report = cleaned
    assert report["duplicates_dropped"] > 0
    assert df[["rainfall_mm", "river_level_m", "temperature_c", "humidity_pct"]].isna().sum().sum() == 0
    assert df["humidity_pct"].between(0, 100).all()
    assert df["rainfall_mm"].ge(0).all()
    assert not df.duplicated(["date", "station"]).any()


def test_risk_rule():
    r = assign_risk([5, 11, 13, 5], [10] * 4, [12] * 4, [0, 0, 0, 1])
    assert list(r) == ["Low", "Medium", "High", "High"]


def test_target_is_next_day_risk(cleaned):
    df, _ = cleaned
    table = build_model_table(df)
    st = table["station"].iloc[0]
    g = table[table["station"] == st].sort_values("date")
    full = df[df["station"] == st].sort_values("date").set_index("date")["risk_today"]
    sample = g.iloc[100]
    assert sample["target"] == full.loc[sample["date"] + pd.Timedelta(days=1)]


def test_no_nan_in_features(cleaned):
    table = build_model_table(cleaned[0])
    assert table[config.FEATURES].isna().sum().sum() == 0


def test_time_split_is_chronological(cleaned):
    table = build_model_table(cleaned[0])
    train, test, cutoff = time_split(table)
    assert train["date"].max() < test["date"].min()
    assert abs(len(test) / len(table) - config.TEST_FRACTION) < 0.05


def test_feature_row_matches_training_schema():
    X = build_feature_row(20, 10, 5, 2, 8.0, 0.2, 10.0, 27, 80, 7)   # inconsistent windows on purpose
    assert list(X.columns) == config.FEATURES
    assert X["rain_3d_mm"].iloc[0] >= X["rainfall_mm"].iloc[0]
    assert X["rain_14d_mm"].iloc[0] >= X["rain_7d_mm"].iloc[0] >= X["rain_3d_mm"].iloc[0]
    assert np.isclose(X["level_to_warning_m"].iloc[0], -2.0)
