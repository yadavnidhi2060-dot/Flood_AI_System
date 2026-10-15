"""Load the raw dataset and make clear whether it is REAL or SAMPLE data."""
from __future__ import annotations

import pandas as pd

from src import config

REAL_TAG = "OBSERVED (real)"
SAMPLE_TAG = "SAMPLE (synthetic)"


def resolve_raw_path():
    """Real observations take priority over the synthetic sample if both exist."""
    if config.OBSERVED_FILE.exists():
        return config.OBSERVED_FILE, REAL_TAG
    if config.SAMPLE_FILE.exists():
        return config.SAMPLE_FILE, SAMPLE_TAG
    raise FileNotFoundError(
        "No dataset found. Run `python -m src.generate_sample_data` to create the synthetic "
        "sample, or put your real data in data/raw/observed_data.csv (see README)."
    )


def load_raw() -> tuple[pd.DataFrame, str]:
    """Return (raw dataframe, data-source label)."""
    path, tag = resolve_raw_path()
    df = pd.read_csv(path)
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}\n"
                         f"Required: {config.REQUIRED_COLUMNS}")
    # The label always follows the file, so sample data can never be mistaken for real data.
    df["data_source"] = tag
    return df, tag


def load_clean() -> pd.DataFrame:
    """Load the cleaned dataset created by `python -m src.preprocessing`."""
    if not config.CLEAN_FILE.exists():
        raise FileNotFoundError("Cleaned data not found. Run `python -m src.preprocessing` first.")
    df = pd.read_csv(config.CLEAN_FILE, parse_dates=["date"])
    return df
