"""
Build a REAL-DATA file (data/raw/observed_data.csv) from public sources.

Real data cannot be bundled in this repository because river-gauge records and flood event
lists are published by agencies under their own terms. This script automates the part that is
openly downloadable and merges it with the files you download manually:

  1. Weather (automatic)  : NASA POWER daily point API  -> rainfall (PRECTOTCORR), temperature (T2M),
                            relative humidity (RH2M). These are satellite/reanalysis-based estimates for a
                            grid cell, not a ground rain-gauge.  https://power.larc.nasa.gov/
  2. River level (manual) : download daily gauge data from India-WRIS (https://indiawris.gov.in/) or
                            CWC (https://cwc.gov.in/), save as CSV with columns  date, river_level_m
  3. Flood records (manual, optional): list of dates when a flood was reported (e.g. NDMA/State SDMA
                            reports, Dartmouth Flood Observatory archive, EM-DAT, news-verified events),
                            CSV with one column  date.  If omitted, flood_occurred is set to 0 and the
                            High class then comes only from the danger-level rule.
  4. Gauge thresholds     : warning / danger levels of your gauge from CWC flood bulletins.

Example:
  python -m src.build_observed_dataset --station "Godavari at Polavaram" --lat 17.25 --lon 81.65 \
      --start 2010-01-01 --end 2023-12-31 --levels data/raw/my_levels.csv \
      --floods data/raw/my_flood_dates.csv --warning 10.5 --danger 12.0

Requires internet access (NASA POWER). Use --weather-csv to reuse an already downloaded file.
"""
from __future__ import annotations

import argparse

import pandas as pd
import requests

from src import config

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_nasa_power(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "parameters": "PRECTOTCORR,T2M,RH2M", "community": "AG", "latitude": lat,
        "longitude": lon, "start": start.replace("-", ""), "end": end.replace("-", ""),
        "format": "JSON",
    }
    r = requests.get(POWER_URL, params=params, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame({"rainfall_mm": p["PRECTOTCORR"], "temperature_c": p["T2M"],
                       "humidity_pct": p["RH2M"]})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df.replace(-999, float("nan")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--weather-csv", help="use an existing CSV (date, rainfall_mm, temperature_c, humidity_pct)")
    ap.add_argument("--levels", required=True, help="CSV with columns: date, river_level_m")
    ap.add_argument("--floods", help="CSV with a 'date' column listing days with a recorded flood")
    ap.add_argument("--warning", type=float, required=True, help="gauge warning level (m)")
    ap.add_argument("--danger", type=float, required=True, help="gauge danger level (m)")
    ap.add_argument("--append", action="store_true", help="append to an existing observed_data.csv (multi-station)")
    a = ap.parse_args()

    if a.weather_csv:
        weather = pd.read_csv(a.weather_csv, parse_dates=["date"])
    else:
        if a.lat is None or a.lon is None:
            ap.error("--lat and --lon are required unless --weather-csv is given")
        print("Downloading NASA POWER weather ...")
        weather = fetch_nasa_power(a.lat, a.lon, a.start, a.end)

    levels = pd.read_csv(a.levels, parse_dates=["date"])[["date", "river_level_m"]]
    df = weather.merge(levels, on="date", how="inner")          # keep days that have a gauge reading
    df["station"] = a.station
    df["warning_level_m"], df["danger_level_m"] = a.warning, a.danger
    df["flood_occurred"] = 0
    if a.floods:
        fl = pd.to_datetime(pd.read_csv(a.floods)["date"])
        df.loc[df["date"].isin(fl), "flood_occurred"] = 1
    df = df[config.REQUIRED_COLUMNS]

    if a.append and config.OBSERVED_FILE.exists():
        old = pd.read_csv(config.OBSERVED_FILE, parse_dates=["date"])
        df = pd.concat([old, df], ignore_index=True)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.OBSERVED_FILE, index=False)
    print(f"Saved {len(df):,} rows -> {config.OBSERVED_FILE}")
    print("Now run:  python -m src.preprocessing && python -m src.train && python -m src.eda")


if __name__ == "__main__":
    main()
