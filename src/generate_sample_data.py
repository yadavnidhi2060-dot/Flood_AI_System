"""
Generate a SYNTHETIC sample dataset for demonstration.

IMPORTANT: Everything produced here is simulated, NOT real observations.
The simulator imitates a monsoon-driven river basin (seasonal rainfall, wet/dry spells,
a two-reservoir rainfall-runoff response for the river level, correlated temperature
and humidity) so the end-to-end pipeline can be run without external downloads.
Every row is tagged `data_source = "SAMPLE (synthetic)"`.

To use REAL data instead, see README.md -> "Using real-world data"
(NASA POWER weather + CWC / India-WRIS river levels + official flood records).

Usage:
    python -m src.generate_sample_data            # default: 12 years, 3 stations
    python -m src.generate_sample_data --years 15 --seed 7
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config

STATIONS = {
    # name: (rain scale multiplier, fast-store gain, slow-store gain, base level m)
    "Station A (Upper Basin)": (1.15, 0.060, 0.012, 2.0),
    "Station B (Mid Basin)": (1.00, 0.050, 0.014, 3.0),
    "Station C (Delta)": (0.90, 0.040, 0.018, 1.5),
}
SOURCE_TAG = "SAMPLE (synthetic)"


def _simulate_station(dates: pd.DatetimeIndex, params: tuple, year_factor: dict,
                      rng: np.random.Generator) -> pd.DataFrame:
    rain_mult, g_fast, g_slow, base = params
    n = len(dates)
    doy = dates.dayofyear.values
    year = dates.year.values

    # Seasonal monsoon strength: Gaussian bump centred on late July (day ~205)
    monsoon = np.exp(-0.5 * ((doy - 205) / 38.0) ** 2)
    # Weak post-monsoon bump (Oct-Nov) typical of the Indian peninsula
    post = 0.25 * np.exp(-0.5 * ((doy - 300) / 20.0) ** 2)
    season = np.clip(monsoon + post, 0, 1)

    rain = np.zeros(n)
    wet = False
    for i in range(n):
        yf = year_factor[year[i]]
        p_wet_after_wet = 0.35 + 0.50 * season[i]
        p_wet_after_dry = 0.04 + 0.45 * season[i]
        wet = rng.random() < (p_wet_after_wet if wet else p_wet_after_dry)
        if wet:
            scale = (3.0 + 20.0 * season[i]) * rain_mult * yf
            amount = rng.gamma(shape=0.75, scale=scale)
            if rng.random() < 0.025 * (0.3 + season[i]):   # rare extreme downpour
                amount += rng.gamma(4.0, 22.0) * yf
            rain[i] = min(amount, 420.0)

    # Rainfall-runoff: fast (flashy) and slow (baseflow) stores
    fast = slow = 0.0
    level = np.zeros(n)
    for i in range(n):
        fast = 0.55 * fast + g_fast * rain[i]
        slow = 0.975 * slow + g_slow * rain[i]
        level[i] = base + fast * 1.6 + slow * 0.9
    level += rng.normal(0, 0.08, n)

    temp = (27.5 + 4.0 * np.sin(2 * np.pi * (doy - 100) / 365.0) - 3.5 * season
            - 0.02 * rain + rng.normal(0, 1.2, n))
    hum = (52 + 32 * season + 0.12 * np.minimum(rain, 80) + rng.normal(0, 6, n))
    hum = np.clip(hum, 25, 100)

    return pd.DataFrame({"date": dates, "rainfall_mm": rain.round(1),
                         "river_level_m": level.round(2),
                         "temperature_c": temp.round(1),
                         "humidity_pct": hum.round(0)})


def generate(years: int = 12, end_year: int = 2023, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{end_year - years + 1}-01-01", f"{end_year}-12-31", freq="D")
    # Year-to-year variability of the monsoon (some wet, some dry years)
    year_factor = {y: float(np.clip(rng.lognormal(0.0, 0.30), 0.55, 1.9))
                   for y in range(end_year - years + 1, end_year + 1)}

    frames = []
    for name, params in STATIONS.items():
        df = _simulate_station(dates, params, year_factor, rng)
        df.insert(1, "station", name)
        # Warning / danger levels = upper quantiles of the simulated levels
        df["warning_level_m"] = round(float(df["river_level_m"].quantile(0.93)), 2)
        df["danger_level_m"] = round(float(df["river_level_m"].quantile(0.98)), 2)
        # Flood record: very likely above danger level, rare below it
        above_d = df["river_level_m"] >= df["danger_level_m"]
        above_w = df["river_level_m"] >= df["warning_level_m"]
        u = rng.random(len(df))
        df["flood_occurred"] = ((above_d & (u < 0.90)) | (above_w & ~above_d & (u < 0.04))).astype(int)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["data_source"] = SOURCE_TAG

    # ---- Inject realistic data-quality problems so cleaning is meaningful ----
    n = len(data)
    for col, frac in [("rainfall_mm", 0.025), ("river_level_m", 0.02),
                      ("temperature_c", 0.03), ("humidity_pct", 0.03)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        data.loc[idx, col] = np.nan
    out_idx = rng.choice(n, 25, replace=False)                 # sensor glitches
    data.loc[out_idx[:8], "rainfall_mm"] = -999.0              # common "missing" code
    data.loc[out_idx[8:16], "humidity_pct"] = 140.0            # physically impossible
    data.loc[out_idx[16:], "temperature_c"] = 85.0             # impossible temperature
    dup = data.sample(20, random_state=seed)                   # duplicate rows
    data = pd.concat([data, dup], ignore_index=True).sample(frac=1, random_state=seed)
    return data.sort_values(["date", "station"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=12)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = ap.parse_args()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df = generate(args.years, args.end_year, args.seed)
    df.to_csv(config.SAMPLE_FILE, index=False)
    print(f"Saved {len(df):,} rows -> {config.SAMPLE_FILE}")
    print("NOTE: this is SYNTHETIC sample data (data_source column says so).")


if __name__ == "__main__":
    main()
