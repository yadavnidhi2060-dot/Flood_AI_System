"""
Prediction interface used by the dashboard and the command line.

The model estimates the flood-risk class for the NEXT day from today's conditions.
It returns class probabilities, not certainties.

CLI example:
    python -m src.predict --rain-today 85 --rain-3d 190 --rain-7d 260 --rain-14d 320 \
        --level 11.2 --level-change 1.3 --warning 10.5 --temp 25 --humidity 92 --month 8
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import joblib
import pandas as pd

from src import config


@dataclass
class Prediction:
    label: str
    probabilities: dict          # {"Low": p, "Medium": p, "High": p}
    risk_score: float            # 0-100 expected-severity score (0 = Low, 100 = High)
    advice: str


ADVICE = {
    "Low": "Conditions are well below warning thresholds. Continue normal monitoring.",
    "Medium": "Conditions are approaching or above the warning level. Stay alert, follow official "
              "bulletins and prepare for possible escalation.",
    "High": "Conditions resemble past flood events. Check official flood warnings immediately and "
            "follow instructions from local disaster-management authorities.",
}


def load_model(path=config.MODEL_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError("Trained model not found. Run `python -m src.train` first.")
    return joblib.load(path)


def build_feature_row(rain_today, rain_3d, rain_7d, rain_14d, river_level, level_change_1d,
                      warning_level, temperature, humidity, month) -> pd.DataFrame:
    """Turn raw user inputs into the exact feature vector the model was trained on."""
    # The multi-day rainfall totals can never be smaller than the shorter windows
    rain_3d = max(rain_3d, rain_today)
    rain_7d = max(rain_7d, rain_3d)
    rain_14d = max(rain_14d, rain_7d)
    row = {
        "rainfall_mm": rain_today, "rain_3d_mm": rain_3d, "rain_7d_mm": rain_7d,
        "rain_14d_mm": rain_14d, "river_level_m": river_level,
        "level_change_1d_m": level_change_1d,
        "level_to_warning_m": river_level - warning_level,
        "temperature_c": temperature, "humidity_pct": humidity,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }
    return pd.DataFrame([row])[config.FEATURES]


def predict(bundle: dict, X: pd.DataFrame) -> Prediction:
    proba = bundle["model"].predict_proba(X[bundle["features"]])[0]
    classes = list(bundle["model"].classes_)
    probs = {config.RISK_LABELS[int(c)]: float(p) for c, p in zip(classes, proba)}
    for lab in config.RISK_LABELS:                   # make sure all 3 keys exist
        probs.setdefault(lab, 0.0)
    label = max(probs, key=probs.get)
    score = 100 * (0.5 * probs["Medium"] + 1.0 * probs["High"])
    return Prediction(label, probs, score, ADVICE[label])


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate next-day flood risk")
    ap.add_argument("--rain-today", type=float, required=True)
    ap.add_argument("--rain-3d", type=float, required=True)
    ap.add_argument("--rain-7d", type=float, required=True)
    ap.add_argument("--rain-14d", type=float, required=True)
    ap.add_argument("--level", type=float, required=True, help="current river level (m)")
    ap.add_argument("--level-change", type=float, default=0.0, help="change in level over 24 h (m)")
    ap.add_argument("--warning", type=float, required=True, help="warning level of the gauge (m)")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--humidity", type=float, required=True)
    ap.add_argument("--month", type=int, required=True)
    a = ap.parse_args()
    X = build_feature_row(a.rain_today, a.rain_3d, a.rain_7d, a.rain_14d, a.level, a.level_change,
                          a.warning, a.temp, a.humidity, a.month)
    bundle = load_model()
    res = predict(bundle, X)
    print(f"Model: {bundle['model_name']}  | trained on: {bundle['data_source']}")
    print(f"Estimated next-day flood risk: {res.label}")
    for k, v in res.probabilities.items():
        print(f"  P({k:6s}) = {v:6.1%}")
    print(res.advice)
    print("\n" + config.DISCLAIMER)


if __name__ == "__main__":
    main()
