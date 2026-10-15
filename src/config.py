"""Central configuration for the Flood Analysis & Prediction System."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

# Dataset files
OBSERVED_FILE = DATA_RAW / "observed_data.csv"        # real observations (optional, user supplied)
SAMPLE_FILE = DATA_RAW / "sample_flood_data.csv"      # synthetic demo data (generated)
CLEAN_FILE = DATA_PROCESSED / "clean_data.csv"
FEATURE_FILE = DATA_PROCESSED / "model_features.csv"

# Model artefacts
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"
TEST_PRED_FILE = MODELS_DIR / "test_predictions.csv"

# Schema of the raw data. Every dataset (real or sample) must contain these columns.
REQUIRED_COLUMNS = [
    "date", "station", "rainfall_mm", "river_level_m",
    "temperature_c", "humidity_pct", "warning_level_m", "danger_level_m",
    "flood_occurred",
]

# Risk classes (ordinal: Low < Medium < High)
RISK_LABELS = ["Low", "Medium", "High"]
RISK_TO_INT = {name: i for i, name in enumerate(RISK_LABELS)}
RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}

# Features fed to the models (all are known at prediction time "today"
# and are used to predict the flood-risk class for the NEXT day).
FEATURES = [
    "rainfall_mm", "rain_3d_mm", "rain_7d_mm", "rain_14d_mm",
    "river_level_m", "level_change_1d_m", "level_to_warning_m",
    "temperature_c", "humidity_pct", "month_sin", "month_cos",
]

# Human-friendly feature names used in charts
FEATURE_LABELS = {
    "rainfall_mm": "Rainfall today (mm)",
    "rain_3d_mm": "Rainfall, last 3 days (mm)",
    "rain_7d_mm": "Rainfall, last 7 days (mm)",
    "rain_14d_mm": "Rainfall, last 14 days (mm)",
    "river_level_m": "River level (m)",
    "level_change_1d_m": "24 h change in river level (m)",
    "level_to_warning_m": "Level minus warning level (m)",
    "temperature_c": "Temperature (°C)",
    "humidity_pct": "Humidity (%)",
    "month_sin": "Season (sin of month)",
    "month_cos": "Season (cos of month)",
}

TEST_FRACTION = 0.20      # last 20 % of the timeline is held out for testing
RANDOM_STATE = 42

DISCLAIMER = (
    "This system is an educational machine-learning project. Its output is a statistical "
    "estimate of flood risk, NOT a guarantee that a flood will or will not occur. "
    "For real flood warnings always follow official agencies (e.g. CWC, IMD, NDMA, "
    "State Disaster Management Authorities)."
)
