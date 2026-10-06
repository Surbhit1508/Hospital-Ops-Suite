"""Shared paths and constants for the Hospital Operations Intelligence Suite."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for _dir in (DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# --- Classification / Regression source (real UCI Diabetes 130-US Hospitals data) ---
DIABETES_TRAIN = DATA_RAW / "diabetes_readmission_train.csv"
DIABETES_TEST = DATA_RAW / "diabetes_readmission_test.csv"
CLASSIFICATION_TARGET = "readmitted"
REGRESSION_TARGET = "time_in_hospital"

# --- NLP source (real MTSamples de-identified medical transcription corpus) ---
MTSAMPLES_TRAIN = DATA_RAW / "mtsamples_train.parquet"
MTSAMPLES_TEST = DATA_RAW / "mtsamples_test.parquet"

# --- Forecasting source (real HHS/CDC weekly hospitalization data, 2020-2024) ---
WEEKLY_HOSPITALIZATIONS = DATA_RAW / "weekly_covid_hospitalizations.csv"
FORECAST_STATE = "USA"  # aggregate national series by default; per-state also supported
FORECAST_TARGET_COLUMN = "total_adm_all_covid_confirmed_past_7days"
FORECAST_HORIZON_WEEKS = 8
FORECAST_BACKTEST_MIN_TRAIN_WEEKS = 100
FORECAST_BACKTEST_STEP_WEEKS = 4

# --- Optimization source (real HHS reported hospital capacity snapshot) ---
HOSPITAL_CAPACITY = DATA_RAW / "hospital_capacity_by_state.csv"
