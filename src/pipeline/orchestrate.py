"""
Run the full pipeline end to end, in dependency order:

  1. Classification (readmission risk)      -> models/readmission_classifier.joblib
  2. Regression (length of stay)             -> models/los_regressor.joblib
  3. NLP (specialty classification)          -> models/specialty_classifier.joblib
  4. Forecasting (weekly admissions)         -> results/admissions_forecast.csv
  5. Optimization (bed allocation)           -> results/optimization_result.json
                                                 (depends on 1-4's saved outputs)

Then logs a one-row summary of all five modules' headline metrics to
SQLite (results/pipeline_runs.db) so the dashboard can show a run-history
trend, not just "whatever the JSON files currently say."

Run `python -m src.fetch_data` first if data/raw/ is empty.
"""
from __future__ import annotations

import json

from db.database import log_run
from src import config
from src.classification import train_readmission
from src.forecasting import train_admissions_forecast
from src.nlp import train_specialty
from src.optimization import allocate_beds
from src.regression import train_los


def _load_json(name: str) -> dict:
    with open(config.RESULTS_DIR / name) as f:
        return json.load(f)


def log_run_summary() -> None:
    classification = _load_json("classification_metrics.json")
    regression = _load_json("regression_metrics.json")
    nlp = _load_json("nlp_metrics.json")
    forecasting = _load_json("forecasting_metrics.json")
    optimization = _load_json("optimization_result.json")

    clf_model = classification["selected_model"]
    reg_model = regression["selected_model"]
    fc_model = forecasting["selected_model"]

    log_run(
        {
            "classification_roc_auc": classification[clf_model]["roc_auc"],
            "classification_selected_model": clf_model,
            "regression_mae_days": regression[reg_model]["mae"],
            "regression_selected_model": reg_model,
            "nlp_macro_f1": nlp["macro_f1"],
            "forecasting_mae": forecasting[fc_model]["mae"],
            "forecasting_selected_model": fc_model,
            "optimization_status": optimization["status"],
            "optimization_expected_admissions": optimization["expected_weekly_admissions"],
            "optimization_objective": optimization["objective_urgency_weighted_shortfall"],
        }
    )


def main() -> None:
    steps = [
        ("classification", train_readmission.main),
        ("regression", train_los.main),
        ("nlp", train_specialty.main),
        ("forecasting", train_admissions_forecast.main),
        ("optimization", allocate_beds.main),
    ]
    for name, fn in steps:
        print(f"\n===== [{name}] =====")
        fn()

    log_run_summary()
    print("\nPipeline complete. See results/ for metrics, models/ for artifacts, results/pipeline_runs.db for history.")


if __name__ == "__main__":
    main()
