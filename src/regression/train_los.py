"""
Regression: predict length of stay (`time_in_hospital`, in days) from the
same real encounter table used for readmission classification.

See src/data_prep.py for the leakage discussion: `readmitted` is dropped
from the features here since it's a later event than length of stay.

Methodology note (matches the classification module's fix): which model
gets saved and used downstream is decided on a held-out VALIDATION split
carved out of the training data, never on the test set. The test set is
touched exactly once, purely for reporting.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, data_prep
from src.stats.bootstrap import bootstrap_ci, paired_bootstrap_test

VALIDATION_FRACTION = 0.2


def build_models() -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            [("scale", StandardScaler()), ("reg", LinearRegression())]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "reg",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=config.RANDOM_SEED,
                    ),
                )
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    abs_errors = np.abs(y_test.values - y_pred)
    mae_ci = bootstrap_ci(abs_errors, np.mean)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mae_ci": mae_ci,
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
        "abs_errors": abs_errors,
    }


def main() -> None:
    train_df, test_df = data_prep.load_diabetes_frames()
    x_train, y_train = data_prep.regression_xy(train_df)
    x_test, y_test = data_prep.regression_xy(test_df)

    # Carve a validation split out of the TRAINING data (never the test set)
    # -- this is what "best model" gets decided on.
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        random_state=config.RANDOM_SEED,
    )

    models = build_models()
    fitted = {}
    metrics = {}
    val_mae = {}

    for name, model in models.items():
        print(f"[regression] training {name} on {len(x_fit):,} real encounters...")
        model.fit(x_fit, y_fit)
        fitted[name] = model

        val_mae[name] = mean_absolute_error(y_val, model.predict(x_val))

        result = evaluate_model(model, x_test, y_test)
        metrics[name] = {k: v for k, v in result.items() if k != "abs_errors"}
        metrics[name]["mae_val"] = val_mae[name]
        print(
            f"[regression] {name}: val MAE={val_mae[name]:.3f}  test MAE={result['mae']:.3f} days  "
            f"RMSE={result['rmse']:.3f}  R2={result['r2']:.4f}"
        )

    lr_errors = np.abs(y_test.values - fitted["linear_regression"].predict(x_test))
    gb_errors = np.abs(y_test.values - fitted["gradient_boosting"].predict(x_test))
    # Lower error is better, so test whether GB's error is significantly LOWER.
    sig_test = paired_bootstrap_test(lr_errors, gb_errors, np.mean)
    metrics["gradient_boosting_vs_linear_regression"] = sig_test
    print(f"[regression] LR-error minus GB-error (test)={sig_test['observed_diff']:.4f} days  p={sig_test['p_value']:.4f}")

    # Model selection happens on the VALIDATION split, not the test set.
    best_name = min(val_mae, key=val_mae.get)
    joblib.dump(fitted[best_name], config.MODELS_DIR / "los_regressor.joblib")
    metrics["selected_model"] = best_name
    metrics["selection_method"] = "lowest MAE on held-out validation split (not the test set)"

    with open(config.RESULTS_DIR / "regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"[regression] saved best model ({best_name}, chosen via validation) and metrics.")


if __name__ == "__main__":
    main()
