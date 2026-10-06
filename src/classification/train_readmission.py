"""
Classification: predict 30-day-ish hospital readmission risk.

Trains a baseline (Logistic Regression, interpretable) and a stronger model
(Gradient Boosting), then uses paired bootstrap significance testing to
check whether the "stronger" model is actually, statistically, better --
not just numerically higher on one run.

Methodology note (this used to be a real bug, now fixed): which model gets
saved and used downstream by the optimization module is decided on a
held-out VALIDATION split carved out of the training data -- never on the
test set. The test set is touched exactly once, purely for reporting, and
never feeds back into any decision. Selecting "best" model based on test
performance and then reporting that same test performance as the headline
number is a subtle form of leakage; with only two candidates the effect is
small, but there's no reason to accept it when a validation split is free.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, data_prep
from src.stats.bootstrap import bootstrap_ci, paired_bootstrap_test

VALIDATION_FRACTION = 0.2


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=config.RANDOM_SEED,
                    ),
                )
            ]
        ),
    }


def per_sample_correct(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true == y_pred).astype(float)


def evaluate_model(model: Pipeline, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    correctness = per_sample_correct(y_test.values, y_pred)
    acc_ci = bootstrap_ci(correctness, np.mean)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "accuracy_ci": acc_ci,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def main() -> None:
    train_df, test_df = data_prep.load_diabetes_frames()
    x_train, y_train = data_prep.classification_xy(train_df)
    x_test, y_test = data_prep.classification_xy(test_df)

    # Carve a validation split out of the TRAINING data (never the test set)
    # -- this is what "best model" gets decided on.
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        stratify=y_train,
        random_state=config.RANDOM_SEED,
    )

    models = build_models()
    fitted = {}
    metrics = {}
    val_roc_auc = {}

    for name, model in models.items():
        print(f"[classification] training {name} on {len(x_fit):,} real encounters...")
        model.fit(x_fit, y_fit)
        fitted[name] = model

        val_roc_auc[name] = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])

        result = evaluate_model(model, x_test, y_test)
        metrics[name] = {k: v for k, v in result.items() if k not in ("y_proba", "y_pred")}
        metrics[name]["roc_auc_val"] = val_roc_auc[name]
        print(
            f"[classification] {name}: val ROC-AUC={val_roc_auc[name]:.4f}  "
            f"test ROC-AUC={result['roc_auc']:.4f}  PR-AUC={result['pr_auc']:.4f}  F1={result['f1']:.4f}"
        )

    # Paired significance test on the TEST set: is gradient boosting's
    # per-sample correctness actually better than logistic regression's, or
    # just noisy variance? Purely observational -- doesn't drive selection.
    lr_correct = per_sample_correct(y_test.values, fitted["logistic_regression"].predict(x_test))
    gb_correct = per_sample_correct(y_test.values, fitted["gradient_boosting"].predict(x_test))
    sig_test = paired_bootstrap_test(gb_correct, lr_correct, np.mean)
    metrics["gradient_boosting_vs_logistic_regression"] = sig_test
    print(f"[classification] GB vs LR test accuracy diff={sig_test['observed_diff']:.4f}  p={sig_test['p_value']:.4f}")

    # Model selection happens on the VALIDATION split, not the test set.
    best_name = max(val_roc_auc, key=val_roc_auc.get)
    joblib.dump(fitted[best_name], config.MODELS_DIR / "readmission_classifier.joblib")
    metrics["selected_model"] = best_name
    metrics["selection_method"] = "highest ROC-AUC on held-out validation split (not the test set)"

    with open(config.RESULTS_DIR / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"[classification] saved best model ({best_name}, chosen via validation) and metrics.")


if __name__ == "__main__":
    main()
