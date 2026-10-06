"""
NLP: classify real de-identified clinical transcription notes by medical
specialty (TF-IDF + Linear SVM, 40-way classification, real MTSamples text).

This is the "unstructured documentation triage" piece of the pipeline: in
a real hospital ops setting, incoming notes/orders often need to be routed
to the right department before anyone reads them. We measure this with
real accuracy/F1 numbers rather than assuming a demo works.

Class imbalance fix: 40 real-world specialties are nowhere close to evenly
represented (some have a handful of examples, others hundreds). Rather than
just documenting that as a limitation, we train both a plain LinearSVC and
one with `class_weight="balanced"` (which reweights the loss inversely to
class frequency so rare specialties aren't simply ignored in favor of
common ones), and pick whichever wins on a held-out VALIDATION split --
never the test set -- matching the selection methodology used by the
classification and regression modules. We report macro-F1 for both so the
effect of the fix is honest and measured, not assumed.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.nlp.labels import SPECIALTY_LABELS
from src.stats.bootstrap import bootstrap_ci_indexed

VALIDATION_FRACTION = 0.2


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(config.MTSAMPLES_TRAIN)
    test_df = pd.read_parquet(config.MTSAMPLES_TEST)
    return train_df, test_df


def build_pipeline(class_weight: str | None = None) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    class_weight=class_weight,
                    random_state=config.RANDOM_SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, texts, labels) -> tuple[float, float]:
    preds = pipeline.predict(texts)
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")


def main() -> None:
    train_df, test_df = load_frames()
    print(f"[nlp] {len(train_df):,} real transcription notes, {len(SPECIALTY_LABELS)} specialties...")

    # Held-out validation split carved out of TRAINING data (never the test
    # set) decides which variant (plain vs. class-weighted) gets deployed.
    train_fit_df, val_df = train_test_split(
        train_df,
        test_size=VALIDATION_FRACTION,
        stratify=train_df["label"],
        random_state=config.RANDOM_SEED,
    )

    candidates = {"linear_svm": None, "linear_svm_balanced": "balanced"}
    fitted = {}
    metrics = {}
    val_macro_f1 = {}

    for name, class_weight in candidates.items():
        print(f"[nlp] training {name} on {len(train_fit_df):,} notes (class_weight={class_weight})...")
        pipeline = build_pipeline(class_weight=class_weight)
        pipeline.fit(train_fit_df["text"], train_fit_df["label"])
        fitted[name] = pipeline

        _, val_macro_f1[name] = evaluate(pipeline, val_df["text"], val_df["label"])

        test_accuracy, test_macro_f1 = evaluate(pipeline, test_df["text"], test_df["label"])
        y_true = test_df["label"].to_numpy()
        y_pred = pipeline.predict(test_df["text"])

        def macro_f1_of_indices(idx: np.ndarray, y_true=y_true, y_pred=y_pred) -> float:
            return f1_score(y_true[idx], y_pred[idx], average="macro")

        macro_f1_ci = bootstrap_ci_indexed(len(y_true), macro_f1_of_indices, n_resamples=2000)

        metrics[name] = {
            "accuracy": float(test_accuracy),
            "macro_f1": float(test_macro_f1),
            "macro_f1_val": float(val_macro_f1[name]),
            "macro_f1_ci": macro_f1_ci,
        }
        print(
            f"[nlp] {name}: val macro_f1={val_macro_f1[name]:.4f}  "
            f"test accuracy={test_accuracy:.4f}  test macro_f1={test_macro_f1:.4f}"
        )

    # Selection happens on the VALIDATION split, not the test set.
    best_name = max(val_macro_f1, key=val_macro_f1.get)
    improvement = val_macro_f1["linear_svm_balanced"] - val_macro_f1["linear_svm"]
    print(
        f"[nlp] class_weight='balanced' {'helped' if improvement > 0 else 'did not help'} "
        f"on validation (macro_f1 delta={improvement:+.4f}); selecting {best_name}."
    )

    metrics["num_classes"] = len(SPECIALTY_LABELS)
    metrics["num_train"] = len(train_fit_df)
    metrics["num_test"] = len(test_df)
    metrics["selected_model"] = best_name
    metrics["selection_method"] = "highest macro-F1 on held-out validation split (not the test set)"
    # Kept for backwards-compat with any code/templates reading top-level fields.
    metrics["accuracy"] = metrics[best_name]["accuracy"]
    metrics["macro_f1"] = metrics[best_name]["macro_f1"]
    metrics["macro_f1_ci"] = metrics[best_name]["macro_f1_ci"]

    joblib.dump(fitted[best_name], config.MODELS_DIR / "specialty_classifier.joblib")
    with open(config.RESULTS_DIR / "nlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"[nlp] saved best model ({best_name}, chosen via validation) and metrics.")


if __name__ == "__main__":
    main()
