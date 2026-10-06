"""
Shared loading/splitting logic for the diabetes readmission encounter table.

This ONE real dataset (UCI "Diabetes 130-US Hospitals for Years 1999-2008",
cleaned by the imodels team) drives TWO different supervised tasks:

  - Classification: predict `readmitted` (1 = readmitted, 0 = not)
  - Regression:     predict `time_in_hospital` (length of stay, in days)

Honest leakage note (see report/REPORT.md "Limitations" for the full
discussion): this is an encounter-level table recorded at/after discharge,
not a pre-admission triage snapshot. That makes `readmitted` a legitimate
classification feature for the regression task's target (LOS is known
before you'd find out about readmission), so we exclude it there. But
several regression features (num_procedures, num_medications,
num_lab_procedures) are themselves partially *caused by* a longer stay, so
the regression R^2 should be read as "how well do discharge-summary
features explain LOS", not "how well can we predict LOS from day one".
"""
from __future__ import annotations

import pandas as pd

from src import config


def load_diabetes_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the real train/test CSVs as-is."""
    train_df = pd.read_csv(config.DIABETES_TRAIN)
    test_df = pd.read_csv(config.DIABETES_TEST)
    return train_df, test_df


def classification_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features/target for readmission classification.

    `time_in_hospital` IS kept as a feature here: by the time you'd assess
    30-day readmission risk (at or after discharge), the actual length of
    stay is already known, so this is not leakage for THIS task.
    """
    y = df[config.CLASSIFICATION_TARGET].astype(int)
    x = df.drop(columns=[config.CLASSIFICATION_TARGET])
    return x, y


def regression_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features/target for length-of-stay regression.

    `readmitted` is dropped: it is a FUTURE event relative to length of
    stay and would leak the classification target into the regression
    features.
    """
    y = df[config.REGRESSION_TARGET].astype(float)
    x = df.drop(columns=[config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET])
    return x, y
