"""
Minimal SQLite persistence for pipeline run history.

Every time the pipeline is (re)run, we log one row summarizing all five
modules' headline metrics, so the dashboard can show a run-history trend
instead of just "the last time someone happened to train it."
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "results" / "pipeline_runs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    classification_roc_auc REAL,
    classification_selected_model TEXT,
    regression_mae_days REAL,
    regression_selected_model TEXT,
    nlp_macro_f1 REAL,
    forecasting_mae REAL,
    forecasting_selected_model TEXT,
    optimization_status TEXT,
    optimization_expected_admissions REAL,
    optimization_objective REAL
);
"""


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(SCHEMA)


def log_run(summary: dict) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (
                classification_roc_auc, classification_selected_model,
                regression_mae_days, regression_selected_model,
                nlp_macro_f1,
                forecasting_mae, forecasting_selected_model,
                optimization_status, optimization_expected_admissions, optimization_objective
            ) VALUES (:classification_roc_auc, :classification_selected_model,
                      :regression_mae_days, :regression_selected_model,
                      :nlp_macro_f1,
                      :forecasting_mae, :forecasting_selected_model,
                      :optimization_status, :optimization_expected_admissions, :optimization_objective)
            """,
            summary,
        )


def get_recent_runs(limit: int = 20) -> list[dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY run_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
