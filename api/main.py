"""
FastAPI + HTMX + Tailwind + Chart.js dashboard for the Hospital Operations
Intelligence Suite. Reads precomputed results/*.json (produced by
`python -m src.pipeline.orchestrate`) rather than retraining on every page
load -- training is the expensive part, serving the results shouldn't be.

Routes:
  GET  /                 full dashboard page
  GET  /api/metrics      raw JSON of every module's metrics (for tests/tools)
  POST /run-pipeline     re-runs the full pipeline synchronously (~1-2 min)
                         and returns the refreshed dashboard content (HTMX)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from db.database import get_recent_runs
from src import config
from src.optimization.departments import DEPARTMENTS

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Hospital Operations Intelligence Suite")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _load_json(name: str) -> dict | None:
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def gather_dashboard_context() -> dict:
    classification = _load_json("classification_metrics.json")
    regression = _load_json("regression_metrics.json")
    nlp = _load_json("nlp_metrics.json")
    forecasting = _load_json("forecasting_metrics.json")
    optimization = _load_json("optimization_result.json")

    forecast_csv = config.RESULTS_DIR / "admissions_forecast.csv"
    forecast_rows = []
    if forecast_csv.exists():
        import csv

        with open(forecast_csv) as f:
            forecast_rows = list(csv.DictReader(f))

    return {
        "classification": classification,
        "regression": regression,
        "nlp": nlp,
        "forecasting": forecasting,
        "optimization": optimization,
        "forecast_rows": forecast_rows,
        "forecast_labels": json.dumps([r["week_ending_date"] for r in forecast_rows]),
        "forecast_values": json.dumps([float(r["forecast_admissions"]) for r in forecast_rows]),
        "run_history": get_recent_runs(limit=10),
        "pipeline_has_run": classification is not None,
        **_optimization_capacity_context(optimization),
        **_optimization_chart_context(optimization),
    }


def _optimization_chart_context(optimization: dict | None) -> dict:
    """JSON-encoded arrays for the required-vs-allocated Chart.js bar chart."""
    if optimization is None:
        return {"optimization_dept_labels": "[]", "optimization_required": "[]", "optimization_allocated": "[]"}
    depts = optimization["departments"]
    return {
        "optimization_dept_labels": json.dumps(list(depts.keys())),
        "optimization_required": json.dumps([d["required_beds"] for d in depts.values()]),
        "optimization_allocated": json.dumps([d["allocated_beds"] for d in depts.values()]),
    }


def _optimization_capacity_context(optimization: dict | None) -> dict:
    """Real bed/ICU utilization totals, derived the same way the LP itself
    computes ICU usage (allocated_beds * department icu_fraction) -- kept
    here instead of duplicated in the template so DEPARTMENTS stays the
    single source of truth for icu_fraction."""
    if optimization is None:
        return {"optimization_beds_used": 0, "optimization_icu_used": 0.0}
    depts = optimization["departments"]
    beds_used = sum(d["allocated_beds"] for d in depts.values())
    icu_used = sum(d["allocated_beds"] * DEPARTMENTS[name]["icu_fraction"] for name, d in depts.items())
    return {
        "optimization_beds_used": round(beds_used, 1),
        "optimization_icu_used": round(icu_used, 1),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    context = gather_dashboard_context()
    return templates.TemplateResponse(request, "dashboard.html", context)


@app.get("/api/metrics")
def api_metrics():
    return gather_dashboard_context()


@app.post("/run-pipeline", response_class=HTMLResponse)
def run_pipeline(request: Request):
    from src.pipeline.orchestrate import main as run_full_pipeline

    run_full_pipeline()
    context = gather_dashboard_context()
    return templates.TemplateResponse(request, "dashboard_content.html", context)
