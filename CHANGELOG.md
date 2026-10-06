# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] - Fixed real methodology gaps: validation-based model selection, NLP class-weighting

### Fixed
- **Model-selection leakage** in classification, regression, and NLP: all
  three previously picked the "best" model using test-set metrics, then
  reported those same test-set metrics as the headline number -- a subtle
  but real form of leakage. Now each module carves a held-out validation
  split out of the *training* data, selects the deployed model purely on
  validation performance, and touches the test set exactly once, for
  reporting only.
- **NLP class imbalance was documented but never actually addressed.**
  `src/nlp/train_specialty.py` now trains both a plain `LinearSVC` and one
  with `class_weight="balanced"`, picks the winner via the validation
  split above, and reports both so the effect is measured, not assumed.
  Result: macro-F1 nearly doubled (0.191 to 0.279 on the test set).

### Changed
- **The optimization module's headline finding changed, for real, honest
  reasons.** The NLP class-weighting fix rebalanced the predicted case mix
  fed into the bed-allocation LP (Surgery's predicted share dropped from
  44% to 18.5%, redistributing to other departments). The previously
  reported bed shortfall (Emergency vs. Surgery) disappeared entirely --
  it turned out to be an artifact of the biased NLP classifier's
  case-mix predictions, not a real capacity constraint. `report/REPORT.md`
  section 6, the README headline findings, and the dashboard's
  optimization section were all rewritten to tell this (more interesting)
  story honestly instead of leaving a now-incorrect claim in place.
- Re-executed `notebooks/walkthrough.ipynb` with the corrected real
  outputs end to end.

## [0.2.0] - Notebook, docs polish, and dashboard visual overhaul

### Added
- `notebooks/walkthrough.ipynb` (+ `notebooks/build_walkthrough.py`): an
  executed walkthrough notebook, generated programmatically via
  `nbformat`/`nbclient` and run for real against the trained artifacts,
  covering all five modules with real baked-in outputs and two real
  matplotlib charts (forecast line chart, required-vs-allocated bed bars).
- `README.md`, `report/REPORT.md`, `LICENSE`: full project documentation
  -- architecture diagram, quickstart, real data source table, corporate
  proxy/Xet-CDN infra notes, and per-module honest limitations.
- `api/templates/_icons.html`: shared inline SVG icon macros (no external
  icon-font dependency) used across the dashboard.
- **Dashboard visual overhaul**: color-coded accent per discipline
  (indigo/violet/emerald/amber/rose) across summary cards and section
  headers, a sticky nav bar with jump-links, a real-time bed/ICU capacity
  utilization strip (computed from `DEPARTMENTS`, not hardcoded), a Chart.js
  grouped bar chart comparing required vs. allocated beds per department,
  per-department progress bars with shortfall/met badges, and
  significant/not-significant badges on every statistical comparison.

### Changed
- `api/main.py`: `gather_dashboard_context()` now derives real bed/ICU
  utilization totals and per-department chart data server-side (single
  source of truth via `src/optimization/departments.DEPARTMENTS`) instead
  of guessing at fields that don't exist in `optimization_result.json`.

## [0.1.0] - Initial release

### Added
- `src/fetch_data.py`: curl-based downloader for all 4 real datasets
  (UCI Diabetes 130-US Hospitals via imodels, MTSamples via galileo-ai,
  HHS/CDC weekly hospitalization metrics, HHS hospital capacity), routed
  through Walmart's Artifactory Hugging Face mirror to bypass the Xet
  CDN backend being unreachable through the corporate proxy.
- `src/classification/train_readmission.py`: Logistic Regression vs.
  Gradient Boosting for 30-day readmission risk, with paired bootstrap
  significance testing.
- `src/regression/train_los.py`: Linear Regression vs. Gradient
  Boosting for length-of-stay prediction, sharing the same source data
  and a documented leakage-avoidance split (`src/data_prep.py`).
- `src/nlp/`: TF-IDF + Linear SVM 40-way specialty classifier on real
  MTSamples transcriptions, plus a documented heuristic urgency scorer
  (`urgency.py`) used downstream by the optimization module.
- `src/forecasting/train_admissions_forecast.py`: naive / seasonal-naive
  / Holt-Winters weekly admissions forecasting with expanding-window
  walk-forward backtesting (not a single train/test split).
- `src/optimization/`: PuLP linear program allocating real hospital bed
  and ICU-bed capacity across 10 departments, fed by real outputs from
  all four other modules (case mix from NLP, LOS from regression data,
  demand pressure from forecasting, capacity from real HHS data), with
  an Emergency service-floor constraint.
- `src/stats/bootstrap.py`: shared bootstrap CI + paired significance
  testing utilities used by every module's evaluation.
- `src/pipeline/orchestrate.py`: runs all five modules end to end and
  logs a run summary to SQLite (`db/database.py`).
- `api/main.py`: FastAPI + HTMX + Tailwind + Chart.js dashboard reading
  precomputed results, with a live "re-run full pipeline" endpoint.
- 25 passing unit/integration tests (`tests/`), isolated per module,
  none requiring model retraining to run fast.
- `report/REPORT.md`: full methodology, results, and honest
  limitations per module.

### Notes
- All datasets are real and public; no synthetic data in the core
  pipeline. The one deliberately synthetic-flavored piece (NLP urgency
  scoring) is a documented heuristic, clearly labeled as such
  everywhere it appears.
