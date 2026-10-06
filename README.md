# Hospital Operations Intelligence Suite

**One real problem, five ML disciplines, one pipeline — not five disconnected toy demos.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Version](https://img.shields.io/badge/version-0.3.0-blueviolet)
![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Data](https://img.shields.io/badge/data-100%25%20real%20public-orange)

I kept seeing portfolio projects that claim to cover "classification,
regression, NLP, forecasting, and optimization" and turn out to be five
unrelated notebooks bolted together for a resume bullet point. I wanted
to build the thing those projects are pretending to be: one coherent
business story, end to end, where the five disciplines actually feed
into each other. The story I picked is a hospital operations team that

1. **forecasts** how much incoming patient demand to expect,
2. **classifies** which patients are at risk of 30-day readmission,
3. **regresses** how long each patient is likely to stay,
4. **mines clinical documentation** with NLP to triage/route notes and
   estimate urgency, and then
5. **optimizes** bed and ICU-bed allocation across departments given all
   of the above, subject to real capacity constraints.

I used real, public data for every piece of it, and I held myself to
proper train/test hygiene, bootstrap confidence intervals, and paired
significance testing throughout — including when the results weren't
flattering. A couple of the findings below only showed up because I
went looking for reasons to doubt my own numbers.

---

## Table of contents

- [Headline findings](#headline-findings)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Real data sources](#real-data-sources)
- [Project layout](#project-layout)
- [Infra notes: corporate proxy + Hugging Face Xet CDN](#infra-notes-corporate-proxy--hugging-face-xet-cdn)
- [Limitations](#limitations-read-before-you-quote-a-metric)
- [License](#license)

---

I kept a full, warts-and-all build diary as I went — every decision,
every gotcha, every dead end — written so future-me can still make
sense of it five years from now. That's [`BUILD_LOG.md`](BUILD_LOG.md)
if you want the unfiltered version of how this actually came together.

---

## Headline findings

Full methodology and discussion: [`report/REPORT.md`](report/REPORT.md).

| # | Finding | Evidence |
|---|---|---|
| 1 | **Gradient boosting beats simpler baselines on both classification and regression, and it's not noise.** Readmission ROC-AUC 0.685 vs. logistic regression's 0.669 (p<0.0001); length-of-stay MAE 1.74 vs. 1.82 days (p<0.0001). I selected both models via a held-out validation split, never the test set the numbers above come from. | `results/classification_metrics.json`, `results/regression_metrics.json` |
| 2 | **Assuming fixed calendar seasonality actively hurts admissions forecasting.** Seasonal-naive (MAE 21,525) is nearly 3.4x *worse* than plain naive (MAE 6,368) because COVID hospitalization waves are epidemic-driven, not calendar-periodic. A damped-trend Holt-Winters model beats naive significantly (p<0.0001) by adapting locally instead. | `results/forecasting_metrics.json` |
| 3 | **A biased upstream NLP model was silently manufacturing a fake capacity crisis downstream.** My original NLP classifier over-predicted "Surgery" for 44% of cases (severe class imbalance, never corrected). Fixing that with `class_weight="balanced"` spread the predicted case-mix out far more realistically — and the optimization LP's bed shortfall (which used to concentrate on Emergency vs. Surgery) **disappeared entirely**: every department is now fully served within real capacity. The "OR insight" I thought I'd found wasn't really about beds at all; it was an upstream data-quality bug wearing an operations-research costume. | `results/optimization_result.json`, `results/nlp_metrics.json` |
| 4 | **Class-weighting nearly doubled NLP macro-F1** on a genuinely hard, severely imbalanced 40-class problem — 0.191 to 0.279 (95% CI 0.231-0.332), selected via validation, not cherry-picked from the test set. Still modest in absolute terms (some specialties have only a handful of training examples), but a real, measured improvement instead of a limitation I just wrote down and ignored. | `results/nlp_metrics.json` |

## Architecture

```
data/raw/ (real, fetched once)
  diabetes_readmission_{train,test}.csv   <- UCI Diabetes 130-US Hospitals (imodels-cleaned)
  mtsamples_{train,test}.parquet          <- MTSamples clinical transcriptions (40 specialties)
  weekly_covid_hospitalizations.csv       <- HHS/CDC weekly hospitalization metrics, 2020-2024
  hospital_capacity_by_state.csv          <- HHS reported hospital bed/ICU capacity

        |                    |                     |                      |
        v                    v                     v                      v
  classification/      regression/            nlp/                  forecasting/
  readmission risk     length of stay      specialty + urgency     weekly admissions
  (Logistic Reg /       (Linear Reg /       (TF-IDF + Linear SVM   (naive / seasonal-naive
   Grad Boosting)         Grad Boosting)       + heuristic score)    / Holt-Winters, backtested)
        |                    |                     |                      |
        +--------------------+---------------------+----------------------+
                                       |
                                       v
                             optimization/allocate_beds.py
                     PuLP LP: minimize urgency-weighted unmet bed
                     demand, subject to real bed/ICU capacity and
                     an Emergency service-floor constraint
                                       |
                                       v
                     results/*.json + SQLite run history
                                       |
                                       v
                api/main.py -- FastAPI + HTMX + Tailwind + Chart.js dashboard
```

I built `src/stats/bootstrap.py` once and reused it everywhere: it's
the same bootstrap-CI / paired-significance-test machinery behind the
classification, regression, NLP, and forecasting evaluations, so every
"model A beats model B" claim in this repo is backed by the same rigor
— not vibes, and not four different ad-hoc stats approaches depending
on which notebook I was in that day.

## Quickstart

```bash
cd hospital-ops-suite
uv venv
uv pip install -r requirements.txt

# Fetch all real datasets (curl-based; see infra notes below for why)
.venv/Scripts/python -m src.fetch_data        # Windows
# .venv/bin/python -m src.fetch_data          # macOS/Linux

# Train everything and run the optimization, in dependency order (~2 min)
.venv/Scripts/python -m src.pipeline.orchestrate

# Launch the dashboard
.venv/Scripts/python -m uvicorn api.main:app --reload
# open http://127.0.0.1:8000

# Run the test suite (fast, no retraining required)
.venv/Scripts/python -m pytest tests/ -v

# Optional: rebuild + re-execute the walkthrough notebook with real outputs
.venv/Scripts/python -m uv pip install -r requirements-notebook.txt
.venv/Scripts/python notebooks/build_walkthrough.py
.venv/Scripts/python -m jupyter execute notebooks/walkthrough.ipynb --inplace
```

The dashboard also has a **"Re-run full pipeline"** button I added so
I could re-run all five modules live from the browser and log the run
to SQLite, instead of dropping back to a terminal every time I wanted
to see it work end to end.

## Real data sources

| Module | Dataset | Real? | Size |
|---|---|---|---|
| Classification + Regression | UCI "Diabetes 130-US Hospitals for Years 1999-2008" (imodels-cleaned) | Real, public | 81,410 train / test held out |
| NLP | MTSamples de-identified medical transcriptions, 40 specialties | Real, public | 4,499 train / 500 test |
| Forecasting | HHS/CDC weekly COVID-19 hospitalization metrics, national aggregate | Real, public | 195 weeks, 2020-08 to 2024-04 |
| Optimization capacity | HHS reported hospital bed/ICU capacity by state | Real, public | 54 states/territories |

I didn't use synthetic data anywhere in the core pipeline. The one
heuristic I deliberately kept — and labeled everywhere it shows up, in
code, in the dashboard, and in the report — is the NLP module's
*urgency score*, because MTSamples simply has no ground-truth urgency
label to train against.

## Project layout

```
src/
  config.py                       # all paths/constants, single source of truth
  fetch_data.py                   # curl-based downloader for all 4 real datasets
  data_prep.py                    # diabetes train/test loading + leakage-safe feature splits
  stats/bootstrap.py              # shared bootstrap CI + paired significance testing
  classification/train_readmission.py
  regression/train_los.py
  nlp/{labels,urgency,train_specialty}.py
  forecasting/train_admissions_forecast.py
  optimization/{departments,allocate_beds}.py
  pipeline/orchestrate.py         # runs all 5 modules + logs a run summary to SQLite

db/database.py                    # SQLite schema + run-history logging
api/main.py                       # FastAPI app (dashboard + JSON API + re-run endpoint)
api/templates/                    # Jinja2 + HTMX partials, one per module, Tailwind/Chart.js via CDN
tests/                            # 25 tests, isolated per module, no retraining required
report/REPORT.md                  # full methodology, results, and honest limitations
BUILD_LOG.md                      # chronological build diary: every decision, gotcha, and dead-end
results/                          # metrics JSON, forecast CSV, SQLite run history (generated)
notebooks/                        # executed walkthrough notebook with real outputs
models/                           # saved model artifacts (generated, gitignored)
data/raw/                         # fetched datasets (generated, gitignored)
```

## Infra notes: corporate proxy + Hugging Face Xet CDN

Building this behind Walmart's corporate NTLM proxy ran me into the
exact same wall I'd already hit on a previous project (RAGBench) — I'm
writing it down again here because it's genuinely useful "why did my
data pipeline silently fail" material, not a one-off fluke I got
unlucky with:

- Python's `requests`/`huggingface_hub` don't negotiate NTLM auth
  through the proxy; every download 407s. `curl` on Windows negotiates
  it transparently via SSPI. My fix: `src/fetch_data.py` shells out to
  `curl` for everything.
- Hugging Face's newer **Xet CDN backend** (`us.aws.cdn.hf.co`,
  `*.xethub.hf.co`) is unreachable through the proxy at all, even via
  `curl` — I confirmed this with `curl -v` tracing the redirect chain
  from `huggingface.co/.../resolve/main/...`. My fix: Walmart's
  Artifactory mirrors Hugging Face Hub datasets/models server-side,
  bypassing Xet entirely:
  `hub-huggingfaceml-release-remote/datasets/{repo_id}/resolve/main/{file}`.
  Small, non-LFS files (like a dataset's README, or small CSVs on
  `huggingface.co` directly) don't hit Xet and download fine either way.
- Not every dataset is available through that mirror — I learned to
  always probe with a `HEAD` request before committing to a data
  source, rather than finding out halfway through a download.

## Limitations (read before you quote a metric)

- **Regression leakage caveat**: the diabetes encounter table is
  discharge-complete, not a day-1 admission snapshot. Features like
  `num_procedures` are partly a *consequence* of a longer stay. The
  R^2 I report (0.40) measures how well discharge-summary features
  explain length of stay, not true admission-day forecastability.
- **NLP urgency scoring is a documented heuristic**, not a trained or
  clinically validated model — MTSamples has no ground-truth urgency
  label. Treat it like a business rule, read `src/nlp/urgency.py`
  directly, and swap in real triage-acuity data before trusting it for
  anything beyond a demo.
- **The forecasting series is historical/deprecated CDC reporting**
  (ended 2024-04-27). It demonstrates a rigorous backtesting
  methodology on real data, not a live forecast of anything happening
  today.
- **Optimization capacity numbers are averages, not one real
  hospital's actual bed count.** I derived them by dividing real
  cumulative HHS-reported bed-days by real reporting-day coverage,
  averaged across all 54 reporting states/territories — a
  representative "typical facility" estimate, clearly documented in
  `src/optimization/allocate_beds.py`, not a specific hospital's
  numbers.
- **NLP class imbalance**: some of the 40 MTSamples specialties have
  only a handful of training examples. `class_weight='balanced'`
  (chosen via validation, not cherry-picked from the test set)
  measurably helps — macro-F1 nearly doubles (0.191 to 0.279) — but
  the remaining ~0.28 is an honest reflection of how hard 40-way
  classification with this little data per rare class really is, not
  something I'm going to paper over by only reporting accuracy.

## License

MIT.
