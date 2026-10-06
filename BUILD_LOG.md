# Build Log -- Hospital Operations Intelligence Suite

**Purpose of this file**: if I open this repo in
five years and think "wait, why did I do it this way?" -- this document
has the answer. It's not the README (that's for recruiters) and it's not
`report/REPORT.md` (that's methodology and results). This is the messy,
honest, chronological record of *how the thing actually got built*,
including the parts that didn't work the first time.

---

## 1. The idea, and why this shape

I wanted a resume project covering classification, regression, NLP,
forecasting, and optimization -- but I'd seen too many projects that
"cover" five disciplines as five disconnected toy demos (iris
classification, titanic survival, sentiment analysis, etc.) and called
it a day. The first and most important decision I made was refusing to
do that, and instead insisting on **one coherent business narrative**
where each discipline's output becomes the next one's input. That's the
entire reason this repo is shaped the way it is: `Forecasting ->
Classification -> Regression -> NLP -> Optimization`, all landing in
one dashboard.

Before writing any code, I locked in four choices on purpose, instead
of letting them drift as I went:
- **Domain**: Healthcare Operations
- **Data**: Real public datasets only, no synthetic data, no exceptions
- **Deliverable**: A full app (FastAPI + dashboard), not just notebooks
- **Rigor**: Full statistical rigor -- bootstrap CIs, significance
  testing, not eyeballed metrics

That last one matters more than it sounds: it's the reason `src/stats/
bootstrap.py` exists as shared infrastructure from day one, and I
carried it over in spirit from an earlier project of mine (RAGBench)
that used the exact same paired-bootstrap-significance methodology.
Reusing that pattern was deliberate -- consistent rigor across my
projects is itself a signal worth having.

## 2. Finding real data behind a corporate proxy (this took longer than any single module)

Worth documenting in detail because it will absolutely happen again on
the next project.

**First attempt**: `archive.ics.uci.edu` (the canonical home of the
Diabetes 130-US Hospitals dataset). Blocked outright by a URL allowlist
on my machine -- separate from and in addition to Walmart's corporate
proxy. No amount of proxy configuration fixes an allowlist block.
Lesson: when a domain is blocked, check whether the same data is
mirrored somewhere on an already-allowed host before fighting the
block.

**The fix**: Hugging Face Hub mirrors an enormous amount of public
research data, and `huggingface.co` itself (plus `raw.githubusercontent.com`)
worked once I set `HTTP_PROXY`/`HTTPS_PROXY` to
`http://sysproxy.wal-mart.com:8080`. I found the same UCI dataset,
already cleaned, at `imodels/diabetes-readmission` on the HF Hub.

**Second problem**: even with the proxy working for small files, larger
LFS-backed files (the actual 49MB+ CSVs) kept failing. I traced this
with `curl -v` and found the redirect chain went to Hugging Face's
newer **Xet CDN backend** (`us.aws.cdn.hf.co`, `*.xethub.hf.co`), which
the corporate proxy couldn't reach *at all* -- not a 407, just
unreachable.

**The actual fix**: Walmart's internal Artifactory instance mirrors
Hugging Face Hub content server-side, which sidesteps Xet entirely
because Artifactory fetches the bytes on its own end and hands them
back over a domain the proxy allows. The exact working URL pattern
(this is the single most useful line in this whole log, I'm memorizing
it):

```
https://generic.ci.artifacts.walmart.com/artifactory/api/huggingfaceml/hub-huggingfaceml-release-remote/datasets/{repo_id}/resolve/main/{file}
```

Gotchas inside that gotcha:
- It's `hub-huggingfaceml-release-remote`, NOT `hub-hf-release-remote`
  (the shorter name looked more plausible and 404'd every time).
- Dataset repos need the literal `datasets/` path segment before
  `repo_id`; model repos don't.
- Small non-LFS files (a dataset's README, small CSVs directly off
  `huggingface.co`) don't hit Xet at all and download fine without any
  of this -- always try the direct route first with a `HEAD` request.

**Why `curl` and not Python's `requests`/`huggingface_hub`**: Windows
`curl` negotiates NTLM proxy auth transparently via SSPI using my
logged-in Windows credentials. Python's HTTP stacks don't do this
automatically, so every download attempt through `requests` or
`huggingface_hub` failed with `407 authenticationrequired` even with
the proxy env vars set correctly. `src/fetch_data.py` shells out to
`curl` for exactly this reason -- it's not a stylistic choice, it's the
only thing that actually authenticates.

**How I actually found the three real datasets**: searched the HF Hub
API for candidates, then used the **datasets-server API**
(`datasets-server.huggingface.co`, reachable directly, no proxy dance
needed) to pull structured schema/label info that wasn't in any README
-- that's how I got the 40 real MTSamples specialty label names, since
the parquet file only stored integer label IDs.

## 3. Environment setup nuances

- `uv venv` + Walmart's internal PyPI mirror
  (`pypi.ci.artifacts.walmart.com`) for all installs, per standing
  convention -- never the public PyPI URL directly, it's blocked/slow.
- Hit a hardlink/copy filesystem error partway through the first
  `uv pip install` run (uv defaults to hardlinking from its global cache,
  which fails across certain filesystem/drive boundaries on Windows).
  Fixed by deleting and recreating `.venv` from scratch rather than
  fighting the linking mode -- faster than debugging it.
- Python 3.13.5 was whatever `uv venv` picked up locally; nothing in this
  project requires that specific version, 3.11+ is the real floor (per
  the README badge), pinned loosely on purpose (YAGNI -- no reason to
  lock a portfolio project to an exact patch version).

## 4. Why the directory structure looks like this

```
src/{classification,regression,nlp,forecasting,optimization,pipeline,stats}/
```

One subpackage per discipline, each with its own `train_*.py` entrypoint
that can run standalone (`python -m src.classification.train_readmission`)
*or* get orchestrated together. This was deliberate: during development,
being able to iterate on one module without re-running the other four
(some of which take real wall-clock time, like NLP's TF-IDF fit over
20k features) saved a lot of time. `src/pipeline/orchestrate.py` is a
thin sequencer I added *after* all five modules worked independently --
it does nothing clever, it just calls `main()` on each in dependency
order and then logs a summary row to SQLite.

`src/stats/bootstrap.py` living outside any single discipline's folder
is intentional: it's genuinely shared infrastructure (used by
classification, regression, NLP, and forecasting), and duplicating
bootstrap-CI logic four times would have violated DRY for no benefit.

## 5. Module-by-module nuances worth remembering

### Classification + Regression (`src/classification/`, `src/regression/`)
- Both tasks share one real dataset (the diabetes encounter table) via
  `src/data_prep.py`, which is the single source of truth for the
  leakage-avoidance logic: `time_in_hospital` is a *legitimate* feature
  for readmission classification (by discharge time, LOS is already
  known) but `readmitted` must be *excluded* from regression features
  (it's a future event relative to LOS). Get this backwards and you've
  silently leaked the wrong direction.
- Gradient Boosting hyperparameters (`n_estimators=200, max_depth=3,
  learning_rate=0.1`) were never tuned via grid/random search -- they're
  reasonable defaults, documented as such. If asked in an interview "did
  you tune this," the honest answer is no, and that's an acknowledged
  gap (see section 9).
- **The validation-leakage bug and fix** (this is the big one, see
  section 9 for the full story): I originally decided "which model do
  we deploy" by comparing test-set metrics directly -- a subtle leakage
  since the same test set then got reported as the headline number.
  Fixed it by carving an 80/20 validation split out of the *training*
  data specifically for the selection decision, touching the test set
  exactly once, for reporting only, never for choosing.

### NLP (`src/nlp/`)
- Specialty labels (40 real medical specialties) came from HF's
  datasets-server `/info` endpoint, not the dataset's README (which
  didn't document them) -- see section 2.
- `urgency.py` is explicitly a **heuristic, not a trained model** -- base
  acuity weight per specialty (Emergency Room Reports = 1.00, Letters =
  0.05) plus a small regex keyword boost (capped at +0.25) for terms
  like "critical," "sepsis," "code blue." MTSamples has zero
  ground-truth urgency labels, so there was never a way to validate this
  against real outcomes. It exists purely to give the optimization stage
  a plausible, inspectable input -- I documented it everywhere it
  appears (code comments, dashboard, report) specifically so nobody
  mistakes it for something it isn't.
- **The class-weight fix and its cascading consequences** (the single
  most interesting thing that happened in this whole project -- full
  story in section 9): `class_weight="balanced"` on the `LinearSVC`
  nearly doubled macro-F1 (0.191 -> 0.279), AND it changed the NLP
  model's predicted case-mix distribution enough to completely change
  the optimization module's headline finding. Upstream fixes have
  downstream consequences you won't predict in advance -- that's not a
  cliche, it's exactly what happened here.

### Forecasting (`src/forecasting/`)
- The real gotcha: the HHS weekly series ends on **Saturdays**, not
  Tuesdays. My first attempt used `series.asfreq("W-TUE")`, which
  silently produced all-NaN resampled data (wrong day-of-week anchor =
  no rows matched = quietly empty, no error thrown). Fixed to `"W-SAT"`.
  This class of bug -- silent NaN cascade from a wrong frequency string
  -- is exactly the kind of thing that's invisible until you print
  intermediate shapes, which is how I caught it.
- Expanding-window walk-forward backtesting (start at 100 weeks of
  history, forecast 8 weeks ahead, slide the origin forward 4 weeks,
  repeat, pool all the absolute errors) was my way of avoiding the
  classic time-series mistake of evaluating on one lucky/unlucky
  train-test split. `FORECAST_HORIZON_WEEKS=8` and the slide-by-4 step
  are in `src/config.py` as named constants for exactly this reason --
  future-me might want to widen the backtest window, and it should be a
  one-line change, not an archaeology dig.
- `statsmodels`' Holt-Winters implementation throws noisy
  `ConvergenceWarning`s on this series constantly; I suppressed them via
  `warnings.filterwarnings` rather than letting them get lost as silent
  console noise -- the distinction matters, one is a deliberate choice,
  the other is just tolerating clutter.

### Optimization (`src/optimization/`)
- `departments.py` is a deliberately curated 10-department taxonomy that
  bridges the diabetes dataset's `medical_specialty` one-hot columns and
  the MTSamples NLP labels. Only departments existing in *both* source
  datasets are included -- "Letters" and "Autopsy" are real MTSamples
  labels but obviously not real inpatient departments, so trimming them
  was a deliberate modeling choice on my part, documented in the file's
  own docstring so it doesn't look like an oversight later.
- `icu_fraction` per department (Emergency=0.30, Cardiology=0.25,
  Surgery=0.20, down to Psychiatry=0.02) are assumption values, not
  measured from data -- there's no real dataset that reports per-department
  ICU-bed fractions at this granularity. Documented as assumptions, not
  presented as measured facts.
- Real capacity numbers (131 beds, 13 ICU beds) are derived by dividing
  cumulative HHS-reported bed-days by real reporting-day coverage,
  averaged across all 54 reporting states/territories -- an "average
  representative facility," explicitly not any one real hospital's
  actual numbers. I call this out in three separate places (code
  comments, README limitations, REPORT.md) because it's the kind of
  nuance that's easy to accidentally overstate.
- Baseline weekly admission rate is backed out from real average
  occupied beds and real hospital-wide average LOS via **Little's Law**
  (L = lambda * W, so lambda = L / W) -- a genuine queueing-theory
  identity, not a made-up formula, applied to real inputs.
- The Emergency service-floor constraint (`allocated["Emergency"] >=
  required_beds["Emergency"]`) went in specifically because the first
  unconstrained LP run shortchanged Emergency in favor of Surgery -- see
  section 9 for what that finding turned out to actually mean once I
  fixed the NLP bug.

## 6. Dashboard build nuances

- FastAPI + HTMX + Tailwind (CDN) + Chart.js (CDN) -- deliberately no
  build step, no npm, no bundler. For a project meant to be cloned and
  run in under two minutes, adding a JS toolchain would have been
  pure friction for zero real benefit (YAGNI).
- Hit a real Starlette API-version bug early: `templates.TemplateResponse
  ("name", context)` (old positional-arg order) threw `TypeError:
  unhashable type: 'dict'` on the version of Starlette I actually had
  installed, which expects `(request, "name", context)` instead. Fixed
  both call sites in `api/main.py`. This is the kind of thing that
  changes silently between library versions and just breaks with a
  confusing error -- worth remembering the exact symptom (`unhashable
  type: 'dict'`) in case it happens again elsewhere.
- Port conflicts happened twice across the build (stale `uvicorn`
  processes holding port 8731 after a restart) -- resolved via
  `netstat -ano | findstr :PORT` then `taskkill /PID ... /F`. Standard
  Windows dev loop friction, not a code bug, but worth remembering the
  exact commands instead of re-deriving them each time.
- The visual overhaul (icons, color-coded sections, progress bars, the
  Chart.js grouped bar chart for optimization) came in a later pass,
  after the functional version already worked end to end. Order
  mattered: get the real data pipeline correct first, make it look good
  second. Building the pretty version first would have meant redesigning
  around data structures that were still changing.
- `_optimization.html`'s original bed/ICU capacity utilization strip
  tried to read `optimization.total_beds_allocated` /
  `optimization.total_icu_beds_used` from the JSON -- fields that never
  actually existed in `optimization_result.json`. I caught it before
  shipping by actually reading the JSON file instead of guessing field
  names from memory. Fixed by computing both totals server-side in
  `api/main.py` from `optimization["departments"]` and the real
  `DEPARTMENTS` mapping (single source of truth for `icu_fraction`),
  rather than inventing new fields in the JSON output that would then
  need to stay in sync with the LP's own internal math.

## 7. Testing philosophy

25 tests, all fast (< 5 seconds total), none requiring model retraining
or network access. This was a deliberate constraint, not an accident:
- `test_bootstrap.py` uses synthetic data with known properties (e.g., a
  constant-1.0 array vs. a coin flip) specifically so the *statistical
  test itself* can be verified independent of any real dataset.
- `test_optimization.py` builds synthetic capacity/case-mix inputs
  rather than depending on the real (large, slow-to-produce)
  `optimization_result.json` -- the LP's *logic* (constraints never
  violated, Emergency floor always met, shortfall appears under scarce
  capacity) is what's being tested, not any specific real-world number.
- `test_api.py` uses FastAPI's `TestClient` against whatever `results/`
  actually contains on disk at test time -- it checks structural
  correctness (right top-level keys present) rather than asserting
  exact metric values, since those values are expected to shift
  slightly every time the pipeline is rerun (different random seeds
  aside, real model training has some run-to-run variance even with
  seeds fixed, depending on library versions).

## 8. Docs, versioning, and git nuances

- `report/REPORT.md` (methodology) is intentionally separate from
  `README.md` (pitch/quickstart) -- different audiences, different
  read-time budgets. A recruiter skims the README in 90 seconds; someone
  who wants to actually evaluate the statistics reads REPORT.md.
- `CHANGELOG.md` uses semantic-ish version numbers (0.1.0, 0.2.0, 0.3.0)
  even though there's no `pyproject.toml`/package to actually version --
  the version lives in the CHANGELOG and a matching annotated git tag
  (`git tag -a v0.X.0`), not in code. This was a genuine gap I caught
  retroactively: the 0.2.0 dashboard-overhaul work initially shipped
  with zero CHANGELOG entry or tag, and I only noticed while double
  checking whether the version info was current -- worth remembering to
  check this proactively next time, not just when it nags at me.
- `LINKEDIN_POST.md` is deliberately **not** tracked in git (added to
  `.gitignore`) -- it's a personal draft artifact, not project source.
  It got accidentally swept into a commit once via `git add -A` and had
  to be explicitly untracked afterward. Lesson: personal/non-project
  files need to be in `.gitignore` *before* the first `git add -A`, not
  after.
- GitHub Enterprise push used **cached Windows Credential Manager
  credentials** for `gecgithub01.walmart.com` (left over from a prior
  project of mine), not `gh` CLI auth -- `gh auth login` needs an
  interactive device-code browser flow I didn't want to deal with mid-
  build, so I created the repo manually via the browser instead, and
  `git push` alone worked fine once the empty remote existed, using
  credentials git already had cached. `gh` CLI auth was never actually
  needed for this project.

## 9. The big one: a data-quality bug wearing an operations-research costume

This deserves its own section because it's the single most valuable
thing that happened in this project, and it happened by accident.

**How it started**: I did a self-critique pass -- sat down and
deliberately asked myself "what would I change if I were a working data
scientist reviewing this code critically?" -- and it surfaced two real,
concrete issues:
1. Classification/regression/forecasting all selected their "best" model
   using the same test set they then reported metrics from -- mild but
   real leakage.
2. The NLP module documented severe 40-class imbalance as an accepted
   limitation without ever trying an obvious, nearly-free mitigation
   (`class_weight="balanced"`).

**What should have been a small, contained fix**: add a validation
split for model selection (classification/regression/NLP), and try
class-weighting on the NLP model, reporting honestly whether it helped.

**What actually happened**: `class_weight="balanced"` worked --
macro-F1 nearly doubled (0.191 -> 0.279). But it also meaningfully
changed *which specialties the NLP model predicts on real notes*.
Specifically, the original (biased) model over-predicted "Surgery" for
44% of real test notes -- a majority-class bias baked in by the
imbalance the fix was designed to address. Once corrected, Surgery's
predicted share dropped to 18.5%, redistributing more realistically
across Orthopedics, Gastroenterology, and Obstetrics/Gynecology.

That predicted case-mix distribution is a **real, direct input** to the
optimization LP (`src/optimization/allocate_beds.py` pulls it straight
from the saved NLP model's predictions on held-out notes). So when the
case-mix changed, the LP's inputs changed -- and the finding I'd
previously reported ("the ICU-bed constraint binds; without an
Emergency service-floor rule, cost-minimization shortchanges Emergency
in favor of Surgery") **completely disappeared**. Post-fix, every
department is fully served, zero shortfall, ICU utilization ~87%, total
beds ~63%.

**The honest conclusion, written up in `report/REPORT.md` section 6**:
my original finding's *reasoning* was entirely correct given its
inputs. The LP faithfully, correctly solved the problem it was handed.
The problem itself was fabricated by an upstream data-quality bug that
had nothing to do with beds, ICU capacity, or hospital operations at
all -- it was a biased text classifier. In a pipeline where one model's
output becomes another model's input, an upstream bug doesn't just
degrade an accuracy number quietly in a corner -- it can manufacture an
entirely plausible-looking downstream decision problem that survives
scrutiny right up until someone traces it back to its source.

**Why this matters for five-years-from-now me**: this is a better,
truer story than the one I originally shipped, and I only found it
because (a) I fixed a real methodology bug instead of leaving it alone,
and (b) I reran the full pipeline end-to-end afterward instead of
assuming "the fix only touches the NLP module, nothing else needs
re-checking." I rewrote every doc that referenced the old finding
(README headline table, REPORT.md section 6, the dashboard's
optimization callout, the walkthrough notebook) rather than leaving it
stale -- a half-fixed pipeline with contradicting docs would have been
worse than either the bug or the fix alone.

## 10. Known gaps, left deliberately unfixed (as of v0.3.0)

Documented here so "did I just forget this" never has to be re-litigated:
- No hyperparameter tuning anywhere (all models use reasonable-but-arbitrary defaults).
- No probability calibration on the readmission classifier (ROC-AUC says nothing about whether "0.7" means "70% actually get readmitted").
- The optimization LP takes the forecast's point estimate, not its confidence interval -- demand uncertainty isn't propagated into the allocation decision.
- No cross-validation (k-fold) anywhere -- one validation split per module, chosen for simplicity over maximal robustness.
- No experiment tracking beyond the SQLite run-history table (no MLflow/W&B).
- No CI/CD running tests automatically on push.
All of these are legitimate "next steps if this were becoming a real production system," explicitly scoped out of a resume/portfolio-sized project rather than accidentally missed.

## 11. Quick-reference: exact final numbers (as of v0.3.0)

| Module | Metric | Value |
|---|---|---|
| Classification | GB ROC-AUC (test) | 0.6849 |
| Classification | LR ROC-AUC (test) | 0.6694 |
| Classification | GB vs LR significance | p < 0.0001 |
| Regression | GB MAE (test) | 1.738 days |
| Regression | LR MAE (test) | 1.815 days |
| Regression | GB vs LR significance | p < 0.0001 |
| NLP | Balanced SVM macro-F1 (test) | 0.2793 (95% CI 0.231-0.332) |
| NLP | Plain SVM macro-F1 (test) | 0.1912 |
| Forecasting | Holt-Winters MAE | 4,663.8 |
| Forecasting | Naive MAE | 6,368.3 |
| Forecasting | Seasonal-naive MAE | 21,524.8 |
| Optimization | Status | Optimal, zero shortfall |
| Optimization | Bed / ICU utilization | ~63% / ~87% |

If these numbers ever drift after a pipeline rerun (they will, slightly
-- library versions change, and gradient boosting has some run-to-run
variance even with a fixed seed), that's expected. What should NOT drift
without a doc update is which model wins, whether a finding is
significant, and whether the optimization status is Optimal.

## 12. If you're reading this before touching the code again

Run the actual pipeline before trusting anything in this file or in
`report/REPORT.md` -- numbers get stale, this log does not self-update:

```bash
cd hospital-ops-suite
uv venv
uv pip install -r requirements.txt
.venv/Scripts/python -m src.fetch_data
.venv/Scripts/python -m src.pipeline.orchestrate
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python -m uvicorn api.main:app --reload
```
