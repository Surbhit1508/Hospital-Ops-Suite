# Hospital Operations Intelligence Suite -- Full Report

## 1. Problem framing

A hospital operations team needs to answer five connected questions
every week:

1. How much patient demand should we expect? (**forecasting**)
2. Which currently-discharged patients are likely to bounce back within
   30 days? (**classification**)
3. How long is each patient likely to stay? (**regression**)
4. What is the incoming case mix, and how urgent is it, based on
   clinical documentation? (**NLP**)
5. Given all of the above, how should we allocate a limited number of
   beds and ICU beds across departments? (**optimization**)

This report walks through each stage with the same rigor: proper
train/test hygiene, bootstrap confidence intervals, and paired
significance testing wherever a model-vs-model claim is made -- plus an
honest accounting of where each result should NOT be over-interpreted.

## 2. Classification -- 30-day readmission risk

**Data**: UCI "Diabetes 130-US Hospitals for Years 1999-2008" (cleaned
by the imodels team), 81,410 real encounters in the train split, a
separate real held-out test split. 150 features: demographics, admission
type/source, discharge disposition, diagnosis groupings, medication
changes, and lab results.

**Models**: Logistic Regression (scaled features) vs. Gradient Boosting
(200 trees, depth 3). Which model gets deployed is decided on an 80/20
validation split carved out of the training data -- never on the test
set the table below reports. (Earlier versions of this project picked
the "winner" using test-set ROC-AUC directly, which is a subtle form of
leakage; fixed once identified.)

| Model | ROC-AUC | PR-AUC | F1 | Accuracy (95% CI) |
|---|---|---|---|---|
| Logistic Regression | 0.6694 | 0.6249 | 0.5312 | -- |
| Gradient Boosting | **0.6849** | **0.6429** | **0.5552** | -- |

Paired bootstrap test on per-sample correctness (10,000 resamples):
Gradient Boosting beats Logistic Regression by 0.0107 accuracy
(p<0.0001, significant at 0.05).

**Reading these numbers honestly**: ROC-AUC in the high 0.60s is
consistent with published work on this exact dataset -- 30-day
readmission from administrative/EHR-summary features alone is a
genuinely hard prediction problem, not something a bigger model fixes.
The finding here isn't "we solved readmission prediction," it's "a
tree-based model extracts a modest but statistically real amount of
extra signal over a linear baseline."

## 3. Regression -- length of stay

**Data**: the same encounter table, predicting `time_in_hospital`
(1-14 days). `readmitted` is excluded from features -- it's a later
event than length of stay and would leak future information backwards.

| Model | MAE (days) | RMSE | R^2 |
|---|---|---|---|
| Linear Regression | 1.815 | 2.387 | 0.3589 |
| Gradient Boosting | **1.738** | **2.299** | **0.4055** |

Same validation-based selection methodology as classification (see
section 2): the model saved and used downstream is chosen on a held-out
validation split, not the test set reported here.

Paired bootstrap test on absolute error: Gradient Boosting's error is
significantly lower than Linear Regression's (p<0.0001).

**Honest leakage discussion** (important -- read this before quoting
R^2 anywhere): this is a discharge-complete encounter table, not a
pre-admission triage snapshot. Features like `num_procedures`,
`num_medications`, and `num_lab_procedures` are partly *caused by* a
longer stay, not purely predictive of it in a causal, day-1 sense. A
production length-of-stay model would need an admission-time feature
snapshot (vitals, diagnosis on arrival, comorbidity flags known at
intake) rather than the full discharge summary. The R^2 of 0.40 should
be read as "how well do discharge-summary features explain length of
stay" -- a legitimate descriptive/retrospective question -- not "how
well can we predict length of stay on day one," which this dataset
cannot answer.

## 4. NLP -- clinical note specialty classification + urgency scoring

**Data**: MTSamples, a real corpus of de-identified medical
transcription samples labeled with one of 40 medical specialties.
4,499 train / 500 test.

**Model**: TF-IDF (unigrams + bigrams, 20k features) + Linear SVM,
compared plain vs. with `class_weight="balanced"`. Which variant gets
deployed is decided on a held-out validation split, not the test set
below -- same methodology as classification and regression.

| Model | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
| Linear SVM (plain) | 0.3200 | 0.1912 | (0.1583, 0.2314) |
| Linear SVM (`class_weight='balanced'`) | **0.3340** | **0.2793** | **(0.2311, 0.3317)** |

**This was a real, measured fix, not a documented shrug.** An earlier
version of this project trained only the plain variant and reported its
modest macro-F1 as an accepted limitation of severe class imbalance.
Actually trying `class_weight="balanced"` -- which reweights the SVM's
loss inversely to class frequency instead of implicitly favoring common
specialties -- raised validation macro-F1 from 0.258 to 0.400
(observed on the held-out validation split, which is what decided the
selection) and test macro-F1 from 0.191 to 0.279. Still a genuinely hard
task (some of the 40 specialties have only a handful of training
examples, and the confidence interval is wide), but no longer a gap left
unaddressed just because it was inconvenient.

**Why macro-F1 is still modest, honestly**: 40 classes with severe
real-world imbalance -- some specialties (e.g., "Autopsy", "IME-QME-Work
Comp etc.") have only a handful of training examples, while others
("Surgery", "Consult - History and Phy.") have hundreds. Macro-F1
penalizes poor performance on rare classes equally to common ones, so it
surfaces this imbalance honestly rather than being flattered by accuracy
alone. A production system would likely also consolidate rare
specialties into broader buckets or use a hierarchical classifier on top
of the class-weighting fix already applied here.

**Urgency scoring** (`src/nlp/urgency.py`) is a transparent, documented
heuristic, not a trained model: a base acuity weight per specialty
(e.g., Emergency Room Reports = 1.00, Letters = 0.05) plus a small
keyword-match boost (capped at +0.25) for terms like "critical",
"sepsis", or "code blue". MTSamples has no ground-truth urgency label,
so this cannot be validated against real outcomes -- it exists purely
to give the optimization stage a plausible, inspectable input signal,
the same way a business rule would.

## 5. Forecasting -- weekly hospital admissions

**Data**: HHS/CDC weekly national COVID-19 hospitalization metrics,
195 weeks (2020-08-08 to 2024-04-27). Target: total admissions in the
past 7 days, national aggregate.

**Methodology**: expanding-window walk-forward backtesting. Starting
from 100 weeks of history, forecast the next 8 weeks, slide the origin
forward 4 weeks, repeat until the series is exhausted, and pool the
absolute errors across every origin and every step. This avoids the
classic time-series evaluation trap of one lucky (or unlucky) train/test
split.

| Method | MAE | 95% CI |
|---|---|---|
| Naive (last value) | 6,368.3 | (5,641.9, 7,100.8) |
| Seasonal-naive (lag 52 weeks) | 21,524.8 | (18,463.1, 24,933.3) |
| Holt-Winters (damped trend, no seasonality) | **4,663.8** | **(4,037.0, 5,351.5)** |

Holt-Winters beats naive by 1,704.5 MAE (p<0.0001).

**Finding worth dwelling on**: seasonal-naive is dramatically *worse*
than even the plain naive baseline. COVID-19 hospitalization waves are
driven by variant emergence and behavior change, not the calendar --
assuming "this week looks like the same week last year" actively hurts
here, because last year's wave timing has nothing to do with this
year's. A model that adapts to the recent local trend (Holt-Winters
with a damped trend and no seasonal component) comfortably outperforms
both naive baselines. This is exactly the kind of assumption-testing a
forecasting exercise should surface instead of blindly reaching for a
seasonal model because the data happens to be weekly.

**Scope note**: this series stopped being updated on 2024-04-27 (HHS
discontinued the reporting requirement). It's used here to demonstrate
a rigorous backtesting methodology on real historical data, not as a
live forecast of current admissions.

## 6. Optimization -- bed and ICU-bed allocation

**Formulation**: a linear program (PuLP) that allocates a hospital's
total staffed beds across 10 departments to minimize urgency-weighted
unmet demand, subject to:

- total allocated beds <= real average per-facility bed capacity,
- ICU-bed-equivalent usage (department allocation x department ICU
  fraction) <= real average per-facility ICU capacity,
- a hard service-floor constraint: Emergency's required beds must
  always be fully met (Emergency departments cannot simply defer
  patients the way an elective specialty can).

**Real inputs, from the other four modules**:

- **Capacity**: average per-facility staffed beds (131) and ICU beds
  (13), derived by dividing real HHS-reported cumulative bed-days by
  real reporting-day coverage, averaged across all 54 reporting
  states/territories.
- **Case mix**: the real distribution of specialties predicted by the
  NLP classifier on held-out MTSamples test notes, restricted to the
  10 departments that exist in both source datasets.
- **Length of stay per department**: real mean `time_in_hospital` from
  the diabetes encounter table, grouped by `medical_specialty`.
- **Demand pressure**: the ratio of the forecasting module's first
  forecasted week to the last real observed week (0.9066 in this run --
  a mild expected decline).
- **Baseline weekly admission rate**: backed out from real average
  occupied beds and real hospital-wide average length of stay via
  Little's Law (L = lambda x W, so lambda = L / W).

**Result** (with the Emergency service floor):

| Department | Case mix | Avg LOS | Urgency | Required beds | Allocated | Shortfall |
|---|---|---|---|---|---|---|
| Cardiology | 15.8% | 3.52d | 0.88 | 10.97 | 10.97 | 0 |
| Orthopedics | 16.5% | 3.96d | 0.58 | 12.96 | 12.96 | 0 |
| Nephrology | 4.3% | 5.09d | 0.75 | 4.36 | 4.36 | 0 |
| Gastroenterology | 12.2% | 4.62d | 0.70 | 11.15 | 11.15 | 0 |
| Psychiatry | 3.2% | 6.49d | 0.55 | 4.04 | 4.04 | 0 |
| Surgery | 18.5% | 4.79d | 0.90 | 17.53 | 17.53 | 0 |
| Obstetrics/Gynecology | 10.6% | 3.06d | 0.68 | 6.43 | 6.43 | 0 |
| Urology | 6.7% | 3.37d | 0.60 | 4.46 | 4.46 | 0 |
| General/Internal Medicine | 6.3% | 4.51d | 0.45 | 5.62 | 5.62 | 0 |
| Emergency | 5.9% | 4.37d | 1.00 | 5.10 | 5.10 | 0 |

Total required: 82.6 of 131 available beds (63%); ICU-equivalent usage:
~11.3 of 13 available (87%). **Zero shortfall anywhere.** Status: Optimal.

**The finding that actually matters here is not about beds at all --
it's about what changed between the previous version of this report and
this one.** The version of this project before the NLP class-imbalance
fix had a biased specialty classifier that over-predicted "Surgery" for
44% of real notes (see section 4). That skewed case-mix fed straight
into this LP as a real input, concentrating a disproportionate share of
bed demand -- and ICU demand specifically -- onto Surgery. The LP then
correctly, faithfully solved the (mis-specified) problem it was given:
it shortchanged Emergency by 1.67 beds to protect Surgery's
urgency-weighted allocation, because Surgery looked artificially
dominant. That result read like a legitimate operations-research
insight ("raw priority score doesn't tell you where to spend a scarce
resource") -- and the *reasoning* was genuinely correct -- but the
*scenario* it was reasoning about was manufactured by an upstream data
quality bug, not a real capacity constraint.

Once `class_weight="balanced"` corrected the NLP model's case-mix
predictions to something far more realistic (Surgery's share dropped
from 44% to 18.5%, spread out to Orthopedics, Gastroenterology, and
Obstetrics/Gynecology instead), the LP's inputs changed, and the
"shortage" evaporated -- because it was never a real shortage of physical
beds, it was a shortage of a fictional amount of Surgery demand that the
biased classifier had invented. **The general lesson survives even
though the specific number didn't: in a pipeline where one model's
output becomes another model's input, an upstream bug doesn't just
degrade accuracy metrics -- it can silently fabricate a downstream
decision problem that looks completely legitimate until you trace it
back.** That is arguably a more useful thing to have caught here than
the original finding was.

## 7. Overall limitations

- See the per-module honesty callouts above; they are not afterthoughts,
  they are load-bearing parts of this report.
- All five modules are demonstrated independently and then wired
  together through real, computed hand-off values (case mix, LOS,
  demand pressure, capacity) -- but they are not trained *jointly*,
  and the "hospital" in the optimization stage is a statistically
  representative composite, not one real, named facility.
- Sample sizes vary a lot across modules (81k encounters vs. 500 NLP
  test notes vs. 195 forecasting weeks) and confidence intervals are
  reported specifically so nobody has to guess how much to trust each
  number.
