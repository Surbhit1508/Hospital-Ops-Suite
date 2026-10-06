"""
Optimization: allocate a hospital's real staffed-bed capacity across
departments to minimize urgency-weighted unmet demand, subject to total
bed and ICU-bed capacity constraints.

This is where the other four modules' outputs become LP inputs:
  - Classification/Regression source data -> real avg length-of-stay per
    department (mean `time_in_hospital` for diabetes encounters flagged
    with that `medical_specialty`).
  - NLP module -> real case-mix shares (specialty distribution predicted
    by the trained classifier on held-out MTSamples notes) and urgency
    weights (src/nlp/urgency.py heuristic).
  - Forecasting module -> a demand-pressure scenario (% change implied by
    the national admissions forecast vs. the last observed real week).
  - Real HHS hospital capacity data -> the actual bed/ICU-bed ceiling.

Little's Law (L = lambda * W) is used twice, honestly labeled both times:
once to back out a plausible weekly admission rate from real average
occupied beds and average LOS, and once to convert weekly bed-day demand
back into an average concurrent-bed requirement per department.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import pulp

from src import config, data_prep
from src.optimization.departments import DEPARTMENTS


def real_hospital_capacity() -> dict[str, float]:
    """Average per-facility bed/ICU-bed capacity and utilization, derived
    from real HHS-reported cumulative totals divided by real reporting-day
    coverage, averaged across all 54 reporting states/territories."""
    df = pd.read_csv(config.HOSPITAL_CAPACITY)
    avg_beds = (df["inpatient_beds"] / df["inpatient_beds_coverage"]).mean()
    avg_icu_beds = (df["total_staffed_adult_icu_beds"] / df["total_staffed_adult_icu_beds_coverage"]).mean()
    avg_beds_used = (df["inpatient_beds_used"] / df["inpatient_beds_used_coverage"]).mean()
    return {
        "total_beds": round(avg_beds),
        "total_icu_beds": round(avg_icu_beds),
        "baseline_beds_used": round(avg_beds_used),
    }


def real_avg_los_by_department(train_df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Real mean `time_in_hospital` per department, computed from the same
    diabetes encounter table used for the classification/regression tasks."""
    hospital_wide_avg_los = train_df[config.REGRESSION_TARGET].mean()
    avg_los = {}
    for dept, spec in DEPARTMENTS.items():
        mask = train_df[spec["diabetes_cols"]].sum(axis=1) > 0
        subset = train_df.loc[mask, config.REGRESSION_TARGET]
        avg_los[dept] = float(subset.mean()) if len(subset) > 0 else float(hospital_wide_avg_los)
    return avg_los, float(hospital_wide_avg_los)


def real_case_mix_shares() -> dict[str, float]:
    """Real specialty distribution, predicted by the trained NLP classifier
    on held-out MTSamples test notes, restricted to our department list."""
    from src.nlp.labels import SPECIALTY_LABELS

    model = joblib.load(config.MODELS_DIR / "specialty_classifier.joblib")
    test_df = pd.read_parquet(config.MTSAMPLES_TEST)
    predicted_ids = model.predict(test_df["text"])
    predicted_labels = pd.Series([SPECIALTY_LABELS[i] for i in predicted_ids])

    nlp_label_by_dept = {dept: spec["nlp_label"] for dept, spec in DEPARTMENTS.items()}
    counts = {dept: int((predicted_labels == label).sum()) for dept, label in nlp_label_by_dept.items()}
    total = sum(counts.values()) or 1
    return {dept: count / total for dept, count in counts.items()}


def demand_pressure_multiplier() -> float:
    """% change implied by the real forecasted next week vs. the last real
    observed week, from the forecasting module's saved output."""
    forecast_df = pd.read_csv(config.RESULTS_DIR / "admissions_forecast.csv")
    history_df = pd.read_csv(config.WEEKLY_HOSPITALIZATIONS)
    history_df = history_df[history_df["state"] == config.FORECAST_STATE].sort_values("week_ending_date")
    last_actual = history_df[config.FORECAST_TARGET_COLUMN].iloc[-1]
    first_forecast = forecast_df["forecast_admissions"].iloc[0]
    if last_actual == 0:
        return 1.0
    return float(first_forecast / last_actual)


def build_and_solve_lp(
    capacity: dict[str, float],
    avg_los: dict[str, float],
    hospital_wide_avg_los: float,
    case_mix: dict[str, float],
    pressure_multiplier: float,
) -> dict:
    from src.nlp.urgency import SPECIALTY_BASE_URGENCY

    # Little's Law: weekly admission rate implied by real average occupied
    # beds and real hospital-wide average LOS (in weeks).
    baseline_weekly_admissions = capacity["baseline_beds_used"] / (hospital_wide_avg_los / 7)
    expected_weekly_admissions = baseline_weekly_admissions * pressure_multiplier

    demand_bed_days = {
        dept: expected_weekly_admissions * case_mix[dept] * avg_los[dept] for dept in DEPARTMENTS
    }
    # Little's Law again, other direction: bed-days per week -> average
    # concurrent beds required.
    required_beds = {dept: demand_bed_days[dept] / 7 for dept in DEPARTMENTS}
    urgency = {dept: SPECIALTY_BASE_URGENCY.get(DEPARTMENTS[dept]["nlp_label"], 0.4) for dept in DEPARTMENTS}

    problem = pulp.LpProblem("bed_allocation", pulp.LpMinimize)
    allocated = {d: pulp.LpVariable(f"allocated_{i}", lowBound=0) for i, d in enumerate(DEPARTMENTS)}
    shortfall = {d: pulp.LpVariable(f"shortfall_{i}", lowBound=0) for i, d in enumerate(DEPARTMENTS)}

    for d in DEPARTMENTS:
        problem += shortfall[d] >= required_beds[d] - allocated[d]

    problem += pulp.lpSum(allocated[d] for d in DEPARTMENTS) <= capacity["total_beds"]
    problem += (
        pulp.lpSum(allocated[d] * DEPARTMENTS[d]["icu_fraction"] for d in DEPARTMENTS) <= capacity["total_icu_beds"]
    )

    # Real-world service floor: Emergency admissions cannot simply be turned
    # away the way an elective specialty's demand can be deferred, so its
    # required beds are a hard constraint rather than something the cost
    # function is free to trade off against other departments.
    problem += allocated["Emergency"] >= required_beds["Emergency"]

    problem += pulp.lpSum(urgency[d] * shortfall[d] for d in DEPARTMENTS)
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "status": pulp.LpStatus[problem.status],
        "expected_weekly_admissions": round(expected_weekly_admissions, 1),
        "demand_pressure_multiplier": round(pressure_multiplier, 4),
        "departments": {
            d: {
                "required_beds": round(required_beds[d], 2),
                "allocated_beds": round(allocated[d].value(), 2),
                "shortfall_beds": round(shortfall[d].value(), 2),
                "urgency_weight": urgency[d],
                "case_mix_share": round(case_mix[d], 4),
                "avg_length_of_stay_days": round(avg_los[d], 2),
            }
            for d in DEPARTMENTS
        },
        "total_beds_available": capacity["total_beds"],
        "total_icu_beds_available": capacity["total_icu_beds"],
        "objective_urgency_weighted_shortfall": round(pulp.value(problem.objective), 3),
    }


def main() -> None:
    train_df, _ = data_prep.load_diabetes_frames()

    capacity = real_hospital_capacity()
    avg_los, hospital_wide_avg_los = real_avg_los_by_department(train_df)
    case_mix = real_case_mix_shares()
    pressure_multiplier = demand_pressure_multiplier()

    print(f"[optimization] real capacity: {capacity}")
    print(f"[optimization] demand pressure multiplier from forecast: {pressure_multiplier:.4f}")

    result = build_and_solve_lp(capacity, avg_los, hospital_wide_avg_los, case_mix, pressure_multiplier)

    with open(config.RESULTS_DIR / "optimization_result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"[optimization] status={result['status']}  expected admissions/week={result['expected_weekly_admissions']}")
    for dept, d in result["departments"].items():
        print(f"  {dept}: required={d['required_beds']}  allocated={d['allocated_beds']}  shortfall={d['shortfall_beds']}")


if __name__ == "__main__":
    main()
