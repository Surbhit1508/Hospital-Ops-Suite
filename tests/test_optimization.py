from src.optimization.allocate_beds import build_and_solve_lp
from src.optimization.departments import DEPARTMENTS


def _synthetic_inputs():
    capacity = {"total_beds": 100, "total_icu_beds": 10, "baseline_beds_used": 70}
    avg_los = {dept: 4.0 for dept in DEPARTMENTS}
    hospital_wide_avg_los = 4.0
    case_mix = {dept: 1.0 / len(DEPARTMENTS) for dept in DEPARTMENTS}
    pressure_multiplier = 1.0
    return capacity, avg_los, hospital_wide_avg_los, case_mix, pressure_multiplier


def test_lp_solves_to_optimal_with_ample_capacity():
    capacity, avg_los, hospital_wide_avg_los, case_mix, pressure = _synthetic_inputs()
    result = build_and_solve_lp(capacity, avg_los, hospital_wide_avg_los, case_mix, pressure)
    assert result["status"] == "Optimal"


def test_lp_never_exceeds_total_bed_capacity():
    capacity, avg_los, hospital_wide_avg_los, case_mix, pressure = _synthetic_inputs()
    result = build_and_solve_lp(capacity, avg_los, hospital_wide_avg_los, case_mix, pressure)
    total_allocated = sum(d["allocated_beds"] for d in result["departments"].values())
    assert total_allocated <= capacity["total_beds"] + 1e-6


def test_lp_never_exceeds_icu_capacity():
    capacity, avg_los, hospital_wide_avg_los, case_mix, pressure = _synthetic_inputs()
    result = build_and_solve_lp(capacity, avg_los, hospital_wide_avg_los, case_mix, pressure)
    icu_used = sum(
        d["allocated_beds"] * DEPARTMENTS[dept]["icu_fraction"] for dept, d in result["departments"].items()
    )
    assert icu_used <= capacity["total_icu_beds"] + 1e-6


def test_emergency_service_floor_is_always_fully_met():
    capacity, avg_los, hospital_wide_avg_los, case_mix, pressure = _synthetic_inputs()
    # Tight ICU capacity so the floor constraint actually has to bind.
    capacity["total_icu_beds"] = 2
    result = build_and_solve_lp(capacity, avg_los, hospital_wide_avg_los, case_mix, pressure)
    emergency = result["departments"]["Emergency"]
    assert emergency["shortfall_beds"] == 0
    assert emergency["allocated_beds"] >= emergency["required_beds"] - 1e-6


def test_scarce_capacity_produces_some_shortfall_elsewhere():
    capacity, avg_los, hospital_wide_avg_los, case_mix, pressure = _synthetic_inputs()
    capacity["total_beds"] = 5  # deliberately far too small
    result = build_and_solve_lp(capacity, avg_los, hospital_wide_avg_los, case_mix, pressure)
    total_shortfall = sum(d["shortfall_beds"] for d in result["departments"].values())
    assert total_shortfall > 0
