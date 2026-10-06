from src.nlp.urgency import SPECIALTY_BASE_URGENCY, keyword_boost, urgency_score
from src.optimization.departments import DEPARTMENTS


def test_departments_have_required_keys_and_valid_icu_fraction():
    for name, spec in DEPARTMENTS.items():
        assert "diabetes_cols" in spec and len(spec["diabetes_cols"]) >= 1
        assert "nlp_label" in spec
        assert 0.0 <= spec["icu_fraction"] <= 1.0


def test_every_department_nlp_label_has_an_urgency_weight():
    for name, spec in DEPARTMENTS.items():
        assert spec["nlp_label"] in SPECIALTY_BASE_URGENCY


def test_diabetes_columns_look_like_one_hot_flags():
    for name, spec in DEPARTMENTS.items():
        for col in spec["diabetes_cols"]:
            assert col.startswith("medical_specialty:")


def test_urgency_score_is_bounded():
    for label in SPECIALTY_BASE_URGENCY:
        score = urgency_score(label, "routine follow up visit")
        assert 0.0 <= score <= 1.0


def test_urgency_score_rises_with_critical_keywords():
    baseline = urgency_score("Office Notes", "patient feels fine, routine checkup")
    boosted = urgency_score("Office Notes", "patient in acute critical condition, code blue called, sepsis")
    assert boosted > baseline


def test_keyword_boost_is_capped():
    text = "stat critical emergent acute severe unstable cardiac arrest respiratory failure hemorrhage sepsis"
    assert keyword_boost(text) <= 0.25
