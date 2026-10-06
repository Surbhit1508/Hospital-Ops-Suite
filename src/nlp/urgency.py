"""
Heuristic clinical urgency scoring.

IMPORTANT/HONEST NOTE: MTSamples has no ground-truth "urgency" label, so
this is NOT a trained/validated model -- it's a transparent, documented
heuristic that turns (predicted specialty + keyword signals) into a 0-1
"how urgently should this note's patient be prioritized for bed/staff
allocation" score. It exists to give the optimization stage a plausible,
inspectable input signal from the NLP stage, not to make a clinical
claim. Treat it the same way you'd treat a business rule, not a model.
"""
from __future__ import annotations

import re

# Baseline acuity weight per specialty: rough proxy for "how urgent is care
# in this department typically", 0 (routine/administrative) to 1 (critical).
# These are illustrative assumptions, documented here so they're inspectable
# and easy to challenge/replace with real triage-acuity data later.
SPECIALTY_BASE_URGENCY = {
    "Emergency Room Reports": 1.00,
    "Neurosurgery": 0.95,
    "Surgery": 0.90,
    "Cardiovascular / Pulmonary": 0.88,
    "Hematology - Oncology": 0.80,
    "Nephrology": 0.75,
    "Gastroenterology": 0.70,
    "Neurology": 0.70,
    "Obstetrics / Gynecology": 0.68,
    "Pediatrics - Neonatal": 0.68,
    "Urology": 0.60,
    "Orthopedic": 0.58,
    "Endocrinology": 0.55,
    "Rheumatology": 0.50,
    "ENT - Otolaryngology": 0.48,
    "Consult - History and Phy.": 0.55,
    "Discharge Summary": 0.40,
    "SOAP / Chart / Progress Notes": 0.45,
    "General Medicine": 0.45,
    "Sleep Medicine": 0.35,
    "Physical Medicine - Rehab": 0.35,
    "Hospice - Palliative Care": 0.60,
    "Allergy / Immunology": 0.35,
    "Dermatology": 0.25,
    "Ophthalmology": 0.30,
    "Podiatry": 0.25,
    "Psychiatry / Psychology": 0.55,
    "Radiology": 0.40,
    "Lab Medicine - Pathology": 0.35,
    "Bariatrics": 0.30,
    "Cosmetic / Plastic Surgery": 0.20,
    "Speech - Language": 0.20,
    "Diets and Nutritions": 0.15,
    "Office Notes": 0.20,
    "Chiropractic": 0.15,
    "Dentistry": 0.15,
    "IME-QME-Work Comp etc.": 0.10,
    "Letters": 0.05,
    "Autopsy": 0.00,
}

_URGENT_KEYWORDS = [
    r"\bstat\b", r"\bcode blue\b", r"\bcritical\b", r"\bemergent\b",
    r"\bacute\b", r"\bsevere\b", r"\bunstable\b", r"\bcardiac arrest\b",
    r"\brespiratory failure\b", r"\bhemorrhage\b", r"\bsepsis\b",
]
_KEYWORD_BOOST = 0.05  # per matched keyword, capped below


def keyword_boost(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for pattern in _URGENT_KEYWORDS if re.search(pattern, text_lower))
    return min(hits * _KEYWORD_BOOST, 0.25)


def urgency_score(specialty: str, text: str) -> float:
    base = SPECIALTY_BASE_URGENCY.get(specialty, 0.4)
    score = base + keyword_boost(text)
    return round(min(score, 1.0), 4)
