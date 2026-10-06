"""
Department taxonomy that bridges the diabetes-encounter dataset's
`medical_specialty` categories and the MTSamples NLP specialty labels, so
the optimization stage can pull REAL numbers from both upstream modules
for the same set of hospital departments.

Only departments that exist (in spirit) in both source datasets are kept --
a real bed-allocation model wouldn't include "Letters" or "Autopsy" as
inpatient departments anyway, so trimming down from 40 NLP classes to this
list is a deliberate, documented modeling choice, not an oversight.
"""

# name -> (diabetes-dataset one-hot column(s) to average time_in_hospital over,
#          NLP specialty label used for case-mix share + urgency lookup,
#          ICU-bed fraction assumption for this department)
DEPARTMENTS = {
    "Cardiology": {
        "diabetes_cols": ["medical_specialty:Cardiology"],
        "nlp_label": "Cardiovascular / Pulmonary",
        "icu_fraction": 0.25,
    },
    "Orthopedics": {
        "diabetes_cols": ["medical_specialty:Orthopedics"],
        "nlp_label": "Orthopedic",
        "icu_fraction": 0.05,
    },
    "Nephrology": {
        "diabetes_cols": ["medical_specialty:Nephrology"],
        "nlp_label": "Nephrology",
        "icu_fraction": 0.15,
    },
    "Gastroenterology": {
        "diabetes_cols": ["medical_specialty:Gastroenterology"],
        "nlp_label": "Gastroenterology",
        "icu_fraction": 0.10,
    },
    "Psychiatry": {
        "diabetes_cols": ["medical_specialty:Psychiatry"],
        "nlp_label": "Psychiatry / Psychology",
        "icu_fraction": 0.02,
    },
    "Surgery": {
        "diabetes_cols": ["medical_specialty:Surgery-General", "medical_specialty:Surgery-Cardiovascular/Thoracic"],
        "nlp_label": "Surgery",
        "icu_fraction": 0.20,
    },
    "Obstetrics/Gynecology": {
        "diabetes_cols": ["medical_specialty:ObstetricsandGynecology"],
        "nlp_label": "Obstetrics / Gynecology",
        "icu_fraction": 0.08,
    },
    "Urology": {
        "diabetes_cols": ["medical_specialty:Urology"],
        "nlp_label": "Urology",
        "icu_fraction": 0.05,
    },
    "General/Internal Medicine": {
        "diabetes_cols": ["medical_specialty:InternalMedicine", "medical_specialty:Family/GeneralPractice"],
        "nlp_label": "General Medicine",
        "icu_fraction": 0.05,
    },
    "Emergency": {
        "diabetes_cols": ["medical_specialty:Emergency/Trauma"],
        "nlp_label": "Emergency Room Reports",
        "icu_fraction": 0.30,
    },
}
