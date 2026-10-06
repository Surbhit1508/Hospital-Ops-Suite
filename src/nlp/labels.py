"""Shared label list for the real MTSamples medical transcription corpus.

Sourced from the dataset's own ClassLabel feature metadata (HF datasets-server
`/info` endpoint) -- these are the 40 real medical specialties/document types
used to label each transcription (hence the dataset name `medical_transcription_40`).
"""

SPECIALTY_LABELS = [
    "Pain Management",
    "Chiropractic",
    "Podiatry",
    "Pediatrics - Neonatal",
    "Discharge Summary",
    "Cosmetic / Plastic Surgery",
    "Neurology",
    "Endocrinology",
    "Rheumatology",
    "Orthopedic",
    "Dentistry",
    "Allergy / Immunology",
    "Psychiatry / Psychology",
    "Consult - History and Phy.",
    "Dermatology",
    "Radiology",
    "Speech - Language",
    "Physical Medicine - Rehab",
    "Sleep Medicine",
    "Hospice - Palliative Care",
    "Diets and Nutritions",
    "Urology",
    "ENT - Otolaryngology",
    "Gastroenterology",
    "Letters",
    "Surgery",
    "Bariatrics",
    "Ophthalmology",
    "Neurosurgery",
    "Emergency Room Reports",
    "Nephrology",
    "Lab Medicine - Pathology",
    "Office Notes",
    "Cardiovascular / Pulmonary",
    "SOAP / Chart / Progress Notes",
    "Autopsy",
    "General Medicine",
    "IME-QME-Work Comp etc.",
    "Obstetrics / Gynecology",
    "Hematology - Oncology",
]
