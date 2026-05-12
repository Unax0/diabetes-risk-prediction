"""Configuration constants for the diabetes risk prediction app.

All thresholds, input bounds, age-bucket boundaries, the model's expected
feature order, and the Russian UI labels live here so the rest of the
codebase has no hard-coded magic numbers or strings.
"""
from __future__ import annotations

from typing import Final

# --- Model artifact ---------------------------------------------------------

MODEL_PATH: Final[str] = "xgb_model_reduced.pkl"

# Decision threshold tuned to favor recall (kept from the original app.py).
RISK_THRESHOLD: Final[float] = 0.37

# --- Input bounds (match the original Streamlit widgets) --------------------

AGE_MIN: Final[int] = 0
AGE_MAX: Final[int] = 150
AGE_DEFAULT: Final[int] = 18

HEIGHT_MIN_CM: Final[int] = 50
HEIGHT_MAX_CM: Final[int] = 250
HEIGHT_DEFAULT_CM: Final[int] = 170

WEIGHT_MIN_KG: Final[int] = 10
WEIGHT_MAX_KG: Final[int] = 300
WEIGHT_DEFAULT_KG: Final[int] = 70

GEN_HLTH_MIN: Final[int] = 1
GEN_HLTH_MAX: Final[int] = 5
GEN_HLTH_DEFAULT: Final[int] = 3

DAYS_MIN: Final[int] = 0
DAYS_MAX: Final[int] = 30
DAYS_DEFAULT: Final[int] = 0

# --- Age buckets ------------------------------------------------------------
# (min_age, max_age_inclusive, bucket_id, russian_label)
# Ages outside any listed range fall back to AGE_FALLBACK_* — this matches
# the original app's `else` branch, which mapped both <18 and >79 to "80+".

AGE_BUCKETS: Final[tuple[tuple[int, int, int, str], ...]] = (
    (18, 24, 1, "18-24 лет"),
    (25, 29, 2, "25-29 лет"),
    (30, 34, 3, "30-34 лет"),
    (35, 39, 4, "35-39 лет"),
    (40, 44, 5, "40-44 лет"),
    (45, 49, 6, "45-49 лет"),
    (50, 54, 7, "50-54 лет"),
    (55, 59, 8, "55-59 лет"),
    (60, 64, 9, "60-64 лет"),
    (65, 69, 10, "65-69 лет"),
    (70, 74, 11, "70-74 лет"),
    (75, 79, 12, "75-79 лет"),
)
AGE_FALLBACK_BUCKET: Final[int] = 13
AGE_FALLBACK_LABEL: Final[str] = "80+ лет"

# --- Model feature order ----------------------------------------------------
# MUST match the column order used when training xgb_model_reduced.pkl.

FEATURE_ORDER: Final[tuple[str, ...]] = (
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
)

# --- UI strings (Russian) ---------------------------------------------------

YES: Final[str] = "Да"
NO: Final[str] = "Нет"
MALE: Final[str] = "Мужчина"
FEMALE: Final[str] = "Женщина"
