"""Pure feature-engineering helpers.

No Streamlit dependency — every function here is deterministic and unit-testable
in isolation.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from . import config


def age_to_bucket(age: int) -> tuple[int, str]:
    """Map an absolute age (years) to its (bucket_id, russian_label) pair.

    Ages outside the 18-79 buckets fall back to bucket 13 / "80+ лет",
    preserving the original app.py behavior (which also caught <18).
    """
    for low, high, bucket_id, label in config.AGE_BUCKETS:
        if low <= age <= high:
            return bucket_id, label
    return config.AGE_FALLBACK_BUCKET, config.AGE_FALLBACK_LABEL


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Body Mass Index from height (cm) and weight (kg)."""
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def yes_no_to_int(value: str) -> int:
    """Map Russian "Да"/"Нет" to 1/0."""
    return 1 if value == config.YES else 0


def sex_to_int(value: str) -> int:
    """Map Russian "Мужчина"/"Женщина" to 1/0."""
    return 1 if value == config.MALE else 0


def build_feature_vector(inputs: Mapping[str, float]) -> np.ndarray:
    """Assemble a 2-D feature array in the order the model expects.

    Args:
        inputs: mapping from feature name (see config.FEATURE_ORDER) to value.

    Returns:
        A (1, n_features) numpy array suitable for model.predict_proba.

    Raises:
        KeyError: if any feature required by FEATURE_ORDER is missing.
    """
    row = [inputs[name] for name in config.FEATURE_ORDER]
    return np.array([row])
