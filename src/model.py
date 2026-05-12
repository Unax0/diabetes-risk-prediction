"""Model loading and inference.

Kept Streamlit-free so it can be reused from a CLI, a FastAPI endpoint,
or tests without dragging in the UI layer.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from . import config


def load_model(path: str = config.MODEL_PATH) -> Any:
    """Load the serialized XGBoost classifier from disk."""
    return joblib.load(path)


def predict_probability(model: Any, features: np.ndarray) -> float:
    """Return P(diabetes = 1) for a single feature row."""
    return float(model.predict_proba(features)[0][1])


def predict_risk(
    model: Any,
    features: np.ndarray,
    threshold: float = config.RISK_THRESHOLD,
) -> tuple[float, bool]:
    """Run inference and apply the decision threshold.

    Returns:
        (probability, is_high_risk) where is_high_risk = probability >= threshold.
    """
    probability = predict_probability(model, features)
    return probability, probability >= threshold
