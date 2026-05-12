"""SHAP-based prediction explanations.

Streamlit-free, so it can be reused from a CLI or tests.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import shap

from . import config


def build_explainer(model: Any) -> shap.TreeExplainer:
    """Construct a TreeExplainer for the loaded XGBoost classifier.

    The explainer captures the model's tree structure once and reuses it
    across many predictions — build it once per session and cache.
    """
    return shap.TreeExplainer(model)


def explain_single(
    explainer: shap.TreeExplainer,
    features: np.ndarray,
) -> shap.Explanation:
    """Compute SHAP values for one feature row.

    Returns a single-sample shap.Explanation with Russian feature names already
    attached, ready for `shap.plots.waterfall` / `shap.plots.bar`.

    Handles the binary-classifier output shape `(1, n_features)` and the
    multi-output shape `(1, n_features, n_classes)` (picks the positive class).
    """
    explanation = explainer(features)

    # Multiclass / multi-output: keep only the positive class.
    if explanation.values.ndim == 3:
        explanation = explanation[..., 1]

    # Override feature names with the Russian display labels.
    russian_names = [
        config.FEATURE_LABELS_RU.get(name, name) for name in config.FEATURE_ORDER
    ]
    explanation.feature_names = russian_names

    return explanation[0]


def top_factors(
    explanation: shap.Explanation,
    n: int = 3,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return the top-n positive and top-n negative SHAP contributions.

    Args:
        explanation: a single-sample Explanation (as returned by `explain_single`).
        n: how many factors to return on each side.

    Returns:
        (positive_factors, negative_factors), each a list of (feature_name, shap_value)
        sorted by absolute contribution descending.
    """
    values = np.asarray(explanation.values, dtype=float)
    names = list(explanation.feature_names or [])

    pairs = list(zip(names, values))
    positives = sorted(
        (p for p in pairs if p[1] > 0), key=lambda p: p[1], reverse=True
    )[:n]
    negatives = sorted(
        (p for p in pairs if p[1] < 0), key=lambda p: p[1]
    )[:n]
    return positives, negatives
