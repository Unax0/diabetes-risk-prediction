"""Reusable Streamlit input widgets.

Each helper renders one widget and returns the value already converted to the
type the model expects (e.g. yes/no → 0/1), so app.py never has to repeat the
"Да"/"Нет" → int dance.
"""
from __future__ import annotations

import streamlit as st

from . import config
from .features import sex_to_int, yes_no_to_int


def yes_no_input(label: str) -> int:
    """Render a "Нет"/"Да" selectbox and return 0/1."""
    raw = st.selectbox(label, [config.NO, config.YES])
    return yes_no_to_int(raw)


def sex_input(label: str) -> int:
    """Render a "Женщина"/"Мужчина" selectbox and return 0/1."""
    raw = st.selectbox(label, [config.FEMALE, config.MALE])
    return sex_to_int(raw)


def age_input(label: str) -> int:
    """Render the age number_input."""
    return int(
        st.number_input(
            label,
            min_value=config.AGE_MIN,
            max_value=config.AGE_MAX,
            value=config.AGE_DEFAULT,
        )
    )


def height_input(label: str) -> int:
    """Render the height (cm) number_input."""
    return int(
        st.number_input(
            label,
            min_value=config.HEIGHT_MIN_CM,
            max_value=config.HEIGHT_MAX_CM,
            value=config.HEIGHT_DEFAULT_CM,
        )
    )


def weight_input(label: str) -> int:
    """Render the weight (kg) number_input."""
    return int(
        st.number_input(
            label,
            min_value=config.WEIGHT_MIN_KG,
            max_value=config.WEIGHT_MAX_KG,
            value=config.WEIGHT_DEFAULT_KG,
        )
    )


def gen_health_input(label: str) -> int:
    """Render the 1..5 general-health slider."""
    return st.slider(
        label,
        config.GEN_HLTH_MIN,
        config.GEN_HLTH_MAX,
        config.GEN_HLTH_DEFAULT,
    )


def days_input(label: str) -> int:
    """Render a 0..30 day-count slider (used for MentHlth / PhysHlth)."""
    return st.slider(
        label,
        config.DAYS_MIN,
        config.DAYS_MAX,
        config.DAYS_DEFAULT,
    )
