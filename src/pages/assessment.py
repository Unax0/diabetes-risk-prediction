"""Risk assessment page — the predictor, grouped into themed sections."""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import ui
from ..components import disclaimer, hero, result_banner, section_header
from ..features import age_to_bucket, build_feature_vector, calculate_bmi
from ..model import predict_risk


def _demographics_section(inputs: dict[str, float]) -> None:
    with st.container(border=True):
        st.markdown("#### 👤 Демографические данные")
        col1, col2 = st.columns(2)
        with col1:
            age_raw = ui.age_input("Возраст (лет)")
        with col2:
            inputs["Sex"] = ui.sex_input("Пол")
        age_bucket, age_range = age_to_bucket(age_raw)
        inputs["Age"] = age_bucket
        st.caption(f"Возрастная группа: **{age_range}**")


def _body_section(inputs: dict[str, float]) -> None:
    with st.container(border=True):
        st.markdown("#### 📏 Физические показатели")
        col1, col2 = st.columns(2)
        with col1:
            height = ui.height_input("Рост (см)")
        with col2:
            weight = ui.weight_input("Вес (кг)")
        bmi = calculate_bmi(height, weight)
        inputs["BMI"] = bmi
        st.caption(f"Индекс массы тела (BMI): **{bmi:.2f}**")


def _health_section(inputs: dict[str, float]) -> None:
    with st.container(border=True):
        st.markdown("#### 🩺 Состояние здоровья")
        col1, col2 = st.columns(2)
        with col1:
            inputs["HighBP"] = ui.yes_no_input("Повышенное давление?")
            inputs["HighChol"] = ui.yes_no_input("Повышенный холестерин?")
            inputs["Stroke"] = ui.yes_no_input("Был ли инсульт?")
        with col2:
            inputs["HeartDiseaseorAttack"] = ui.yes_no_input(
                "Болезни сердца или инфаркт?"
            )
            inputs["DiffWalk"] = ui.yes_no_input("Трудности при ходьбе?")


def _lifestyle_section(inputs: dict[str, float]) -> None:
    with st.container(border=True):
        st.markdown("#### 🏃 Образ жизни")
        col1, col2 = st.columns(2)
        with col1:
            inputs["Smoker"] = ui.yes_no_input(
                "Курили ли вы более 100 сигарет за жизнь?"
            )
            inputs["HvyAlcoholConsump"] = ui.yes_no_input(
                "Чрезмерное употребление алкоголя?"
            )
            inputs["PhysActivity"] = ui.yes_no_input(
                "Физическая активность за последние 30 дней?"
            )
        with col2:
            inputs["Fruits"] = ui.yes_no_input("Регулярно едите фрукты?")
            inputs["Veggies"] = ui.yes_no_input("Регулярно едите овощи?")


def _wellbeing_section(inputs: dict[str, float]) -> None:
    with st.container(border=True):
        st.markdown("#### 🧠 Самочувствие")
        inputs["GenHlth"] = ui.gen_health_input(
            "Общее состояние здоровья (1 = отличное, 5 = плохое)"
        )
        col1, col2 = st.columns(2)
        with col1:
            inputs["MentHlth"] = ui.days_input(
                "Дней плохого психического состояния (за 30 дней)"
            )
        with col2:
            inputs["PhysHlth"] = ui.days_input(
                "Дней плохого физического состояния (за 30 дней)"
            )


def _result_section(model: Any, inputs: dict[str, float]) -> None:
    section_header("Результат")
    features = build_feature_vector(inputs)
    probability, is_high_risk = predict_risk(model, features)

    result_banner(probability, is_high_risk)
    st.progress(min(probability, 1.0))

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("Смотреть рекомендации", key="assess_to_rec"):
            st.session_state.current_page = "💡 Рекомендации"
            st.rerun()
    with col2:
        if st.button("Узнать больше о диабете", key="assess_to_edu"):
            st.session_state.current_page = "📚 О диабете"
            st.rerun()


def render(model: Any) -> None:
    hero(
        title="Оценка риска диабета",
        subtitle=(
            "Ответьте на вопросы ниже. Модель машинного обучения оценит вероятность "
            "наличия или развития диабета на основе ваших ответов."
        ),
        badge="🩺 ОЦЕНКА",
    )
    disclaimer(
        "<strong>Дисклеймер.</strong> Результат носит ориентировочный характер и не "
        "заменяет консультацию врача. Не используйте этот инструмент для самостоятельной "
        "диагностики."
    )

    inputs: dict[str, float] = {}
    _demographics_section(inputs)
    _body_section(inputs)
    _health_section(inputs)
    _lifestyle_section(inputs)
    _wellbeing_section(inputs)

    if st.button(
        "Узнать результат",
        use_container_width=True,
        type="primary",
        key="assess_submit",
    ):
        _result_section(model, inputs)
