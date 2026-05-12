"""Risk assessment page — the predictor, grouped into themed sections."""
from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from .. import ui
from ..components import disclaimer, hero, result_banner, section_header
from ..features import age_to_bucket, build_feature_vector, calculate_bmi
from ..model import predict_risk

# shap, matplotlib, and src.explain are imported lazily inside
# `_explanation_section` so the app still starts when those (optional, heavy)
# packages are not yet installed.


@st.cache_resource
def _get_explainer(_model: Any) -> Any:
    """Build a SHAP TreeExplainer once per session.

    The leading underscore on `_model` tells Streamlit's cache machinery to
    skip hashing the (large, unhashable) model object.
    """
    from .. import explain  # lazy: avoid importing shap at module load
    return explain.build_explainer(_model)


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


def _explanation_section(model: Any, features: np.ndarray) -> None:
    """Render SHAP-based explanation of the prediction."""
    section_header(
        "🔍 Объяснение результата",
        "Какие признаки больше всего повлияли на оценку. Положительные значения "
        "увеличивают риск, отрицательные — снижают его.",
    )

    # Lazy imports — keep the rest of the app working if these aren't installed.
    try:
        import matplotlib.pyplot as plt
        import shap
        from .. import explain
    except ImportError as e:
        st.info(
            "Для отображения SHAP-объяснений установите дополнительные пакеты:\n\n"
            "```bash\npip install -r requirements.txt\n```\n\n"
            f"Отсутствует модуль: `{e.name}`"
        )
        return

    try:
        explainer = _get_explainer(model)
        explanation = explain.explain_single(explainer, features)
    except Exception as e:  # noqa: BLE001 — surface to user, don't crash the page
        st.warning(f"Не удалось построить SHAP-объяснение: {e}")
        return

    # --- Text summary: top factors on each side ---
    positives, negatives = explain.top_factors(explanation, n=3)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔺 Увеличивают риск:**")
        if positives:
            for name, val in positives:
                st.markdown(f"- {name} &nbsp;`+{val:.3f}`")
        else:
            st.caption("Нет факторов, повышающих риск.")
    with col2:
        st.markdown("**🔻 Снижают риск:**")
        if negatives:
            for name, val in negatives:
                st.markdown(f"- {name} &nbsp;`{val:.3f}`")
        else:
            st.caption("Нет факторов, снижающих риск.")

    # --- Waterfall plot: full breakdown from base value to prediction ---
    with st.expander("Подробная диаграмма вкладов (waterfall)", expanded=True):
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation, show=False, max_display=12)
        st.pyplot(plt.gcf(), use_container_width=True, clear_figure=True)
        plt.close("all")

    # --- Bar plot: absolute magnitude of each feature's contribution ---
    with st.expander("Величина вклада признаков (bar)"):
        fig = plt.figure(figsize=(9, 5))
        shap.plots.bar(explanation, show=False, max_display=12)
        st.pyplot(plt.gcf(), use_container_width=True, clear_figure=True)
        plt.close("all")

    st.caption(
        "SHAP-значения показывают вклад каждого признака в отклонение прогноза "
        "от среднего по обучающей выборке. Это объяснение модели, а не медицинская оценка."
    )


def _result_section(model: Any, inputs: dict[str, float]) -> None:
    section_header("Результат")
    features = build_feature_vector(inputs)
    probability, is_high_risk = predict_risk(model, features)

    result_banner(probability, is_high_risk)
    st.progress(min(probability, 1.0))

    _explanation_section(model, features)

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
