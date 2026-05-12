"""Streamlit entrypoint for the diabetes risk prediction app.

Run with:
    streamlit run app.py

Heavy lifting (feature engineering, model I/O) lives in `src/`.
This file only wires widgets to those helpers and renders the result.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from src import ui
from src.features import age_to_bucket, build_feature_vector, calculate_bmi
from src.model import load_model, predict_risk


@st.cache_resource
def get_model() -> Any:
    """Load the model once per Streamlit session."""
    return load_model()


def main() -> None:
    model = get_model()

    st.title("Опросник: Прогноз риска диабета")
    st.subheader("Ответьте на вопросы")

    age_raw = ui.age_input("Введите ваш возраст")
    age_bucket, age_range = age_to_bucket(age_raw)
    st.write(f"Ваш возраст попадает в диапазон: {age_range}")

    height = ui.height_input("Введите ваш рост (в см)")
    weight = ui.weight_input("Введите ваш вес (в кг)")
    bmi = calculate_bmi(height, weight)
    st.write(f"Ваш индекс массы тела (BMI): {bmi:.2f}")

    # Order of dict insertion preserves widget render order — keep this
    # identical to the original app.py question sequence.
    inputs: dict[str, float] = {
        "HighBP": ui.yes_no_input("Есть ли повышенное давление?"),
        "HighChol": ui.yes_no_input("Есть ли повышенный холестерин?"),
        "BMI": bmi,
        "Smoker": ui.yes_no_input("Курили ли вы более 100 сигарет за жизнь?"),
        "Stroke": ui.yes_no_input("Был ли инсульт?"),
        "HeartDiseaseorAttack": ui.yes_no_input("Были ли болезни сердца / инфаркт?"),
        "PhysActivity": ui.yes_no_input("Была ли физическая активность за последние 30 дней?"),
        "Fruits": ui.yes_no_input("Употребляете ли фрукты регулярно?"),
        "Veggies": ui.yes_no_input("Употребляете ли овощи регулярно?"),
        "HvyAlcoholConsump": ui.yes_no_input("Чрезмерное употребление алкоголя?"),
        "GenHlth": ui.gen_health_input("Общее состояние здоровья (1 = отличное, 5 = плохое)"),
        "MentHlth": ui.days_input("Дней плохого психического состояния за 30 дней"),
        "PhysHlth": ui.days_input("Дней плохого физического состояния за 30 дней"),
        "DiffWalk": ui.yes_no_input("Есть ли трудности при ходьбе?"),
        "Sex": ui.sex_input("Пол (0 = женщина, 1 = мужчина)"),
        "Age": age_bucket,
    }

    if st.button("Узнать результат"):
        features = build_feature_vector(inputs)
        probability, is_high_risk = predict_risk(model, features)

        st.write(f"Вероятность риска диабета: {probability * 100:.2f}%")
        if is_high_risk:
            st.error("⚠️ Повышенный риск диабета. Рекомендуется обратиться к врачу.")
        else:
            st.success("✅ Низкий риск диабета.")


if __name__ == "__main__":
    main()
