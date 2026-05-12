"""Homepage — hero, statistics, and feature overview."""
from __future__ import annotations

from typing import Any

import streamlit as st

from ..components import (
    disclaimer,
    feature_grid,
    hero,
    section_header,
    stat_grid,
)


def render(model: Any = None) -> None:
    hero(
        title="Прогноз риска диабета",
        subtitle=(
            "Бесплатная онлайн-оценка вероятности развития сахарного диабета "
            "по 16 признакам здоровья и образа жизни. Узнайте свой риск за 2 минуты."
        ),
        badge="🩺 ИНСТРУМЕНТ ПРОФИЛАКТИКИ",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(
            "Пройти тест →",
            use_container_width=True,
            type="primary",
            key="home_cta_top",
        ):
            st.session_state.current_page = "🩺 Оценка риска"
            st.rerun()

    section_header(
        "Диабет в цифрах",
        "Сахарный диабет — одно из самых распространённых хронических заболеваний в мире.",
    )
    stat_grid(
        [
            ("537 млн", "взрослых с диабетом в мире"),
            ("1 из 11", "взрослых живёт с диабетом"),
            ("≈ 50%", "случаев не диагностированы"),
        ],
        columns=3,
    )

    section_header(
        "Что вы найдёте в портале",
        "Четыре раздела, которые помогут оценить риск и принять меры профилактики.",
    )
    feature_grid(
        [
            (
                "🩺",
                "Оценка риска",
                "Ответьте на 16 коротких вопросов — модель машинного обучения "
                "оценит вашу вероятность развития диабета.",
            ),
            (
                "📚",
                "О диабете",
                "Образовательные материалы: типы, симптомы, факторы риска, "
                "осложнения и группы повышенного риска.",
            ),
            (
                "💡",
                "Рекомендации",
                "Практические шаги для снижения риска: питание, физическая "
                "активность, привычки и регулярный контроль здоровья.",
            ),
            (
                "❓",
                "FAQ",
                "Ответы на частые вопросы о диабете, профилактике и о том, "
                "как работает этот инструмент.",
            ),
        ],
        columns=2,
    )

    section_header("Готовы узнать свой риск?")
    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        if st.button(
            "Пройти оценку",
            use_container_width=True,
            type="primary",
            key="home_cta_assess",
        ):
            st.session_state.current_page = "🩺 Оценка риска"
            st.rerun()
    with col_b:
        if st.button("Смотреть рекомендации", use_container_width=True, key="home_cta_rec"):
            st.session_state.current_page = "💡 Рекомендации"
            st.rerun()

    disclaimer(
        "<strong>Важно.</strong> Этот портал предоставляет ориентировочную оценку и "
        "не является медицинским заключением. Для диагностики и лечения обращайтесь "
        "к квалифицированному врачу."
    )
