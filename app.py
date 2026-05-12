"""Streamlit entrypoint — sidebar-driven multi-page portal.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from src import styles
from src.model import load_model
from src.pages import assessment, education, faq, home, recommendations

# Page registry — keep keys in sync with any st.session_state.current_page writes.
PAGES: dict[str, Callable[[Any], None]] = {
    "🏠 Главная": home.render,
    "🩺 Оценка риска": assessment.render,
    "📚 О диабете": education.render,
    "💡 Рекомендации": recommendations.render,
    "❓ FAQ": faq.render,
}
PAGE_NAMES: list[str] = list(PAGES.keys())
DEFAULT_PAGE: str = PAGE_NAMES[0]


@st.cache_resource
def get_model() -> Any:
    """Load the model once per Streamlit session."""
    return load_model()


def _render_sidebar() -> str:
    """Draw the sidebar and return the currently selected page name.

    Uses only native Streamlit widgets — st.title/st.caption/st.radio/st.warning —
    so the sidebar respects the user's Streamlit theme (light or dark) and we
    don't need to fight Streamlit's internal CSS for contrast.
    """
    with st.sidebar:
        st.title("🩺 Diabetes Portal")
        st.caption("Прогноз и профилактика")
        st.divider()

        current = st.session_state.get("current_page", DEFAULT_PAGE)
        selected = st.radio(
            "Навигация",
            PAGE_NAMES,
            index=PAGE_NAMES.index(current) if current in PAGE_NAMES else 0,
        )

        st.divider()
        st.warning("⚠️ Этот инструмент не заменяет консультацию врача.")
    return selected


def main() -> None:
    st.set_page_config(
        page_title="Diabetes Portal — Прогноз риска диабета",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    styles.inject()

    if "current_page" not in st.session_state:
        st.session_state.current_page = DEFAULT_PAGE

    selected = _render_sidebar()
    st.session_state.current_page = selected

    model = get_model()
    PAGES[selected](model)


if __name__ == "__main__":
    main()
