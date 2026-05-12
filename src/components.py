"""Reusable display components.

Where Streamlit has a native widget that does the job (subheader, divider,
warning, button), we use it. Only purely-visual components (hero, info_card,
stat_card, disclaimer, result_banner) are rendered as injected HTML, and
each one sets explicit colors so it stays legible in any theme.
"""
from __future__ import annotations

from typing import Iterable

import streamlit as st


def hero(
    title: str,
    subtitle: str,
    badge: str | None = None,
) -> None:
    """Top-of-page gradient banner.

    Uses <div> elements instead of <h1>/<p> so Streamlit's own heading CSS
    doesn't fight the white-on-teal coloring.
    """
    badge_html = f"<div class='hero-badge'>{badge}</div>" if badge else ""
    st.markdown(
        f"""
        <div class="hero">
            {badge_html}
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, lead: str | None = None) -> None:
    """Native subheader with an optional gray lead paragraph beneath it."""
    st.subheader(title)
    if lead:
        st.markdown(f"<p class='section-lead'>{lead}</p>", unsafe_allow_html=True)


def info_card(icon: str, title: str, body: str) -> None:
    """White card with icon, title, and body text."""
    st.markdown(
        f"""
        <div class="info-card">
            <span class="icon">{icon}</span>
            <div class="title">{title}</div>
            <div class="body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(number: str, label: str) -> None:
    """Big number + caption card (left-bordered)."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-number">{number}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer(text: str) -> None:
    """Yellow warning callout — content is rendered as HTML so <strong> works."""
    st.markdown(f"<div class='disclaimer'>{text}</div>", unsafe_allow_html=True)


def feature_grid(items: Iterable[tuple[str, str, str]], columns: int = 3) -> None:
    """Render a responsive grid of info_cards.

    Args:
        items: iterable of (icon, title, body) triples.
        columns: number of columns (Streamlit stacks on narrow screens).
    """
    items_list = list(items)
    if not items_list:
        return
    cols = st.columns(columns)
    for idx, (icon, title, body) in enumerate(items_list):
        with cols[idx % columns]:
            info_card(icon, title, body)


def stat_grid(items: Iterable[tuple[str, str]], columns: int = 3) -> None:
    """Render a responsive grid of stat_cards."""
    items_list = list(items)
    if not items_list:
        return
    cols = st.columns(columns)
    for idx, (number, label) in enumerate(items_list):
        with cols[idx % columns]:
            stat_card(number, label)


def result_banner(probability: float, is_high_risk: bool) -> None:
    """Big colored banner displayed after running the risk assessment."""
    if is_high_risk:
        cls, headline, advice = (
            "high",
            "⚠️ Повышенный риск",
            "Рекомендуется обратиться к врачу для углублённого обследования.",
        )
    else:
        cls, headline, advice = (
            "low",
            "✅ Низкий риск",
            "Продолжайте поддерживать здоровый образ жизни.",
        )
    st.markdown(
        f"""
        <div class="result-banner {cls}">
            <div class="headline">{headline}</div>
            <div class="score">{probability * 100:.1f}%</div>
            <div>{advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
