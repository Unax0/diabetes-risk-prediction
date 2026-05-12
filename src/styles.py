"""Component-scoped CSS for the diabetes portal.

Design rules:
- Only style classes that we ourselves inject via `st.markdown(unsafe_allow_html=True)`.
- Never target Streamlit internals (`[data-testid=...]`, `[role=...]`, `.stButton`, etc.) —
  those selectors break across Streamlit versions and conflict with light/dark themes.
- Set both background and text color on every component so it stays legible regardless
  of the user's Streamlit theme.

Call `inject()` once per Streamlit run, after `set_page_config`.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ---------- Hero banner --------------------------------------------- */
.hero {
    background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
    color: #ffffff;
    padding: 2.25rem 2rem;
    border-radius: 0.875rem;
    margin: 0.25rem 0 1.5rem 0;
    box-shadow: 0 10px 30px -12px rgba(15, 118, 110, 0.35);
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    color: #ffffff;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.65rem;
}
.hero-title {
    color: #ffffff;
    font-size: 2.05rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 0.5rem 0;
}
.hero-subtitle {
    color: rgba(255, 255, 255, 0.94);
    font-size: 1.02rem;
    line-height: 1.55;
    margin: 0;
    max-width: 720px;
}

/* ---------- Section lead (sits under st.subheader) ------------------ */
.section-lead {
    color: #6B7280;
    font-size: 0.98rem;
    line-height: 1.55;
    max-width: 780px;
    margin: 0.25rem 0 1rem 0;
}

/* ---------- Info card ----------------------------------------------- */
.info-card {
    background: #ffffff;
    color: #1F2937;
    padding: 1.25rem;
    border-radius: 0.75rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    height: 100%;
    margin-bottom: 0.75rem;
}
.info-card .icon {
    display: block;
    font-size: 1.75rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.info-card .title {
    color: #0F766E;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
}
.info-card .body {
    color: #4B5563;
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0;
}

/* ---------- Stat card ----------------------------------------------- */
.stat-card {
    background: #ffffff;
    color: #1F2937;
    padding: 1.25rem;
    border-radius: 0.75rem;
    border-left: 4px solid #14B8A6;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 0.75rem;
}
.stat-card .stat-number {
    color: #0F766E;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}
.stat-card .stat-label {
    color: #6B7280;
    font-size: 0.875rem;
    margin-top: 0.25rem;
}

/* ---------- Disclaimer ---------------------------------------------- */
.disclaimer {
    background: #FEF3C7;
    color: #78350F;
    border-left: 4px solid #F59E0B;
    padding: 0.9rem 1.1rem;
    border-radius: 0.5rem;
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 1rem 0;
}
.disclaimer strong { color: #78350F; }

/* ---------- Result banner ------------------------------------------- */
.result-banner {
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    border-left: 6px solid;
    margin: 1rem 0;
}
.result-banner.high {
    background: #FEE2E2;
    color: #991B1B;
    border-color: #EF4444;
}
.result-banner.low {
    background: #D1FAE5;
    color: #065F46;
    border-color: #10B981;
}
.result-banner .headline {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.result-banner .score {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0.25rem 0;
}
</style>
"""


def inject() -> None:
    """Inject the portal stylesheet into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
