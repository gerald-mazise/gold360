import streamlit as st

from gold360.dashboard.theme import Gold360Theme


def risk_indicator(score: float, label: str = None):
    color = Gold360Theme.risk_color(score)
    label = label or Gold360Theme.risk_label(score).upper()
    st.markdown(
        f'<span class="risk-badge" style="background-color:{color}20;color:{color};'
        f'border:1px solid {color};">{label}</span>',
        unsafe_allow_html=True,
    )


def risk_bar(score: float, height: int = 20, width: int = 200):
    color = Gold360Theme.risk_color(score)
    pct = min(score * 100, 100)
    bar_html = f"""
    <div style="width:{width}px;height:{height}px;background-color:{Gold360Theme.BG_LIGHT};
                border-radius:4px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background-color:{color};
                    border-radius:4px;transition:width 0.3s;"></div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
