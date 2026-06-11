import streamlit as st


def metric_card(label: str, value: str, delta: str = None,
                help_text: str = None, color: str = None):
    color_style = f"color: {color};" if color else ""
    delta_html = f'<p style="font-size:0.8rem;color:{"#38A169" if delta and delta.startswith("+") else "#E53E3E"}">{delta}</p>' if delta else ""
    help_attr = f'title="{help_text}"' if help_text else ""

    st.markdown(
        f'<div class="metric-card" {help_attr}>'
        f'<p style="font-size:0.85rem;color:#94A3B8;margin:0">{label}</p>'
        f'<p style="font-size:1.8rem;font-weight:700;margin:0;{color_style}">{value}</p>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
