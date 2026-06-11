import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from gold360.dashboard.theme import Gold360Theme
from gold360.dashboard.components.navbar import get_navigation

st.set_page_config(page_title="GOLD360", page_icon="\U0001F48E", layout="wide")
Gold360Theme.apply_custom_css()

nav = get_navigation()

with st.sidebar:
    st.markdown(
        "<div style='padding:0.5rem 0;'>"
        "<h1 style='font-size:1.8rem;color:#D4AF37;margin:0;font-weight:800;letter-spacing:0.05em;'>GOLD360</h1>"
        "<p style='font-size:0.85rem;color:#94A3B8;margin:0;font-weight:500;'>Intelligence Platform</p>"
        "</div>",
        unsafe_allow_html=True,
    )

nav.run()
