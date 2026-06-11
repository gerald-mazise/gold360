import streamlit as st
from pathlib import Path


PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"


def get_navigation():
    pages = [
        st.Page(str(PAGES_DIR / "00_overview.py"), title="Overview", icon=":material/home:"),
        st.Page(str(PAGES_DIR / "01_data_pipeline.py"), title="Data Pipeline", icon=":material/database:"),
        st.Page(str(PAGES_DIR / "02_feature_engineering.py"), title="Feature Engineering", icon=":material/engineering:"),
        st.Page(str(PAGES_DIR / "03_weak_supervision.py"), title="Weak Supervision", icon=":material/label:"),
        st.Page(str(PAGES_DIR / "04_anomaly_detection.py"), title="Anomaly Detection", icon=":material/warning:"),
        st.Page(str(PAGES_DIR / "05_fusion_layer.py"), title="Fusion Layer", icon=":material/account_tree:"),
        st.Page(str(PAGES_DIR / "06_model_performance.py"), title="Model Performance", icon=":material/monitoring:"),
        st.Page(str(PAGES_DIR / "07_explainability.py"), title="Explainability", icon=":material/search:"),
        st.Page(str(PAGES_DIR / "08_scenario_analysis.py"), title="Scenario Analysis", icon=":material/science:"),
        st.Page(str(PAGES_DIR / "09_geospatial.py"), title="Geospatial", icon=":material/public:"),
    ]
    return st.navigation(pages, position="sidebar")
