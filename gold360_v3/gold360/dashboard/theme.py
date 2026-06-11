import streamlit as st


class Gold360Theme:
    PRIMARY = "#D4AF37"
    SECONDARY = "#1E293B"
    BG_DARK = "#0F172A"
    BG_CARD = "#1E293B"
    BG_LIGHT = "#334155"
    TEXT_PRIMARY = "#F1F5F9"
    TEXT_SECONDARY = "#94A3B8"
    ACCENT_GREEN = "#38A169"
    ACCENT_RED = "#E53E3E"
    ACCENT_AMBER = "#D69E2E"
    ACCENT_ORANGE = "#DD6B20"
    BORDER = "#2D3748"

    RISK_COLORS = {
        "low": ACCENT_GREEN,
        "moderate": ACCENT_AMBER,
        "elevated": ACCENT_ORANGE,
        "high": ACCENT_RED,
    }

    @staticmethod
    def apply_custom_css():
        st.markdown(f"""
        <style>
        /* ===== RESPONSIVE LAYOUT ===== */
        .block-container {{ max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; padding-top: 1rem !important; }}
        
        /* Sidebar collapsed state - content expands */
        [data-testid="stSidebar"][aria-collapsed="true"] ~ .main .block-container {{ 
            max-width: 100% !important; 
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}

        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{ background-color: {Gold360Theme.BG_CARD} !important; }}
        [data-testid="stSidebarNav"] a {{ color: {Gold360Theme.TEXT_SECONDARY} !important; font-weight: 500 !important; }}
        [data-testid="stSidebarNav"] a:hover {{ color: {Gold360Theme.PRIMARY} !important; }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{ color: {Gold360Theme.PRIMARY} !important; font-weight: 700 !important; }}
        [data-testid="stSidebarNav"] {{ overflow-y: auto !important; max-height: calc(100vh - 180px) !important; }}
        
        /* Keep icons visible when sidebar is collapsed */
        [data-testid="stSidebar"][aria-collapsed="true"] [data-testid="stSidebarNav"] a span[data-testid="stMarkdownContainer"] {{ display: inline !important; }}
        [data-testid="stSidebar"][aria-collapsed="true"] [data-testid="stSidebarNav"] a {{ justify-content: center !important; padding: 0.5rem !important; }}
        [data-testid="stSidebar"][aria-collapsed="true"] [data-testid="stSidebarNav"] a svg {{ margin-right: 0 !important; }}

        /* ===== TYPOGRAPHY ===== */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{ color: {Gold360Theme.PRIMARY} !important; }}
        .stMarkdown p {{ color: {Gold360Theme.TEXT_PRIMARY} !important; }}
        .stMarkdown li {{ color: {Gold360Theme.TEXT_PRIMARY} !important; }}

        /* ===== CARDS ===== */
        [data-testid="stMetric"] {{ background-color: {Gold360Theme.BG_CARD}; border: 1px solid {Gold360Theme.BORDER}; border-radius: 8px; padding: 1rem; }}
        [data-testid="stMetric"] label {{ color: {Gold360Theme.TEXT_SECONDARY} !important; }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {Gold360Theme.PRIMARY} !important; }}
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {{ color: {Gold360Theme.TEXT_SECONDARY} !important; }}

        /* ===== DATAFRAME ===== */
        [data-testid="stDataFrame"] {{ border: 1px solid {Gold360Theme.BORDER}; }}

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab"] {{ color: {Gold360Theme.TEXT_SECONDARY} !important; }}
        .stTabs [aria-selected="true"] {{ color: {Gold360Theme.PRIMARY} !important; }}

        /* ===== EXPANDERS ===== */
        .stExpander summary {{ color: {Gold360Theme.TEXT_PRIMARY} !important; }}

        /* ===== ALERTS ===== */
        .stAlert {{ background-color: {Gold360Theme.BG_CARD} !important; border: 1px solid {Gold360Theme.BORDER} !important; color: {Gold360Theme.TEXT_PRIMARY} !important; }}

        /* ===== SELECTBOX / RADIO ===== */
        .stSelectbox label, .stRadio label, .stMultiSelect label, .stSlider label, .stCheckbox label {{ color: {Gold360Theme.TEXT_SECONDARY} !important; }}

        /* ===== DIVIDER ===== */
        hr {{ border-color: {Gold360Theme.BORDER} !important; opacity: 0.5; }}

        /* ===== PLOTLY ===== */
        .stPlotlyChart {{ border-radius: 8px; }}

        /* ===== FOLIUM ===== */
        .stFolium {{ border-radius: 8px; }}

        /* ===== BUTTONS ===== */
        .stButton > button {{ color: {Gold360Theme.TEXT_PRIMARY} !important; border-color: {Gold360Theme.BORDER} !important; }}
        .stButton > button:hover {{ color: {Gold360Theme.PRIMARY} !important; border-color: {Gold360Theme.PRIMARY} !important; }}

        /* ===== SLIDER WIDTH ===== */
        .threshold-container [data-testid="stSlider"] {{ max-width: 400px !important; }}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def risk_color(score: float) -> str:
        if score >= 0.75:
            return Gold360Theme.ACCENT_RED
        elif score >= 0.50:
            return Gold360Theme.ACCENT_ORANGE
        elif score >= 0.25:
            return Gold360Theme.ACCENT_AMBER
        return Gold360Theme.ACCENT_GREEN

    @staticmethod
    def risk_label(score: float) -> str:
        if score >= 0.75:
            return "high"
        elif score >= 0.50:
            return "elevated"
        elif score >= 0.25:
            return "moderate"
        return "low"

    @staticmethod
    def risk_tier(prob: float, threshold: float) -> tuple:
        """Return (label, color) based on threshold-based risk tiers."""
        if prob < threshold * 0.1:
            return "Clear", Gold360Theme.ACCENT_GREEN
        elif prob < threshold:
            return "Monitoring", Gold360Theme.ACCENT_AMBER
        elif prob < 0.5:
            return "Elevated", Gold360Theme.ACCENT_ORANGE
        else:
            return "Critical", Gold360Theme.ACCENT_RED

    @staticmethod
    def tier_badge_html(label: str, color: str) -> str:
        return f"<span style='background:{color}20;color:{color};padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;'>{label}</span>"
