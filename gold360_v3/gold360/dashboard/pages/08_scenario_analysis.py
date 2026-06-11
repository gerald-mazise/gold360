import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yaml

from gold360.dashboard.theme import Gold360Theme
from gold360.dashboard.report_loader import load_report

Gold360Theme.apply_custom_css()

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Scenario Intelligence Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>Micro-simulation of policy interventions — directional analysis only</p>", unsafe_allow_html=True)

st.markdown("""
<div style='background:#D69E2E15;border:1px solid #D69E2E40;border-radius:8px;padding:1rem;margin-bottom:1rem;'>
    <span style='color:#D69E2E;font-weight:600;'>⚠ CAVEAT:</span>
    <span style='color:#94A3B8;'> These are directional signals, not deterministic predictions. Elasticities are estimated from historical data and may not hold under regime change. Analyst interpretation required.</span>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def _load_raw_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    path = os.path.join(data_dir, "synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "default.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_data
def _load_elasticities(config):
    from gold360.policy.elasticities import PolicyElasticities
    e = PolicyElasticities()
    df = _load_raw_data()
    if df is not None and "policy_shock_flag" in df.columns and "official_delivery_kg" in df.columns:
        df["policy_shock_flag"] = pd.to_numeric(df["policy_shock_flag"], errors="coerce").fillna(0)
        df["official_delivery_kg"] = pd.to_numeric(df["official_delivery_kg"], errors="coerce").fillna(0)
        e.estimate(df)
    else:
        e.elasticities["delivery_response"] = 0.15
    return e


def _run_simulation(params, elasticities, df):
    from gold360.policy.engine import ScenarioEngine
    engine = ScenarioEngine(elasticities)
    return engine.simulate("custom", params, df)


def _scenario_result_to_dict(result):
    return {
        "scenario": result.scenario_name,
        "delivery_change_pct": round(result.predicted_delivery_change_pct, 2),
        "residual_direction": result.predicted_residual_direction,
        "anomaly_shift": round(result.predicted_anomaly_shift, 2),
        "confidence": result.confidence,
    }


def _confidence_color(conf):
    return {"high": "#38A169", "medium": "#D4AF37", "low": "#E53E3E"}.get(conf, "#94A3B8")


def _residual_color(direction):
    return {"widening": "#E53E3E", "narrowing": "#38A169", "stable": "#94A3B8"}.get(direction, "#94A3B8")


# ──────────────────────────────────────────────────────────────────────
# SECTION 1: PRESET SCENARIOS (loaded from evaluation data + engine)
# ──────────────────────────────────────────────────────────────────────
st.markdown("### Preset Policy Scenarios")

config = _load_config()
elasticities = _load_elasticities(config)
raw_df = _load_raw_data()

preset_params = {
    "FX Reform": {"retention": -0.10, "fx_retention": -0.10},
    "Tax Increase": {"tax_rate": 0.05},
    "Compliance Tightening": {"compliance": 0.15},
    "FX Retention Increase": {"retention": 0.10, "fx_retention": 0.10},
    "Combined Tightening": {"retention": -0.05, "tax_rate": 0.03, "compliance": 0.10},
    "Baseline": {},
}

preset_results = {}
if raw_df is not None:
    for name, params in preset_params.items():
        result = _run_simulation(params, elasticities, raw_df)
        preset_results[name] = _scenario_result_to_dict(result)
else:
    for name, params in preset_params.items():
        preset_results[name] = {"scenario": name, "delivery_change_pct": 0, "residual_direction": "stable", "anomaly_shift": 0, "confidence": "low"}

# Impact chart
scenarios_list = list(preset_results.keys())
delivery_impact = [preset_results[s]["delivery_change_pct"] for s in scenarios_list]
residual_impact = [preset_results[s]["anomaly_shift"] for s in scenarios_list]
confidence_vals = [preset_results[s]["confidence"] for s in scenarios_list]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=scenarios_list, y=delivery_impact, name="Delivery Change (%)",
    marker_color="#D4AF37",
    text=[f"{v:+.1f}%" for v in delivery_impact],
    textposition="outside", textfont=dict(color="#D4AF37", size=11),
))
fig.add_trace(go.Bar(
    x=scenarios_list, y=residual_impact, name="Anomaly Shift (%)",
    marker_color="#38A169",
    text=[f"{v:+.1f}%" for v in residual_impact],
    textposition="outside", textfont=dict(color="#38A169", size=11),
))
fig.update_layout(
    barmode="group",
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font_color="#94A3B8",
    xaxis=dict(gridcolor="#2D3748"),
    yaxis=dict(gridcolor="#2D3748", title="% Change"),
    legend=dict(font=dict(color="#94A3B8"), orientation="h"),
    margin=dict(t=10, b=10, l=10, r=10), height=380,
)
st.plotly_chart(fig, use_container_width=True)

# Scenario detail cards
col1, col2, col3 = st.columns(3)
preset_cards = [
    ("FX Reform", "Parallel market premium halved", "FX Reform"),
    ("Tax Increase", "5% increase in mining tax", "Tax Increase"),
    ("Compliance Tightening", "15% stricter enforcement", "Compliance Tightening"),
]
for col, (label, desc, key) in zip([col1, col2, col3], preset_cards):
    r = preset_results[key]
    color = _confidence_color(r["confidence"])
    with col:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid {color}30;border-radius:8px;padding:1.2rem;'>
            <div style='font-size:1.1rem;font-weight:700;color:{color};'>{label}</div>
            <div style='font-size:0.8rem;color:#94A3B8;margin:0.3rem 0;'>{desc}</div>
            <hr style='border-color:#2D3748;margin:0.5rem 0;'>
            <div style='display:flex;justify-content:space-between;'>
                <div><span style='color:#94A3B8;font-size:0.75rem;'>Delivery:</span> <b style='color:#38A169;'>{r['delivery_change_pct']:+.1f}%</b></div>
                <div><span style='color:#94A3B8;font-size:0.75rem;'>Residual:</span> <b style='color:{_residual_color(r['residual_direction'])};'>{r['residual_direction']}</b></div>
            </div>
            <div style='margin-top:0.5rem;'>
                <span style='color:#94A3B8;font-size:0.75rem;'>Confidence: </span>
                <b style='color:{color};'>{r['confidence']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SECTION 2: WHAT-IF SCENARIO BUILDER
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### What-If Scenario Builder")
st.markdown("<p style='color:#94A3B8;margin:0 0 1rem 0;'>Adjust policy parameters and see real-time directional impact estimates.</p>", unsafe_allow_html=True)

# Initialize defaults BEFORE any widgets render
for _k, _v in [("wi_retention", 0.0), ("wi_fx", 0.0), ("wi_tax", 0.0), ("wi_compliance", 0.0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Handle pending preset BEFORE sliders render (set by button callback below)
if "_pending_preset" in st.session_state and st.session_state._pending_preset is not None:
    _pp = st.session_state._pending_preset
    st.session_state.wi_retention = _pp.get("retention", 0.0)
    st.session_state.wi_fx = _pp.get("fx_retention", 0.0)
    st.session_state.wi_tax = _pp.get("tax_rate", 0.0)
    st.session_state.wi_compliance = _pp.get("compliance", 0.0)
    st.session_state._pending_preset = None

col_sliders, col_results = st.columns([1, 1])

with col_sliders:
    st.markdown("**Policy Parameters**")
    retention = st.slider(
        "Retention rate change",
        min_value=-0.30, max_value=0.30, step=0.01,
        format="%.2f", key="wi_retention",
        help="Positive = higher retention (more gold kept in-country)"
    )
    fx_retention = st.slider(
        "FX retention change",
        min_value=-0.30, max_value=0.30, step=0.01,
        format="%.2f", key="wi_fx",
        help="Positive = higher FX retention for miners"
    )
    tax_rate = st.slider(
        "Tax rate change",
        min_value=-0.20, max_value=0.20, step=0.01,
        format="%.2f", key="wi_tax",
        help="Positive = higher mining tax"
    )
    compliance = st.slider(
        "Compliance enforcement change",
        min_value=-0.30, max_value=0.30, step=0.01,
        format="%.2f", key="wi_compliance",
        help="Positive = stricter enforcement"
    )

    custom_params = {}
    if retention != 0.0:
        custom_params["retention"] = retention
    if fx_retention != 0.0:
        custom_params["fx_retention"] = fx_retention
    if tax_rate != 0.0:
        custom_params["tax_rate"] = tax_rate
    if compliance != 0.0:
        custom_params["compliance"] = compliance

    # Preset quick-apply buttons
    st.markdown("**Quick Presets**")
    preset_cols = st.columns(3)
    preset_apply = {
        "FX Reform": {"retention": -0.10, "fx_retention": -0.10},
        "Tax Hike": {"tax_rate": 0.05},
        "Compliance": {"compliance": 0.15},
        "Retention+": {"retention": 0.10, "fx_retention": 0.10},
        "Tightening": {"retention": -0.05, "tax_rate": 0.03, "compliance": 0.10},
        "Reset": {},
    }
    for i, (pname, pparams) in enumerate(preset_apply.items()):
        with preset_cols[i % 3]:
            if st.button(pname, key=f"wi_preset_{pname}", use_container_width=True):
                st.session_state._pending_preset = pparams
                st.rerun()

with col_results:
    st.markdown("**Simulation Results**")
    if custom_params and raw_df is not None:
        result = _run_simulation(custom_params, elasticities, raw_df)
        rd = _scenario_result_to_dict(result)
        conf_color = _confidence_color(rd["confidence"])

        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid {conf_color}40;border-radius:8px;padding:1.2rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div style='font-size:1rem;font-weight:700;color:{conf_color};'>Custom Scenario</div>
                <span style='background:{conf_color}20;color:{conf_color};padding:0.2rem 0.6rem;border-radius:12px;font-size:0.75rem;font-weight:600;'>{rd['confidence'].upper()} CONFIDENCE</span>
            </div>
            <hr style='border-color:#2D3748;margin:0.5rem 0;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:0.5rem;'>
                <span style='color:#94A3B8;'>Delivery Change:</span>
                <b style='color:#38A169;font-size:1.1rem;'>{rd['delivery_change_pct']:+.1f}%</b>
            </div>
            <div style='display:flex;justify-content:space-between;margin-bottom:0.5rem;'>
                <span style='color:#94A3B8;'>Residual Direction:</span>
                <b style='color:{_residual_color(rd['residual_direction'])};'>{rd['residual_direction']}</b>
            </div>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#94A3B8;'>Anomaly Shift:</span>
                <b style='color:#D4AF37;'>{rd['anomaly_shift']:+.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Parameter summary
        param_str = ", ".join(f"{k}={v:+.0%}" for k, v in custom_params.items())
        st.caption(f"Parameters: {param_str}")
    elif raw_df is None:
        st.warning("Raw data not found. Cannot run simulation.")
    else:
        st.info("Adjust sliders or click a preset to see results.")

# ──────────────────────────────────────────────────────────────────────
# SECTION 3: COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Scenario Comparison")

all_scenarios = dict(preset_results)
if custom_params and raw_df is not None:
    result = _run_simulation(custom_params, elasticities, raw_df)
    all_scenarios["Custom"] = _scenario_result_to_dict(result)

comp_df = pd.DataFrame(all_scenarios.values())
comp_df = comp_df.rename(columns={
    "scenario": "Scenario",
    "delivery_change_pct": "Delivery Δ%",
    "residual_direction": "Residual Dir.",
    "anomaly_shift": "Anomaly Shift",
    "confidence": "Confidence",
})

def _highlight_delivery(val):
    if isinstance(val, (int, float)):
        color = "#38A169" if val > 0 else "#E53E3E" if val < 0 else "#94A3B8"
        return f"color: {color}; font-weight: bold;"
    return ""

def _highlight_confidence(val):
    color = _confidence_color(str(val))
    return f"color: {color}; font-weight: bold;"

styled = comp_df.style.applymap(_highlight_delivery, subset=["Delivery Δ%"]).applymap(_highlight_confidence, subset=["Confidence"])
st.dataframe(styled, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────
# SECTION 4: ELASTICITY SENSITIVITY
# ──────────────────────────────────────────────────────────────────────
st.markdown("### Elasticity Sensitivity Analysis")

delivery_elast = elasticities.elasticities.get("delivery_response", 0.15)
fx_range = np.linspace(-30, 30, 100)
delivery_response = delivery_elast * fx_range * 100

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=fx_range, y=delivery_response, mode="lines+markers",
    marker=dict(color="#D4AF37", size=4, opacity=0.6),
    line=dict(color="#D4AF37", width=2),
    name="Delivery Response",
))
fig2.update_layout(
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font_color="#94A3B8",
    xaxis=dict(gridcolor="#2D3748", title="FX Spread Change (%)"),
    yaxis=dict(gridcolor="#2D3748", title="Estimated Delivery Response (%)"),
    margin=dict(t=10, b=10, l=10, r=10), height=300,
)
fig2.add_vline(x=0, line_dash="dash", line_color="#94A3B8", opacity=0.5)
fig2.add_hline(y=0, line_dash="dash", line_color="#94A3B8", opacity=0.5)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(f"<p style='color:#64748B;font-size:0.8rem;'>Estimated delivery response elasticity: <b style='color:#D4AF37;'>{delivery_elast:+.4f}</b>. This represents the historical delivery response to policy shocks.</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='color:#64748B;font-size:0.8rem;text-align:center;'>Scenario outputs are directional intelligence signals. No deterministic claims are made about future outcomes.</p>", unsafe_allow_html=True)
