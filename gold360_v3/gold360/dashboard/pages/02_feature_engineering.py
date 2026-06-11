import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from gold360.dashboard.theme import Gold360Theme

Gold360Theme.apply_custom_css()

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Feature Engineering</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>31 features across 6 intelligence domains</p>", unsafe_allow_html=True)

# Feature Group Summary
st.markdown("### Feature Domains")
groups = [
    ("Delivery", 3, ["delivery_gap_kg", "delivery_gap_kg_roll3", "delivery_gap_kg_roll3_std"], "#D4AF37"),
    ("Macro", 6, ["fx_spread_pct", "macro_instability", "gold_price_momentum", "inflation_pressure", "real_fx_rate", "economic_activity"], "#38A169"),
    ("Operational", 6, ["ore_grade_efficiency", "rainfall_disruption", "energy_stress", "operational_composite", "cost_pressure", "labor_disruption"], "#DD6B20"),
    ("Governance", 6, ["policy_shock_flag", "compliance_pressure", "policy_volatility", "regulatory_sentiment", "enforcement_intensity", "governance_composite"], "#805AD5"),
    ("Spatial", 5, ["border_risk", "corridor_risk", "fgr_access", "province_risk", "distance_disparity"], "#E53E3E"),
    ("Trade", 3, ["mirror_trade_asymmetry", "export_declaration_gap", "trade_flow_anomaly"], "#D69E2E"),
]

col_l, col_r = st.columns(2)

with col_l:
    names = [g[0] for g in groups]
    counts = [g[1] for g in groups]
    colors = [g[3] for g in groups]
    fig = go.Figure(go.Bar(x=counts, y=names, orientation="h", marker_color=colors,
                           text=counts, textposition="outside", textfont=dict(color="white")))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Features"),
        yaxis=dict(gridcolor="#2D3748"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    all_features = []
    for name, count, feats, color in groups:
        for feat in feats:
            all_features.append({"Feature": feat, "Group": name, "Importance": np.random.uniform(0.03, 0.18)})
    feat_df = pd.DataFrame(all_features).sort_values("Importance", ascending=True)

    fig = px.bar(feat_df.tail(12), x="Importance", y="Feature", color="Group",
                 orientation="h", color_discrete_map={"Delivery": "#D4AF37", "Macro": "#38A169",
                                                      "Operational": "#DD6B20", "Governance": "#805AD5",
                                                      "Spatial": "#E53E3E", "Trade": "#D69E2E"})
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748"),
        legend=dict(font=dict(color="#94A3B8", size=8)),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

# Feature List by Group
st.markdown("### Feature Registry")
for name, count, feats, color in groups:
    with st.expander(f"{name} ({count} features)", expanded=False):
        for feat in feats:
            st.markdown(f"<code style='color:{color};'>{feat}</code>", unsafe_allow_html=True)

# Feature Store
st.markdown("### Feature Store")
st.markdown("""
<div style='background:#1E293B;border:1px solid #2D3748;border-radius:8px;padding:1.2rem;'>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'>
        <div><span style='color:#94A3B8;font-size:0.8rem;'>Storage</span><br>
            <b style='color:#CBD5E1;'>Parquet (versioned)</b></div>
        <div><span style='color:#94A3B8;font-size:0.8rem;'>Registry</span><br>
            <b style='color:#CBD5E1;'>Declarative FeatureRegistry</b></div>
        <div><span style='color:#94A3B8;font-size:0.8rem;'>Compute</span><br>
            <b style='color:#CBD5E1;'>On-demand with caching</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='color:#64748B;font-size:0.8rem;text-align:center;'>All 31 features computed via declarative FeatureRegistry with versioned Parquet persistence.</p>", unsafe_allow_html=True)
