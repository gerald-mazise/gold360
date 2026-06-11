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

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Weak Supervision</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>7 labeling functions — Probabilistic pseudo-label generation</p>", unsafe_allow_html=True)

# LF Cards
st.markdown("### Labeling Functions")
lfs = [
    ("extreme_delivery_collapse", "Delivery volumes below 50% of expected", "Delivery", 0.72, "#D4AF37"),
    ("fx_arbitrage_stress", "Extreme FX spread (>50%)", "Macro", 0.65, "#38A169"),
    ("impossible_yield_contradiction", "Contradictory ore grade vs recovery", "Operational", 0.58, "#DD6B20"),
    ("corridor_inconsistency", "Production near borders with low delivery", "Spatial", 0.55, "#E53E3E"),
    ("inventory_anomaly", "Unusual inventory level changes", "Operational", 0.48, "#DD6B20"),
    ("policy_contradiction", "Policy events with contradictory outcomes", "Governance", 0.42, "#805AD5"),
    ("operational_mismatch", "Energy/rainfall vs production mismatch", "Operational", 0.38, "#DD6B20"),
]

fig = px.bar(x=[lf[0] for lf in lfs], y=[lf[3] for lf in lfs],
             color=[lf[2] for lf in lfs],
             color_discrete_map={"Delivery": "#D4AF37", "Macro": "#38A169",
                                 "Operational": "#DD6B20", "Spatial": "#E53E3E",
                                 "Governance": "#805AD5"})
fig.update_layout(
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", tickangle=-45),
    yaxis=dict(gridcolor="#2D3748", title="Activation Rate"),
    showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300,
)
st.plotly_chart(fig, use_container_width=True)

# Fusion Methods
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### Fusion Methods Comparison")
    np.random.seed(42)
    n = 100
    signals = np.random.beta(2, 5, n)
    methods = pd.DataFrame({
        "Sample": range(n),
        "Majority Vote": (signals > 0.5).astype(float),
        "Weighted": signals * np.random.uniform(0.8, 1.2, n),
        "Mean": np.full(n, signals.mean()),
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=methods["Sample"], y=methods["Majority Vote"],
                             name="Majority Vote", line=dict(color="#D4AF37", width=1)))
    fig.add_trace(go.Scatter(x=methods["Sample"], y=methods["Weighted"],
                             name="Weighted", line=dict(color="#38A169", width=2)))
    fig.add_trace(go.Scatter(x=methods["Sample"], y=methods["Mean"],
                             name="Mean", line=dict(color="#DD6B20", dash="dash", width=1)))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Sample Index"),
        yaxis=dict(gridcolor="#2D3748", title="Pseudo-Risk Probability"),
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("### Pseudo-Label Confidence Distribution")
    confidences = np.random.beta(5, 2, 500)
    fig = px.histogram(x=confidences, nbins=30, color_discrete_sequence=["#D4AF37"])
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Confidence Score"),
        yaxis=dict(gridcolor="#2D3748", title="Count"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    fig.add_vline(x=0.7, line_dash="dash", line_color="#38A169", opacity=0.7,
                  annotation_text="High Confidence Threshold")
    st.plotly_chart(fig, use_container_width=True)

# Audit Trail
st.markdown("### Audit Trail Summary")
audit_data = pd.DataFrame({
    "LF Function": ["extreme_delivery_collapse", "fx_arbitrage_stress", "impossible_yield_contradiction",
                    "corridor_inconsistency", "inventory_anomaly", "policy_contradiction", "operational_mismatch"],
    "Calls": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
    "Active": [720, 650, 580, 550, 480, 420, 380],
    "Avg Signal": [0.42, 0.38, 0.31, 0.28, 0.24, 0.19, 0.15],
    "Avg Confidence": [0.68, 0.62, 0.55, 0.51, 0.44, 0.38, 0.32],
})
st.dataframe(audit_data, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='color:#64748B;font-size:0.8rem;text-align:center;'>Pseudo-labels are probabilistic, not ground truth. Every pseudo-label is traceable via the LabelAuditTrail.</p>", unsafe_allow_html=True)
