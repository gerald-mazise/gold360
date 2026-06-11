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

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Data Pipeline</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>14 datasets — Multi-source temporal harmonization</p>", unsafe_allow_html=True)

# Data Source Overview
st.markdown("### Dataset Overview")
datasets = [
    ("FGR Deliveries", "Quarterly", 1000, "Economic", "#D4AF37"),
    ("ZIMSTAT Production", "Quarterly", 800, "Economic", "#D4AF37"),
    ("Synthetic Mine Ops", "Monthly", 12500, "Operational", "#DD6B20"),
    ("Gold Price", "Monthly", 60, "Market", "#38A169"),
    ("FX Distortion", "Annual", 5, "Macro", "#805AD5"),
    ("Inflation", "Annual", 5, "Macro", "#805AD5"),
    ("Rainfall", "Monthly", 360, "Environmental", "#38A169"),
    ("Energy", "Annual", 5, "Operational", "#DD6B20"),
    ("Policy Events", "Event", 20, "Governance", "#E53E3E"),
    ("FGR Offices", "Static", 5, "Spatial", "#38A169"),
    ("Mirror Trade", "Annual", 5, "Trade", "#DD6B20"),
    ("Gold Exports", "Annual", 5, "Trade", "#DD6B20"),
    ("Smuggling Incidents", "Event", 15, "Enforcement", "#E53E3E"),
]

ds_df = pd.DataFrame(datasets, columns=["Dataset", "Frequency", "Records", "Type", "Color"])

col_l, col_r = st.columns(2)

with col_l:
    fig = px.bar(ds_df, x="Records", y="Dataset", orientation="h", color="Type",
                 color_discrete_map={"Economic": "#D4AF37", "Operational": "#DD6B20",
                                     "Market": "#38A169", "Macro": "#805AD5",
                                     "Environmental": "#38A169", "Governance": "#E53E3E",
                                     "Spatial": "#38A169", "Trade": "#DD6B20",
                                     "Enforcement": "#E53E3E"})
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Records"),
        yaxis=dict(gridcolor="#2D3748"),
        legend=dict(font=dict(color="#94A3B8", size=9)),
        margin=dict(t=10, b=10, l=10, r=10), height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    type_counts = ds_df["Type"].value_counts()
    fig = px.pie(values=type_counts.values, names=type_counts.index,
                 color_discrete_sequence=["#D4AF37", "#DD6B20", "#38A169", "#805AD5",
                                          "#E53E3E", "#38A169", "#DD6B20", "#E53E3E", "#38A169"][:len(type_counts)])
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", showlegend=True,
        legend=dict(font=dict(color="#94A3B8", size=10)),
        margin=dict(t=10, b=10, l=10, r=10), height=400,
    )
    fig.update_traces(textinfo="label+value", textfont=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

# Pipeline Stages
st.markdown("### Pipeline Stages")
stages = [
    ("Loading", "Unified DataLoader with caching", "#38A169", 14),
    ("Validation", "Missing, temporal, duplicate, outlier checks", "#D4AF37", 4),
    ("Temporal Alignment", "Quarterly/Annual → Monthly expansion", "#DD6B20", 10),
    ("Harmonization", "Multi-source fusion → unified intelligence table", "#805AD5", 1),
]
cols = st.columns(len(stages))
for col, (name, desc, color, count) in zip(cols, stages):
    with col:
        st.markdown(f"""
        <div style='background:#1E293B;border-left:3px solid {color};border-radius:0 8px 8px 0;padding:1rem;'>
            <div style='font-size:1.2rem;font-weight:700;color:{color};'>{count}</div>
            <div style='font-size:0.85rem;color:#CBD5E1;font-weight:600;'>{name}</div>
            <div style='font-size:0.75rem;color:#94A3B8;margin-top:0.2rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Synthetic Data Notice
st.markdown("### Data Governance")
st.markdown("""
<div style='background:#DD6B2015;border:1px solid #DD6B2040;border-radius:8px;padding:1rem;'>
    <span style='color:#DD6B20;font-weight:600;'>⚠ Synthetic Data Notice:</span>
    <span style='color:#94A3B8;'> The 'Synthetic Mine Ops' dataset is simulated data used for development and visualization.
    It is explicitly distinguished from observed data and must not drive final research conclusions.</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='color:#64748B;font-size:0.8rem;text-align:center;'>Data lineage tracked via DataLineage registry. All datasets validated before feature engineering.</p>", unsafe_allow_html=True)
