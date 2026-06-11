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

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Anomaly Detection Ensemble</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>Three-detector ensemble with weighted consensus scoring</p>", unsafe_allow_html=True)

# Detector Cards
col1, col2, col3 = st.columns(3)
detectors = [
    ("IsolationForest", "0.40", "Tree-based isolation. Flags samples requiring few splits to isolate.", "#D4AF37"),
    ("ECOD", "0.30", "Empirical CDF-based. Non-parametric tail probability estimation.", "#38A169"),
    ("LOF", "0.30", "Local Outlier Factor. Density-based local anomaly detection.", "#DD6B20"),
]
for col, (name, weight, desc, color) in zip([col1, col2, col3], detectors):
    with col:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid {color}40;border-top:3px solid {color};
            border-radius:8px;padding:1.2rem;'>
            <div style='font-size:1rem;font-weight:700;color:{color};'>{name}</div>
            <div style='font-size:0.75rem;color:#94A3B8;margin:0.2rem 0;'>Weight: {weight}</div>
            <div style='font-size:0.85rem;color:#CBD5E1;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# Consensus Score Timeline
st.markdown("### Anomaly Consensus Score Timeline")
np.random.seed(42)
months = pd.date_range("2020-01-01", periods=60, freq="ME")
if_scores = np.random.beta(2, 5, 60) * 100
ecod_scores = np.random.beta(2, 5, 60) * 100
lof_scores = np.random.beta(2, 5, 60) * 100
consensus = 0.4 * if_scores + 0.3 * ecod_scores + 0.3 * lof_scores
consensus[15:18] = 82
consensus[42:44] = 88
flagged = consensus > 50

score_df = pd.DataFrame({
    "Month": months,
    "IsolationForest": if_scores,
    "ECOD": ecod_scores,
    "LOF": lof_scores,
    "Consensus": consensus,
    "Flagged": flagged,
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=score_df["Month"], y=score_df["IsolationForest"],
                         name="IF", line=dict(color="#D4AF37", width=1), opacity=0.5))
fig.add_trace(go.Scatter(x=score_df["Month"], y=score_df["ECOD"],
                         name="ECOD", line=dict(color="#38A169", width=1), opacity=0.5))
fig.add_trace(go.Scatter(x=score_df["Month"], y=score_df["LOF"],
                         name="LOF", line=dict(color="#DD6B20", width=1), opacity=0.5))
fig.add_trace(go.Scatter(x=score_df["Month"], y=score_df["Consensus"],
                         name="Consensus", line=dict(color="#E53E3E", width=3)))
fig.add_hline(y=50, line_dash="dash", line_color="#E53E3E", opacity=0.5, annotation_text="Flag Threshold")
fig.update_layout(
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748"),
    yaxis=dict(gridcolor="#2D3748", title="Anomaly Score (0-100)", range=[0, 100]),
    legend=dict(font=dict(color="#94A3B8"), orientation="h"),
    margin=dict(t=10, b=10, l=10, r=10), height=350,
)
st.plotly_chart(fig, use_container_width=True)

# Agreement Distribution + Flagged Months
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### Detector Agreement Distribution")
    agreement_levels = []
    for i in range(60):
        above = sum([if_scores[i] > 50, ecod_scores[i] > 50, lof_scores[i] > 50])
        if above == 3:
            agreement_levels.append("Full (3/3)")
        elif above == 2:
            agreement_levels.append("Majority (2/3)")
        elif above == 1:
            agreement_levels.append("Single (1/3)")
        else:
            agreement_levels.append("None (0/3)")

    agree_df = pd.DataFrame({"Agreement": agreement_levels})
    counts = agree_df["Agreement"].value_counts()
    colors_map = {"Full (3/3)": "#E53E3E", "Majority (2/3)": "#DD6B20",
                  "Single (1/3)": "#D69E2E", "None (0/3)": "#38A169"}

    fig = px.bar(x=counts.index, y=counts.values,
                 color=counts.index, color_discrete_map=colors_map)
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748", title="Count"),
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("### Flagged Months")
    flagged_df = score_df[score_df["Flagged"]].copy()
    flagged_df["Risk Level"] = flagged_df["Consensus"].apply(
        lambda x: "HIGH" if x > 75 else "MODERATE" if x > 60 else "ELEVATED")
    flagged_df["Risk Color"] = flagged_df["Risk Level"].map(
        {"HIGH": "#E53E3E", "MODERATE": "#D69E2E", "ELEVATED": "#DD6B20"})

    for _, row in flagged_df.head(6).iterrows():
        st.markdown(f"""
        <div style='background:#1E293B;border-left:3px solid {row["Risk Color"]};
            border-radius:0 8px 8px 0;padding:0.6rem 1rem;margin-bottom:0.4rem;'>
            <span style='font-weight:600;color:{row["Risk Color"]};'>{row["Risk Level"]}</span>
            <span style='color:#94A3B8;margin-left:0.5rem;'>{row["Month"].strftime("%b %Y")}</span>
            <span style='color:#CBD5E1;float:right;'>Score: {row["Consensus"]:.1f}</span>
        </div>
        """, unsafe_allow_html=True)

# Calibration
st.markdown("### Calibration Curve")
np.random.seed(99)
cal_scores = np.random.uniform(0, 1, 200)
cal_labels = (cal_scores > 0.5).astype(int)
calibrated = np.clip(cal_scores + np.random.normal(0, 0.05, 200), 0, 1)

cal_df = pd.DataFrame({"Raw": cal_scores, "Calibrated": calibrated, "True Label": cal_labels})
fig = go.Figure()
fig.add_trace(go.Scatter(x=cal_df["Raw"], y=cal_df["Calibrated"], mode="markers",
                         marker=dict(color=cal_df["True Label"], colorscale=["#38A169", "#E53E3E"],
                                     size=6, opacity=0.6), name="Samples"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                         line=dict(color="#94A3B8", dash="dash", width=1)))
fig.update_layout(
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Raw Score"),
    yaxis=dict(gridcolor="#2D3748", title="Calibrated Probability"),
    legend=dict(font=dict(color="#94A3B8")),
    margin=dict(t=10, b=10, l=10, r=10), height=300,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<p style='color:#64748B;font-size:0.8rem;text-align:center;'>Anomaly scores are normalized to 0-100. Consensus = 0.40×IF + 0.30×ECOD + 0.30×LOF.</p>", unsafe_allow_html=True)
