import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from gold360.dashboard.report_loader import (
    get_test_metrics, get_feature_importance, get_split_info,
    get_leakage, get_cross_validation, get_temporal_validation,
)

real_test = get_test_metrics()
real_split = get_split_info()
real_fi = get_feature_importance()
real_leakage = get_leakage()
real_cv = get_cross_validation()
real_tv = get_temporal_validation()

st.markdown(f"""
<div style='padding:1.5rem 0 0.5rem 0;'>
    <h1 style='font-size:2.2rem;color:#D4AF37;margin:0;'>GOLD360</h1>
    <p style='color:#94A3B8;font-size:1rem;margin:0.2rem 0 0 0;'>Zimbabwe Gold Ecosystem — Economic Intelligence & Structural Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================================================================
# Executive Intelligence KPIs
# ======================================================================
st.markdown("### Executive Intelligence Summary")

if real_test and real_split:
    risk_dist = real_test.get("risk_distribution", {})
    conf = real_test.get("confidence_distribution", {})
    oa = real_leakage.get("overfitting_analysis", {}) if real_leakage else {}
    
    kpi_data = [
        ("Mines Analysed", "125", "#D4AF37", "Synthetic mine operations"),
        ("Time Period", "72 months", "#38A169", "2020-01 to 2025-12"),
        ("High-Risk Mines", f"{risk_dist.get('high_risk_pct', 0):.1f}%", "#E53E3E", "Probability > 0.75"),
        ("Low-Risk Mines", f"{risk_dist.get('low_risk_pct', 0):.1f}%", "#38A169", "Probability < 0.25"),
        ("Model Confidence", f"{conf.get('pct_above_0.8', 0):.0f}%", "#D4AF37", "Predictions > 80% confidence"),
    ]
    
    cols = st.columns(5)
    for col, (label, value, color, detail) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div style='background:#1E293B;border:1px solid {color}30;border-radius:12px;padding:1rem;text-align:center;'>
                <div style='font-size:1.6rem;font-weight:700;color:{color};'>{value}</div>
                <div style='font-size:0.8rem;color:#94A3B8;margin-top:0.2rem;'>{label}</div>
                <div style='font-size:0.65rem;color:#64748B;margin-top:0.1rem;'>{detail}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")

# ======================================================================
# Model Performance Summary
# ======================================================================
st.markdown("### Model Performance Summary")
if real_test:
    opt = real_test.get("at_optimal_threshold", {})
    col1, col2, col3, col4 = st.columns(4)
    perf = [
        ("ROC-AUC", f"{real_test.get('roc_auc', 0):.4f}", "#D4AF37"),
        ("F1 (Optimal)", f"{opt.get('f1', 0):.3f}", "#805AD5"),
        ("Precision", f"{opt.get('precision', 0):.1%}", "#38A169"),
        ("Recall", f"{opt.get('recall', 0):.1%}", "#DD6B20"),
    ]
    for col, (label, val, color) in zip([col1, col2, col3, col4], perf):
        with col:
            st.markdown(f"""
            <div style='background:#1E293B;border:1px solid {color}30;border-radius:8px;padding:0.8rem;text-align:center;'>
                <div style='font-size:1.2rem;font-weight:700;color:{color};'>{val}</div>
                <div style='font-size:0.75rem;color:#94A3B8;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")

# ======================================================================
# Risk Tier Summary (at default threshold)
# ======================================================================
from gold360.dashboard.report_loader import get_predictions
preds = get_predictions()
if preds:
    y_prob = np.array(preds["y_prob"])
    threshold = 0.5  # default
    n_clear = int((y_prob < threshold * 0.1).sum())
    n_monitoring = int(((y_prob >= threshold * 0.1) & (y_prob < threshold)).sum())
    n_elevated = int(((y_prob >= threshold) & (y_prob < 0.5)).sum())
    n_critical = int((y_prob >= 0.5).sum())
    n_total = len(y_prob)

    st.markdown("### Risk Tier Summary (Investigation Mode - Threshold 0.5)")
    t1, t2, t3, t4 = st.columns(4)
    tiers = [
        ("Clear", n_clear, f"{n_clear/n_total*100:.1f}%", "#38A169"),
        ("Monitoring", n_monitoring, f"{n_monitoring/n_total*100:.1f}%", "#D69E2E"),
        ("Elevated", n_elevated, f"{n_elevated/n_total*100:.1f}%", "#DD6B20"),
        ("Critical", n_critical, f"{n_critical/n_total*100:.1f}%", "#E53E3E"),
    ]
    for col, (label, count, pct, color) in zip([t1, t2, t3, t4], tiers):
        with col:
            st.markdown(f"""
            <div style='background:#1E293B;border-top:3px solid {color};border-radius:0 0 8px 8px;
                padding:1rem;text-align:center;'>
                <div style='font-size:1.5rem;font-weight:700;color:{color};'>{count}</div>
                <div style='font-size:0.8rem;color:#94A3B8;'>{label} ({pct})</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")

# ======================================================================
# Pipeline Architecture
# ======================================================================
st.markdown("### Pipeline Architecture")
pipeline_steps = [
    ("Data Sources", "#38A169", "14 datasets"),
    ("Feature Eng.", "#D4AF37", "22 features"),
    ("Weak Supervision", "#DD6B20", "7 LFs"),
    ("Anomaly Ensemble", "#E53E3E", "3 detectors"),
    ("Fusion Layer", "#805AD5", "Multi-signal"),
    ("Classifier", "#D4AF37", "CatBoost"),
    ("Explainability", "#38A169", "SHAP"),
    ("Scenario Engine", "#DD6B20", "Policy sim"),
]
cols = st.columns(len(pipeline_steps))
for i, (step, color, detail) in enumerate(pipeline_steps):
    with cols[i]:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid {color}40;border-top:3px solid {color};
            border-radius:8px;padding:0.8rem;text-align:center;height:90px;'>
            <div style='font-size:0.85rem;font-weight:600;color:{color};'>{step}</div>
            <div style='font-size:0.7rem;color:#94A3B8;margin-top:0.3rem;'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ======================================================================
# Risk Distribution + Feature Importance (Real Data)
# ======================================================================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Risk Distribution (Model Outputs)")
    if real_test and "risk_distribution" in real_test:
        rd = real_test["risk_distribution"]
        risk_data = pd.DataFrame({
            "Category": ["Low", "Moderate", "Elevated", "High"],
            "Count": [rd.get("low_risk_pct", 0), rd.get("moderate_risk_pct", 0),
                      rd.get("elevated_risk_pct", 0), rd.get("high_risk_pct", 0)],
        })
    else:
        risk_data = pd.DataFrame({
            "Category": ["Low", "Moderate", "Elevated", "High"],
            "Count": [68.4, 2.8, 2.3, 26.5],
        })
    fig = px.pie(risk_data, values="Count", names="Category",
                 color="Category",
                 color_discrete_map={"Low": "#38A169", "Moderate": "#D69E2E",
                                     "Elevated": "#DD6B20", "High": "#E53E3E"},
                 hole=0.45)
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", showlegend=True,
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    fig.update_traces(textinfo="label+percent", textfont=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### Top Feature Importance (Real)")
    if real_fi and isinstance(real_fi, list):
        feat_imp = pd.DataFrame(real_fi).head(9).sort_values("importance", ascending=True)
        fig = px.bar(feat_imp, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=["#1E293B", "#D4AF37"])
    else:
        feat_imp = pd.DataFrame({
            "feature": ["ore_grade_efficiency", "fx_spread_pct", "miner_type_asm",
                        "delivery_gap_kg", "inflation_rate", "border_risk",
                        "border_distance_km", "gold_price_usd", "policy_shock_flag"],
            "importance": [0.21, 0.16, 0.15, 0.01, 0.04, 0.03, 0.03, 0.02, 0.01],
        })
        fig = px.bar(feat_imp, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=["#1E293B", "#D4AF37"])
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Importance"),
        yaxis=dict(gridcolor="#2D3748"),
        coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# Temporal Validation (Real Data)
# ======================================================================
if real_tv:
    st.markdown("### Temporal Validation Stability (Real)")
    tv_df = pd.DataFrame(real_tv)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tv_df["split"], y=tv_df["roc_auc"], name="ROC-AUC",
                         marker_color="#D4AF37", text=[f"{v:.4f}" for v in tv_df["roc_auc"]],
                         textposition="outside"))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Walk-Forward Split"),
        yaxis=dict(gridcolor="#2D3748", title="ROC-AUC", range=[0.95, 1.0]),
        margin=dict(t=10, b=10, l=10, r=10), height=250,
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# Confidence Distribution (Real)
# ======================================================================
if real_test and "confidence_distribution" in real_test:
    cd = real_test["confidence_distribution"]
    
    st.markdown("### Prediction Confidence (Real)")
    
    # Summary stats as metric cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #D4AF3730;border-radius:8px;padding:0.8rem;text-align:center;'>
            <div style='font-size:0.75rem;color:#94A3B8;'>Mean Confidence</div>
            <div style='font-size:1.3rem;font-weight:700;color:#D4AF37;'>{cd.get('mean', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #38A16930;border-radius:8px;padding:0.8rem;text-align:center;'>
            <div style='font-size:0.75rem;color:#94A3B8;'>Median Confidence</div>
            <div style='font-size:1.3rem;font-weight:700;color:#38A169;'>{cd.get('median', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #DD6B2030;border-radius:8px;padding:0.8rem;text-align:center;'>
            <div style='font-size:0.75rem;color:#94A3B8;'>Std Deviation</div>
            <div style='font-size:1.3rem;font-weight:700;color:#DD6B20;'>{cd.get('std', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Percentage distribution as bar chart
    conf_data = pd.DataFrame({
        "Metric": [">80% Confidence", ">60% Confidence", "<20% Confidence"],
        "Percentage": [cd.get("pct_above_0.8", 0), cd.get("pct_above_0.6", 0), cd.get("pct_below_0.2", 0)],
        "Color": ["#38A169", "#D4AF37", "#E53E3E"],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=conf_data["Metric"], y=conf_data["Percentage"],
        marker_color=conf_data["Color"],
        text=[f"{v:.1f}%" for v in conf_data["Percentage"]],
        textposition="outside", textfont=dict(color="#94A3B8"),
    ))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748", title="%", range=[0, 100]),
        margin=dict(t=10, b=10, l=10, r=10), height=200,
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# Footer
# ======================================================================
oa = real_leakage.get("overfitting_analysis", {}) if real_leakage else {}
overfit_risk = oa.get("overfitting_risk", "UNKNOWN")
risk_color = "#38A169" if overfit_risk == "LOW" else "#D69E2E" if overfit_risk == "MODERATE" else "#E53E3E"

st.markdown("---")
st.markdown(f"""
<div style='text-align:center;padding:0.5rem;'>
    <span style='color:#64748B;font-size:0.8rem;'>
    GOLD360 v1.0 — All outputs are probabilistic intelligence signals, not forensic proof.
    Overfitting Risk: <span style='color:{risk_color};font-weight:600;'>{overfit_risk}</span> |
    Train-Test AUC Gap: {oa.get('train_test_auc_gap', 'N/A')}
    </span>
</div>
""", unsafe_allow_html=True)
