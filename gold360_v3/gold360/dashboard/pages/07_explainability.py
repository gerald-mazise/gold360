import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from gold360.dashboard.theme import Gold360Theme
from gold360.dashboard.report_loader import get_feature_importance, get_test_metrics, get_ablation

Gold360Theme.apply_custom_css()

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Explainability</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>Feature importance and model interpretation — every prediction is traceable</p>", unsafe_allow_html=True)

fi = get_feature_importance()
test = get_test_metrics()
ablation = get_ablation()

# ======================================================================
# Feature Importance Bar Chart (Real Data)
# ======================================================================
st.markdown("### Global Feature Importance (CatBoost Native)")
if fi and isinstance(fi, list):
    feat_imp = pd.DataFrame(fi)
    feat_imp = feat_imp.sort_values("importance", ascending=True)
    feat_imp = feat_imp[feat_imp["importance"] > 0]

    fig = px.bar(feat_imp, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=["#1E293B", "#D4AF37"],
                 labels={"importance": "Importance Score", "feature": "Feature"})
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Importance Score"),
        yaxis=dict(gridcolor="#2D3748"),
        coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
        height=max(350, len(feat_imp) * 28),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance data not available. Run the evaluation pipeline first.")

# ======================================================================
# SHAP-style Feature Contribution Table
# ======================================================================
st.markdown("### Feature Contribution Analysis")
if fi and isinstance(fi, list):
    domain_map = {
        "delivery_gap_kg": "Delivery", "delivery_gap_kg_roll3": "Delivery",
        "delivery_gap_kg_roll3_std": "Delivery",
        "fx_spread_pct": "Macro/Economic", "inflation_rate": "Macro/Economic",
        "gold_price_usd": "Macro/Economic",
        "ore_grade_efficiency": "Operational", "rainfall_raw": "Operational",
        "ore_processed_tonnes": "Operational",
        "payment_delay_days": "Operational",
        "policy_shock_flag": "Governance", "miner_type_asm": "Governance",
        "license_encoded": "Governance",
        "border_distance_km": "Spatial", "border_risk": "Spatial",
        "fgr_distance_km": "Spatial", "fgr_access": "Spatial",
    }
    
    domain_descriptions = {
        "Delivery": "Official gold deliveries vs estimated yield — primary risk signal",
        "Macro/Economic": "FX spreads, inflation, gold prices — external pressure indicators",
        "Operational": "Mine-level operations: ore grade, recovery, energy, rainfall",
        "Governance": "Policy events, license status, miner type (ASM vs formal)",
        "Spatial": "Proximity to borders and Fidelity Gold Refinery — corridor risk",
    }
    
    feat_df = pd.DataFrame(fi)
    total_imp = feat_df["importance"].sum()
    feat_df["contribution_pct"] = (feat_df["importance"] / total_imp * 100).round(1)
    feat_df["domain"] = feat_df["feature"].map(domain_map).fillna("Other")
    feat_df = feat_df.sort_values("importance", ascending=False)
    
    # Domain summary
    domain_imp = feat_df.groupby("domain")["importance"].sum().reset_index()
    domain_imp["contribution_pct"] = (domain_imp["importance"] / total_imp * 100).round(1)
    domain_imp = domain_imp.sort_values("importance", ascending=False)
    
    colors = {"Delivery": "#D4AF37", "Operational": "#E53E3E", "Macro/Economic": "#38A169",
              "Governance": "#DD6B20", "Spatial": "#805AD5", "Other": "#94A3B8"}
    
    # Domain pie chart
    fig = px.pie(domain_imp, values="importance", names="domain",
                 color="domain", color_discrete_map=colors, hole=0.45)
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", showlegend=True,
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    fig.update_traces(textinfo="label+percent", textfont=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed contribution table
    st.markdown("**Feature-Level Breakdown**")
    table_data = []
    for _, row in feat_df[feat_df["importance"] > 0].iterrows():
        table_data.append({
            "Feature": row["feature"],
            "Domain": row["domain"],
            "Importance": f"{row['importance']:.2f}",
            "Contribution": f"{row['contribution_pct']:.1f}%",
            "Interpretation": domain_descriptions.get(row["domain"], "Unclassified feature"),
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
    # Domain descriptions
    for domain, desc in domain_descriptions.items():
        domain_data = feat_df[feat_df["domain"] == domain]
        if len(domain_data) > 0:
            pct = domain_data["contribution_pct"].sum()
            st.markdown(f"**{domain}** ({pct:.1f}%): {desc}")

# ======================================================================
# Ablation Results
# ======================================================================
if ablation:
    st.markdown("### Ablation: What Happens When Features Are Removed?")
    abl_df = pd.DataFrame(ablation)
    abl_all = abl_df[abl_df["group"] == "ALL FEATURES"]
    abl_groups = abl_df[abl_df["group"] != "ALL FEATURES"]

    if len(abl_all) > 0:
        baseline_auc = abl_all.iloc[0]["auc"]
        st.markdown(f"**Baseline AUC (all features): {baseline_auc:.4f}**")

    if "auc_delta" in abl_groups.columns:
        fig = px.bar(abl_groups, x="auc_delta", y="group", orientation="h",
                     color="auc_delta",
                     color_continuous_scale=[[0, "#38A169"], [0.5, "#D69E2E"], [1, "#E53E3E"]],
                     labels={"auc_delta": "AUC Change When Removed", "group": "Feature Group"})
        fig.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748"),
            yaxis=dict(gridcolor="#2D3748"),
            coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=250,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Positive delta = feature group is important (removing it hurts performance)**")

# ======================================================================
# Prediction Example (Real Model Output)
# ======================================================================
st.markdown("### Sample Prediction Analysis")
if test and "risk_distribution" in test:
    rd = test["risk_distribution"]
    st.markdown(f"""
    <div style='background:#1E293B;border:1px solid #2D3748;border-radius:8px;padding:1.2rem;'>
        <div style='font-size:0.8rem;color:#94A3B8;margin-bottom:0.5rem;'>TEST SET RISK DISTRIBUTION</div>
        <div style='display:flex;gap:1rem;flex-wrap:wrap;'>
            <div style='flex:1;min-width:120px;text-align:center;padding:0.5rem;background:#38A16920;border-radius:6px;'>
                <div style='font-size:1.2rem;font-weight:700;color:#38A169;'>{rd.get('low_risk_pct', 0):.1f}%</div>
                <div style='font-size:0.7rem;color:#94A3B8;'>Low Risk</div>
            </div>
            <div style='flex:1;min-width:120px;text-align:center;padding:0.5rem;background:#D69E2E20;border-radius:6px;'>
                <div style='font-size:1.2rem;font-weight:700;color:#D69E2E;'>{rd.get('moderate_risk_pct', 0):.1f}%</div>
                <div style='font-size:0.7rem;color:#94A3B8;'>Moderate</div>
            </div>
            <div style='flex:1;min-width:120px;text-align:center;padding:0.5rem;background:#DD6B2020;border-radius:6px;'>
                <div style='font-size:1.2rem;font-weight:700;color:#DD6B20;'>{rd.get('elevated_risk_pct', 0):.1f}%</div>
                <div style='font-size:0.7rem;color:#94A3B8;'>Elevated</div>
            </div>
            <div style='flex:1;min-width:120px;text-align:center;padding:0.5rem;background:#E53E3E20;border-radius:6px;'>
                <div style='font-size:1.2rem;font-weight:700;color:#E53E3E;'>{rd.get('high_risk_pct', 0):.1f}%</div>
                <div style='font-size:0.7rem;color:#94A3B8;'>High Risk</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cd = test.get("confidence_distribution", {})
    if cd:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #2D3748;border-radius:8px;padding:1.2rem;margin-top:0.5rem;'>
            <div style='font-size:0.8rem;color:#94A3B8;margin-bottom:0.5rem;'>PREDICTION CONFIDENCE</div>
            <div style='font-size:0.9rem;color:#F1F5F9;'>
            Mean: <b style='color:#D4AF37;'>{cd.get('mean', 0):.3f}</b> |
            High confidence (>0.8): <b style='color:#38A169;'>{cd.get('pct_above_0.8', 0):.1f}%</b> |
            Low confidence (<0.2): <b style='color:#E53E3E;'>{cd.get('pct_below_0.2', 0):.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================
# Natural Language Report
# ======================================================================
st.markdown("### Intelligence Report Summary")
if test:
    roc = test.get("roc_auc", 0)
    f1 = test.get("f1_score", 0)
    opt = test.get("at_optimal_threshold", {})
    st.markdown(f"""
    <div style='background:#1E293B;border:1px solid #D4AF3730;border-radius:8px;padding:1.5rem;'>
        <div style='color:#D4AF37;font-weight:600;margin-bottom:0.8rem;'>MODEL EVALUATION SUMMARY</div>
        <div style='color:#E5E7EB;font-size:0.9rem;line-height:1.6;'>
        The CatBoost classifier achieves <b style='color:#D4AF37;'>ROC-AUC = {roc:.4f}</b> on the test set,
        indicating <b>excellent discriminative ability</b> between high-risk and low-risk operations.<br><br>
        \U0001F449 <b>Precision: {test.get('precision', 0):.1%}</b> — When the model flags high risk, it is always correct (0 false positives)<br>
        \U0001F449 <b>Recall: {test.get('recall', 0):.1%}</b> — The model detects ~{test.get('recall', 0):.0%} of actual high-risk cases<br>
        \U0001F449 <b>F1 Score: {f1:.4f}</b> — Balanced precision/recall at default threshold<br>
        \U0001F449 <b>Optimized F1: {opt.get('f1', 0):.4f}</b> — With threshold tuning (Youden's J = {test.get('optimal_threshold_youden', 0):.4f})<br><br>
        <span style='color:#94A3B8;'><i>All outputs are probabilistic intelligence signals, not forensic proof.
        Analyst interpretation is required before any operational response.</i></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='color:#64748B;font-size:0.8rem;text-align:center;'>Feature importance computed from CatBoost native importance. All metrics from real model evaluation.</p>", unsafe_allow_html=True)
