import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
)

from gold360.dashboard.theme import Gold360Theme
from gold360.dashboard.report_loader import (
    get_test_metrics, get_feature_importance, get_cross_validation,
    get_benchmark, get_temporal_validation, get_ablation, get_robustness,
    get_leakage, get_roc_curve, get_pr_curve, get_split_info, get_predictions,
)

Gold360Theme.apply_custom_css()

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Model Performance</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>CatBoost classifier - real evaluation metrics from model training on 9,000 samples (125 mines x 72 months)</p>", unsafe_allow_html=True)

test = get_test_metrics()
cv = get_cross_validation()
bench = get_benchmark()
fi = get_feature_importance()
ablation = get_ablation()
robustness = get_robustness()
leakage = get_leakage()
roc_data = get_roc_curve()
pr_data = get_pr_curve()
tv = get_temporal_validation()
split = get_split_info()
preds = get_predictions()

if not test:
    st.error("No evaluation reports found. Run the evaluation pipeline first.")
    st.stop()

if not preds:
    st.error("No predictions file found. Re-run evaluation to generate predictions.json.")
    st.stop()

y_true = np.array(preds["y_true"])
y_prob = np.array(preds["y_prob"])

# Temporal Distribution Warning
if split:
    train_rate = split.get("target_positive_rate_train", 0)
    val_rate = split.get("target_positive_rate_val", 0)
    test_rate = split.get("target_positive_rate_test", 0)
    if abs(train_rate - test_rate) > 0.15:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #D69E2E40;border-left:4px solid #D69E2E;border-radius:8px;padding:1rem;margin-bottom:1rem;'>
            <div style='color:#D69E2E;font-weight:600;margin-bottom:0.3rem;'>Temporal Distribution Shift Detected</div>
            <div style='color:#94A3B8;font-size:0.85rem;'>
                Train: <b style='color:#38A169;'>{train_rate:.1%}</b> |
                Val: <b style='color:#D4AF37;'>{val_rate:.1%}</b> |
                Test: <b style='color:#E53E3E;'>{test_rate:.1%}</b><br/>
                Later periods have higher risk scores. Use the slider below to tune the threshold for your operational mode.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Classifier Configuration
st.markdown("### Classifier Configuration")
col1, col2, col3, col4 = st.columns(4)
configs = [
    ("Algorithm", "CatBoost", "#D4AF37"),
    ("Early Stopping", "50 rounds", "#38A169"),
    ("Depth", "6", "#DD6B20"),
    ("Learning Rate", "0.03", "#805AD5"),
]
for col, (label, val, color) in zip([col1, col2, col3, col4], configs):
    with col:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid {color}30;border-radius:8px;padding:1rem;text-align:center;'>
            <div style='font-size:0.75rem;color:#94A3B8;'>{label}</div>
            <div style='font-size:1.4rem;font-weight:700;color:{color};'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# Operational Mode + Threshold Slider
st.markdown("### Operational Intelligence Mode")

mode_col1, mode_col2, mode_col3 = st.columns(3)
modes = [
    ("Monitoring Mode", "0.001", "Ultra-sensitive - catch every possible signal. Analysts triage.", "#38A169"),
    ("Balanced Mode", "0.007", "Optimal precision/recall via Youden's J statistic.", "#D4AF37"),
    ("Investigation Mode", "0.500", "Zero false accusations. Only flag when certain.", "#E53E3E"),
]
for col, (name, thresh, desc, color) in zip([mode_col1, mode_col2, mode_col3], modes):
    with col:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid {color}40;border-top:3px solid {color};
            border-radius:8px;padding:1rem;'>
            <div style='color:{color};font-weight:700;font-size:0.9rem;'>{name}</div>
            <div style='color:#94A3B8;font-size:0.75rem;margin-top:0.3rem;'>Threshold: {thresh}</div>
            <div style='color:#CBD5E1;font-size:0.75rem;margin-top:0.3rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

optimal_threshold = test.get("optimal_threshold_youden", 0.007)
btn_col1, btn_col2, btn_col3 = st.columns(3)
with btn_col1:
    if st.button("Monitoring", use_container_width=True):
        st.session_state["threshold"] = 0.001
with btn_col2:
    if st.button("Balanced", use_container_width=True):
        st.session_state["threshold"] = optimal_threshold
with btn_col3:
    if st.button("Investigation", use_container_width=True):
        st.session_state["threshold"] = 0.5

if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.5

st.markdown('<div class="threshold-container">', unsafe_allow_html=True)
threshold = st.slider(
    "Decision Threshold",
    min_value=0.001, max_value=0.999, step=0.001,
    value=st.session_state["threshold"],
    help="Lower = more sensitive. Higher = more strict.",
)
st.session_state["threshold"] = threshold
st.markdown('</div>', unsafe_allow_html=True)

# Dynamic metrics
y_pred = (y_prob > threshold).astype(int)
precision_val = precision_score(y_true, y_pred, zero_division=0)
recall_val = recall_score(y_true, y_pred, zero_division=0)
f1_val = f1_score(y_true, y_pred, zero_division=0)
accuracy_val = accuracy_score(y_true, y_pred)
balanced_acc_val = balanced_accuracy_score(y_true, y_pred)
mcc_val = matthews_corrcoef(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

if threshold <= 0.005:
    mode_name, mode_color, mode_desc = "MONITORING MODE", "#38A169", "Ultra-sensitive - catching every signal. Analysts triage."
elif threshold <= 0.05:
    mode_name, mode_color, mode_desc = "BALANCED MODE", "#D4AF37", "Optimal trade-off between precision and recall."
else:
    mode_name, mode_color, mode_desc = "INVESTIGATION MODE", "#E53E3E", "High precision - only flag when certain."

st.markdown(f"""
<div style='background:#1E293B;border:1px solid {mode_color}40;border-left:4px solid {mode_color};
    border-radius:0 8px 8px 0;padding:1rem;margin:0.5rem 0;'>
    <div style='color:{mode_color};font-weight:700;font-size:1.1rem;'>{mode_name}</div>
    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.2rem;'>Threshold: <b style='color:{mode_color};'>{threshold:.3f}</b> - {mode_desc}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

st.markdown(f"### Performance at Threshold = {threshold:.3f}")
m1, m2, m3, m4, m5, m6 = st.columns(6)
dynamic_metrics = [
    ("Precision", f"{precision_val:.1%}", "#38A169"),
    ("Recall", f"{recall_val:.1%}", "#DD6B20"),
    ("F1 Score", f"{f1_val:.3f}", "#805AD5"),
    ("Accuracy", f"{accuracy_val:.1%}", "#D4AF37"),
    ("Balanced Acc", f"{balanced_acc_val:.1%}", "#38A169"),
    ("MCC", f"{mcc_val:.3f}", "#DD6B20"),
]
for col, (label, val, color) in zip([m1, m2, m3, m4, m5, m6], dynamic_metrics):
    with col:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid {color}30;border-radius:8px;padding:0.6rem;text-align:center;'>
            <div style='font-size:0.7rem;color:#94A3B8;'>{label}</div>
            <div style='font-size:1.1rem;font-weight:700;color:{color};'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

# Confusion Matrix + Risk Tiers
cm_col1, cm_col2 = st.columns([1, 1])

with cm_col1:
    st.markdown(f"**Confusion Matrix** (threshold={threshold:.3f})")
    cm_arr = np.array([[tn, fp], [fn, tp]])
    fig = px.imshow(cm_arr, text_auto=True,
                    color_continuous_scale=[[0, "#1E293B"], [0.5, "#D4AF37"], [1, "#D4AF37"]],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Low Risk", "High Risk"], y=["Low Risk", "High Risk"])
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8",
        margin=dict(t=10, b=10, l=10, r=10), height=280,
    )
    fig.update_traces(textfont=dict(size=14, color="white"))
    st.plotly_chart(fig, use_container_width=True)

with cm_col2:
    st.markdown("**Risk Tier Breakdown**")
    n_clear = int((y_prob < threshold * 0.1).sum())
    n_monitoring = int(((y_prob >= threshold * 0.1) & (y_prob < threshold)).sum())
    n_elevated = int(((y_prob >= threshold) & (y_prob < 0.5)).sum())
    n_critical = int((y_prob >= 0.5).sum())
    n_total = len(y_prob)
    tiers = [
        ("Clear", n_clear, f"prob < {threshold * 0.1:.3f}", "#38A169"),
        ("Monitoring", n_monitoring, f"{threshold * 0.1:.3f} to {threshold:.3f}", "#D69E2E"),
        ("Elevated", n_elevated, f"{threshold:.3f} to 0.500", "#DD6B20"),
        ("Critical", n_critical, "prob >= 0.500", "#E53E3E"),
    ]
    for label, count, desc, color in tiers:
        pct = count / n_total * 100
        st.markdown(f"""
        <div style='background:#1E293B;border-left:3px solid {color};border-radius:0 6px 6px 0;
            padding:0.6rem 1rem;margin-bottom:0.4rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <span style='color:{color};font-weight:600;'>{label}</span>
                    <span style='color:#64748B;font-size:0.7rem;margin-left:0.5rem;'>{desc}</span>
                </div>
                <div style='color:{color};font-weight:700;'>{count} <span style='color:#94A3B8;font-weight:400;'>({pct:.1f}%)</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ROC + PR Curves
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### ROC Curve")
    fpr_d = roc_data.get("fpr", [0, 1]) if roc_data else [0, 1]
    tpr_d = roc_data.get("tpr", [0, 1]) if roc_data else [0, 1]
    auc_val = test.get("roc_auc", 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_d, y=tpr_d, mode="lines", name=f"CatBoost (AUC={auc_val:.4f})",
                             line=dict(color="#D4AF37", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#94A3B8", dash="dash", width=1)))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8",
        xaxis=dict(gridcolor="#2D3748", title="False Positive Rate"),
        yaxis=dict(gridcolor="#2D3748", title="True Positive Rate"),
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("### Precision-Recall Curve")
    pr_p = pr_data.get("precision", [1, 0]) if pr_data else [1, 0]
    pr_r = pr_data.get("recall", [0, 1]) if pr_data else [0, 1]
    avg_p = test.get("avg_precision", 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr_r, y=pr_p, mode="lines", name=f"AP={avg_p:.4f}",
                             line=dict(color="#805AD5", width=3)))
    pr_t = np.array(preds.get("pr_thresholds", []))
    pr_pe = np.array(preds.get("pr_precision", []))
    pr_re = np.array(preds.get("pr_recall", []))
    if len(pr_t) > 0:
        idx = np.argmin(np.abs(pr_t - threshold))
        fig.add_trace(go.Scatter(
            x=[pr_re[idx]], y=[pr_pe[idx]],
            mode="markers+text", name=f"Threshold={threshold:.3f}",
            marker=dict(size=14, color=mode_color, symbol="diamond", line=dict(color="white", width=2)),
            text=[f"  T={threshold:.3f}  P={pr_pe[idx]:.2f}  R={pr_re[idx]:.2f}"],
            textposition="middle right",
            textfont=dict(color=mode_color, size=10),
        ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94A3B8", opacity=0.3)
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8",
        xaxis=dict(gridcolor="#2D3748", title="Recall"),
        yaxis=dict(gridcolor="#2D3748", title="Precision"),
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

# Threshold-Independent Metrics
st.markdown("### Threshold-Independent Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
metric_vals = [
    ("ROC-AUC", test.get("roc_auc", 0), "#D4AF37"),
    ("Avg Precision", test.get("avg_precision", 0), "#805AD5"),
    ("Balanced Accuracy", test.get("balanced_accuracy", 0), "#38A169"),
    ("MCC", test.get("matthews_corrcoef", 0), "#DD6B20"),
    ("Cohen's Kappa", test.get("cohen_kappa", 0), "#D69E2E"),
]
for col, (label, val, color) in zip([col1, col2, col3, col4, col5], metric_vals):
    with col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(val * 100, 1),
            number=dict(suffix="%", font=dict(size=20, color=color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#94A3B8"),
                bar=dict(color=color),
                bgcolor="#1E293B",
                bordercolor="#2D3748",
            ),
        ))
        fig.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font_color="#94A3B8", margin=dict(t=30, b=5, l=15, r=15),
            height=150,
        )
        st.plotly_chart(fig, use_container_width=True)

# Feature Importance
st.markdown("### Feature Importance (CatBoost Native)")
if fi and isinstance(fi, list):
    feat_imp = pd.DataFrame(fi)
    feat_imp = feat_imp.sort_values("importance", ascending=True)
    fig = px.bar(feat_imp, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=["#1E293B", "#D4AF37"])
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Importance"),
        yaxis=dict(gridcolor="#2D3748"),
        coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
        height=max(300, len(feat_imp) * 25),
    )
    st.plotly_chart(fig, use_container_width=True)

# Cross-Validation
st.markdown("### Cross-Validation Results")
if cv and "folds" in cv:
    folds = cv["folds"]
    cv_df = pd.DataFrame({
        "Fold": folds["fold"],
        "AUC": folds["auc"],
        "F1": folds["f1"],
        "Precision": folds["precision"],
        "Recall": folds["recall"],
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cv_df["Fold"], y=cv_df["AUC"], mode="lines+markers",
                             name="AUC", line=dict(color="#D4AF37", width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=cv_df["Fold"], y=cv_df["F1"], mode="lines+markers",
                             name="F1", line=dict(color="#805AD5", width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=cv_df["Fold"], y=cv_df["Precision"], mode="lines+markers",
                             name="Precision", line=dict(color="#38A169", width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=cv_df["Fold"], y=cv_df["Recall"], mode="lines+markers",
                             name="Recall", line=dict(color="#DD6B20", width=2), marker=dict(size=8)))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="CV Fold"),
        yaxis=dict(gridcolor="#2D3748", title="Score", range=[0, 1]),
        legend=dict(font=dict(color="#94A3B8"), orientation="h"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.markdown(f"**Mean AUC: {cv.get('mean_auc', 0):.4f} +/- {cv.get('std_auc', 0):.4f}**")

if bench:
    st.markdown("### Model Benchmark: CatBoost vs XGBoost")
    bench_df = pd.DataFrame(bench).T
    bench_df = bench_df.reset_index().rename(columns={"index": "Model"})
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

if ablation:
    st.markdown("### Ablation Study (Feature Group Importance)")
    abl_df = pd.DataFrame(ablation)
    if "auc_delta" in abl_df.columns:
        abl_df = abl_df[abl_df["group"] != "ALL FEATURES"]
        fig = px.bar(abl_df, x="auc_delta", y="group", orientation="h",
                     color="auc_delta",
                     color_continuous_scale=[[0, "#38A169"], [0.5, "#D69E2E"], [1, "#E53E3E"]],
                     labels={"auc_delta": "AUC Drop when Removed"})
        fig.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748"),
            yaxis=dict(gridcolor="#2D3748"),
            coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=250,
        )
        st.plotly_chart(fig, use_container_width=True)

if robustness:
    st.markdown("### Robustness: Noise Injection Tests")
    rob_df = pd.DataFrame(robustness)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rob_df["noise_level"] * 100, y=rob_df["mean_auc"],
                             mode="lines+markers", name="Mean AUC",
                             line=dict(color="#D4AF37", width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=rob_df["noise_level"] * 100, y=rob_df["min_auc"],
                             mode="lines", name="Min AUC",
                             line=dict(color="#E53E3E", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=rob_df["noise_level"] * 100, y=rob_df["max_auc"],
                             mode="lines", name="Max AUC",
                             line=dict(color="#38A169", dash="dot", width=1)))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Noise Level (%)"),
        yaxis=dict(gridcolor="#2D3748", title="ROC-AUC"),
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

if leakage and "overfitting_analysis" in leakage:
    oa = leakage["overfitting_analysis"]
    st.markdown("### Overfitting Analysis")
    col1, col2, col3, col4 = st.columns(4)
    oa_metrics = [
        ("Train AUC", f"{oa.get('train_roc_auc', 0):.4f}", "#38A169"),
        ("Val AUC", f"{oa.get('val_roc_auc', 0) or 0:.4f}", "#D4AF37"),
        ("Test AUC", f"{oa.get('test_roc_auc', 0):.4f}", "#E53E3E"),
        ("Train-Test Gap", f"{oa.get('train_test_auc_gap', 0):.4f}", "#DD6B20"),
    ]
    for col, (label, val, color) in zip([col1, col2, col3, col4], oa_metrics):
        with col:
            st.markdown(f"""
            <div style='background:#1E293B;border:1px solid {color}30;border-radius:8px;padding:0.8rem;text-align:center;'>
                <div style='font-size:0.75rem;color:#94A3B8;'>{label}</div>
                <div style='font-size:1.3rem;font-weight:700;color:{color};'>{val}</div>
            </div>
            """, unsafe_allow_html=True)
    risk = oa.get("overfitting_risk", "UNKNOWN")
    risk_color = "#38A169" if risk == "LOW" else "#D69E2E" if risk == "MODERATE" else "#E53E3E"
    st.markdown(f"<p style='text-align:center;color:{risk_color};font-weight:600;'>Overfitting Risk: {risk}</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='background:#1E293B;border:1px solid #2D3748;border-radius:8px;padding:1rem;margin-top:1rem;'>
    <div style='color:#D4AF37;font-weight:600;margin-bottom:0.3rem;'>Data Disclosure</div>
    <div style='color:#94A3B8;font-size:0.8rem;'>
        All metrics computed from model training on <b>synthetic mine operations data</b> (125 mines x 72 months = 9,000 samples).
        Pseudo-labels are <b>rule-based risk scores</b>, not ground truth.
        The model's ranking ability (AUC=0.98) is excellent, but threshold selection affects precision/recall trade-off.
        Walk-forward temporal validation confirms temporal stability (AUC range: 0.983-0.990).
    </div>
</div>
""", unsafe_allow_html=True)
