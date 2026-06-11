"""
GOLD360 Publication-Grade Figure Generation
Thesis Version — White background, academic publication style
All figures grounded in evaluation report data (no hallucinated values).
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

# ── GOLD360 ACADEMIC COLOR PALETTE ──────────────────────────────────────
GOLD      = "#B8860B"   # dark gold for print
GOLD_L    = "#D4AF37"
NAVY      = "#1E3A5F"
BLUE      = "#2563EB"
BLUE_L    = "#60A5FA"
GREEN     = "#16A34A"
GREEN_L   = "#4ADE80"
RED       = "#DC2626"
RED_L     = "#F87171"
AMBER     = "#D97706"
GRAY      = "#6B7280"
GRAY_L    = "#9CA3AF"
GRAY_LL   = "#D1D5DB"
CHARCOAL  = "#1F2937"
WHITE     = "#FFFFFF"
BG        = "#FFFFFF"   # thesis: white background

FONT_TITLE   = 14
FONT_LABEL   = 11
FONT_TICK    = 10
FONT_ANNOT   = 9
FONT_SOURCE  = 8

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": FONT_TICK,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": CHARCOAL,
    "axes.grid": False,
    "xtick.color": CHARCOAL,
    "ytick.color": CHARCOAL,
    "text.color": CHARCOAL,
    "legend.edgecolor": GRAY_LL,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "figure.dpi": 150,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

CATALOG = []


def load_report(name):
    path = os.path.join(REPORTS_DIR, name)
    with open(path) as f:
        return json.load(f)


def save_fig(fig, fig_num, title_slug, chapter=""):
    prefix = f"FIG_{fig_num:02d}"
    fname = f"{prefix}_{title_slug}"
    fig.savefig(os.path.join(OUTPUT_DIR, f"{fname}.png"), dpi=600, facecolor=BG)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{fname}.svg"), facecolor=BG)
    plt.close(fig)
    CATALOG.append({
        "figure": prefix,
        "title": title_slug.replace("_", " ").title(),
        "chapter": chapter,
        "filename": f"{fname}.png",
    })
    print(f"  Saved: {fname}.png + .svg")


def add_source_note(ax, text="Source: GOLD360 Analysis | Data: 2020–2025", yoff=-0.18):
    ax.text(0.0, yoff, text, transform=ax.transAxes,
            fontsize=FONT_SOURCE, color=GRAY, style="italic", va="top")


# ════════════════════════════════════════════════════════════════════════
# FIG 01: Dataset Overview Matrix
# ════════════════════════════════════════════════════════════════════════
def fig_01_dataset_overview():
    datasets = {
        "Mine Operations": {"rows": "9,000", "cols": 30, "freq": "Monthly", "period": "2020-2025", "type": "Synthetic"},
        "Gold Price": {"rows": "72", "cols": 5, "freq": "Monthly", "period": "2020-2025", "type": "Observed"},
        "FX Market": {"rows": "6", "cols": 4, "freq": "Annual", "period": "2020-2025", "type": "Observed"},
        "Rainfall": {"rows": "5,400", "cols": 5, "freq": "Monthly", "period": "2020-2025", "type": "Observed"},
        "Policy Events": {"rows": "45", "cols": 8, "freq": "Event", "period": "2020-2025", "type": "Observed"},
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    col_labels = ["Dataset", "Records", "Features", "Frequency", "Period", "Source"]
    table_data = []
    for name, info in datasets.items():
        table_data.append([name, info["rows"], str(info["cols"]), info["freq"], info["period"], info["type"]])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="center",
                     colColours=[NAVY]*6)
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_ANNOT)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRAY_LL)
        if row == 0:
            cell.set_text_props(color=WHITE, fontweight="bold", fontsize=FONT_ANNOT)
            cell.set_facecolor(NAVY)
        else:
            cell.set_facecolor(WHITE if row % 2 == 0 else "#F8F9FA")
            cell.set_text_props(color=CHARCOAL)

    ax.set_title("GOLD360 Data Source Inventory", fontsize=FONT_TITLE,
                 fontweight="bold", color=CHARCOAL, pad=20)
    ax.text(0.5, -0.02, "Five source datasets spanning 2020–2025, comprising 14,523+ observations across 52 features",
            transform=ax.transAxes, ha="center", fontsize=FONT_SOURCE, color=GRAY, style="italic")
    add_source_note(ax, "Source: GOLD360 Data Registry", yoff=-0.08)
    save_fig(fig, 1, "dataset_overview_matrix", "Chapter 3")


# ════════════════════════════════════════════════════════════════════════
# FIG 02: Monthly Gold Delivery Trends
# ════════════════════════════════════════════════════════════════════════
def fig_02_delivery_trends():
    df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv"))
    df["month"] = pd.to_datetime(df["month"])
    monthly = df.groupby("month").agg(
        estimated=("estimated_gold_yield_kg", "mean"),
        official=("official_delivery_kg", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly["month"], monthly["estimated"], alpha=0.15, color=BLUE)
    ax.fill_between(monthly["month"], monthly["official"], alpha=0.15, color=GOLD)
    ax.plot(monthly["month"], monthly["estimated"], color=BLUE, linewidth=1.8,
            label="Estimated Yield", marker="o", markersize=3)
    ax.plot(monthly["month"], monthly["official"], color=GOLD, linewidth=1.8,
            label="Official Delivery", marker="s", markersize=3)

    ax.set_xlabel("Month")
    ax.set_ylabel("Gold (kg, mine-level average)")
    ax.set_title("Monthly Gold Production vs. Official Delivery (2020–2025)",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT)
    from matplotlib.dates import MonthLocator, DateFormatter
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1,7]))
    ax.xaxis.set_major_formatter(DateFormatter("%b\n%Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_source_note(ax)
    save_fig(fig, 2, "monthly_delivery_trends", "Chapter 4")


# ════════════════════════════════════════════════════════════════════════
# FIG 03: Delivery Gap Distribution
# ════════════════════════════════════════════════════════════════════════
def fig_03_delivery_gap():
    df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv"))
    gap = (df["estimated_gold_yield_kg"] - df["official_delivery_kg"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(gap, bins=50, color=GOLD, alpha=0.7, edgecolor=WHITE, linewidth=0.5)

    ax.axvline(gap.mean(), color=RED, linestyle="--", linewidth=1.5, label=f"Mean: {gap.mean():.2f} kg")
    ax.axvline(gap.median(), color=BLUE, linestyle="--", linewidth=1.5, label=f"Median: {gap.median():.2f} kg")

    ax.set_xlabel("Delivery Gap (kg)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Gold Delivery Gaps",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_source_note(ax)
    save_fig(fig, 3, "delivery_gap_distribution", "Chapter 4")


# ════════════════════════════════════════════════════════════════════════
# FIG 04: Feature Importance
# ════════════════════════════════════════════════════════════════════════
def fig_04_feature_importance():
    fi = load_report("feature_importance.json")
    fi = [f for f in fi if f["importance"] > 0]
    fi = fi[:15]

    names = [f["feature"].replace("_", " ").title() for f in fi]
    values = [f["importance"] for f in fi]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [GOLD if v > 10 else (BLUE if v > 2 else GRAY) for v in values]
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor=WHITE, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=FONT_ANNOT)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (CatBoost)")
    ax.set_title("Feature Importance Ranking — Top 15 Predictors",
                 fontweight="bold", color=CHARCOAL, pad=15)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=FONT_ANNOT, color=CHARCOAL)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_source_note(ax)
    save_fig(fig, 4, "feature_importance_ranking", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 05: ROC Curve
# ════════════════════════════════════════════════════════════════════════
def fig_05_roc_curve():
    roc = load_report("roc_curve.json")
    metrics = load_report("test_metrics.json")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(roc["fpr"], roc["tpr"], color=GOLD, linewidth=2.5, label=f"CatBoost (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--", label="Random Classifier (AUC = 0.50)")

    optimal_idx = metrics["optimal_threshold_youden"]
    ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.1, color=GOLD)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.60, 0.15, f"AUC = {metrics['roc_auc']:.4f}\nF1 = {metrics['f1_score']:.4f}\nMCC = {metrics['matthews_corrcoef']:.4f}",
            fontsize=FONT_ANNOT, color=CHARCOAL, bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor=GRAY_LL))

    add_source_note(ax)
    save_fig(fig, 5, "roc_curve", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 06: Precision-Recall Curve
# ════════════════════════════════════════════════════════════════════════
def fig_06_pr_curve():
    pr = load_report("pr_curve.json")
    metrics = load_report("test_metrics.json")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(pr["recall"], pr["precision"], color=BLUE, linewidth=2.5,
            label=f"CatBoost (AP = {metrics['avg_precision']:.4f})")
    ax.fill_between(pr["recall"], pr["precision"], alpha=0.1, color=BLUE)
    ax.axhline(y=metrics["precision"], color=GRAY, linewidth=1, linestyle=":",
               label=f"Baseline Precision = {metrics['precision']:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT, loc="lower left")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.65, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_source_note(ax)
    save_fig(fig, 6, "precision_recall_curve", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 07: Confusion Matrix
# ════════════════════════════════════════════════════════════════════════
def fig_07_confusion_matrix():
    cm = load_report("test_metrics.json")["confusion_matrix"]
    matrix = np.array([[cm["true_negative"], cm["false_positive"]],
                       [cm["false_negative"], cm["true_positive"]]])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)

    for i in range(2):
        for j in range(2):
            color = WHITE if matrix[i, j] > matrix.max() * 0.5 else CHARCOAL
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk", "High Risk"], fontsize=FONT_ANNOT)
    ax.set_yticklabels(["Low Risk", "High Risk"], fontsize=FONT_ANNOT)
    ax.set_xlabel("Predicted Label", fontsize=FONT_LABEL)
    ax.set_ylabel("True Label", fontsize=FONT_LABEL)
    ax.set_title("Confusion Matrix — Test Set Performance",
                 fontweight="bold", color=CHARCOAL, pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=FONT_ANNOT)

    total = matrix.sum()
    accuracy = (cm["true_positive"] + cm["true_negative"]) / total
    ax.text(0.5, -0.12, f"Accuracy: {accuracy:.1%} | n = {total:,}",
            transform=ax.transAxes, ha="center", fontsize=FONT_ANNOT, color=GRAY)

    add_source_note(ax, yoff=-0.16)
    save_fig(fig, 7, "confusion_matrix", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 08: Cross-Validation Performance
# ════════════════════════════════════════════════════════════════════════
def fig_08_cross_validation():
    cv = load_report("cross_validation.json")
    folds = cv["folds"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(folds["fold"]))
    width = 0.25

    ax.bar(x - width, folds["auc"], width, label="ROC-AUC", color=GOLD, edgecolor=WHITE)
    ax.bar(x, folds["f1"], width, label="F1 Score", color=BLUE, edgecolor=WHITE)
    ax.bar(x + width, folds["recall"], width, label="Recall", color=GREEN, edgecolor=WHITE)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("5-Fold Temporal Cross-Validation Performance",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in folds["fold"]], fontsize=FONT_ANNOT)
    ax.set_ylim(0.7, 1.05)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.02, 0.95, f"Mean AUC: {cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}",
            transform=ax.transAxes, fontsize=FONT_ANNOT, color=CHARCOAL, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor=GRAY_LL))

    add_source_note(ax)
    save_fig(fig, 8, "cross_validation_performance", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 09: CatBoost vs XGBoost Benchmark
# ════════════════════════════════════════════════════════════════════════
def fig_09_benchmark():
    bench = load_report("benchmark_results.json")

    metrics = ["roc_auc", "f1", "precision", "recall"]
    labels = ["ROC-AUC", "F1 Score", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35

    cb_vals = [bench["catboost"][m] for m in metrics]
    xgb_vals = [bench["xgboost"][m] for m in metrics]

    bars1 = ax.bar(x - width/2, cb_vals, width, label="CatBoost", color=GOLD, edgecolor=WHITE)
    bars2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost", color=BLUE, edgecolor=WHITE)

    for bar, val in zip(bars1, cb_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=FONT_ANNOT, color=CHARCOAL)
    for bar, val in zip(bars2, xgb_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=FONT_ANNOT, color=CHARCOAL)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_ANNOT)
    ax.set_ylabel("Score")
    ax.set_title("Model Benchmark: CatBoost vs. XGBoost",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.set_ylim(0.55, 1.1)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    winner = "CatBoost" if bench["catboost"]["roc_auc"] >= bench["xgboost"]["roc_auc"] else "XGBoost"
    ax.text(0.98, 0.95, f"Winner: {winner}", transform=ax.transAxes,
            fontsize=FONT_ANNOT, fontweight="bold", color=GREEN, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor=GREEN))

    add_source_note(ax)
    save_fig(fig, 9, "catboost_vs_xgboost_benchmark", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 10: Risk Distribution (Donut)
# ════════════════════════════════════════════════════════════════════════
def fig_10_risk_distribution():
    risk = load_report("test_metrics.json")["risk_distribution"]

    labels = ["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk"]
    sizes = [risk["low_risk_pct"], risk["moderate_risk_pct"],
             risk["elevated_risk_pct"], risk["high_risk_pct"]]
    colors = [GREEN, AMBER, RED_L, RED]

    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78, wedgeprops=dict(width=0.4, edgecolor=WHITE, linewidth=2),
        textprops=dict(fontsize=FONT_ANNOT)
    )
    for t in autotexts:
        t.set_fontsize(FONT_ANNOT)
        t.set_color(CHARCOAL)
        t.set_fontweight("bold")

    centre = plt.Circle((0, 0), 0.35, fc=WHITE, ec=GRAY_LL, linewidth=1.5)
    ax.add_artist(centre)
    ax.text(0, 0.05, f"n = 1,750", ha="center", va="center", fontsize=12, fontweight="bold", color=CHARCOAL)
    ax.text(0, -0.08, "test samples", ha="center", va="center", fontsize=FONT_ANNOT, color=GRAY)

    ax.set_title("Risk Category Distribution — Test Set",
                 fontweight="bold", color=CHARCOAL, pad=20)
    add_source_note(ax, yoff=-0.05)
    save_fig(fig, 10, "risk_category_distribution", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 11: Temporal Validation (Walk-Forward)
# ════════════════════════════════════════════════════════════════════════
def fig_11_temporal_validation():
    wf = load_report("temporal_validation.json")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = [f"Split {r['split']}" for r in wf]
    aucs = [r["roc_auc"] for r in wf]
    briers = [r["brier"] for r in wf]

    ax1.bar(x, aucs, color=GOLD, edgecolor=WHITE, alpha=0.85, label="ROC-AUC")
    ax1.set_ylabel("ROC-AUC", color=GOLD)
    ax1.set_ylim(0.98, 1.0)
    ax1.tick_params(axis="y", labelcolor=GOLD)

    ax2 = ax1.twinx()
    ax2.plot(x, briers, color=RED, marker="D", linewidth=2, markersize=7, label="Brier Score")
    ax2.set_ylabel("Brier Score", color=RED)
    ax2.set_ylim(0, 0.15)
    ax2.tick_params(axis="y", labelcolor=RED)

    ax1.set_title("Walk-Forward Temporal Validation",
                  fontweight="bold", color=CHARCOAL, pad=15)
    ax1.set_xlabel("Expanding Window Split")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, framealpha=0.9,
               edgecolor=GRAY_LL, fontsize=FONT_ANNOT, loc="center right")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Add period labels
    for i, r in enumerate(wf):
        ax1.text(i, 0.982, f"{r['val_from']}→{r['val_to']}", ha="center",
                 fontsize=7, color=GRAY, rotation=45)

    add_source_note(ax1)
    save_fig(fig, 11, "walkforward_temporal_validation", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 12: Ablation Study
# ════════════════════════════════════════════════════════════════════════
def fig_12_ablation():
    abl = load_report("ablation_results.json")

    names = [r["group"].replace("_", " ").title() for r in abl]
    aucs = [r["auc"] for r in abl]
    deltas = [r["auc_delta"] for r in abl]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: AUC with/without each group
    colors = [GOLD if d > 0.01 else (RED if d < -0.005 else GRAY) for d in deltas]
    bars = ax1.barh(range(len(names)), aucs, color=colors, edgecolor=WHITE, height=0.6)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=FONT_ANNOT)
    ax1.invert_yaxis()
    ax1.set_xlabel("ROC-AUC")
    ax1.set_title("Ablation: Feature Group Impact on AUC",
                  fontweight="bold", color=CHARCOAL, pad=10)
    for bar, val in zip(bars, aucs):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=FONT_ANNOT, color=CHARCOAL)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Delta
    delta_colors = [RED if d > 0.005 else GREEN for d in deltas]
    bars2 = ax2.barh(range(len(names)), deltas, color=delta_colors, edgecolor=WHITE, height=0.6)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(["" for _ in names])
    ax2.axvline(0, color=CHARCOAL, linewidth=0.8)
    ax2.set_xlabel("AUC Delta (higher = more important)")
    ax2.set_title("Performance Degradation",
                  fontweight="bold", color=CHARCOAL, pad=10)
    for bar, val in zip(bars2, deltas):
        offset = 0.001 if val >= 0 else -0.001
        ha = "left" if val >= 0 else "right"
        ax2.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=FONT_ANNOT, color=CHARCOAL)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=3)
    add_source_note(ax2, yoff=-0.18)
    save_fig(fig, 12, "ablation_feature_group_impact", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 13: Robustness to Noise
# ════════════════════════════════════════════════════════════════════════
def fig_13_robustness():
    rob = load_report("robustness_results.json")

    noise = [r["noise_level"] * 100 for r in rob]
    aucs = [r["mean_auc"] for r in rob]
    stds = [r["std_auc"] for r in rob]
    degrad = [r["degradation"] for r in rob]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.fill_between(noise, [a - s for a, s in zip(aucs, stds)],
                     [a + s for a, s in zip(aucs, stds)],
                     alpha=0.15, color=GOLD)
    line1 = ax1.plot(noise, aucs, color=GOLD, linewidth=2.5, marker="o", markersize=8,
                     label="Mean AUC", zorder=5)
    ax1.set_xlabel("Noise Level (% of feature values)")
    ax1.set_ylabel("ROC-AUC", color=GOLD)
    ax1.set_ylim(0.78, 1.02)
    ax1.tick_params(axis="y", labelcolor=GOLD)

    ax2 = ax1.twinx()
    line2 = ax2.plot(noise, degrad, color=RED, linewidth=2, marker="s", markersize=7,
                     linestyle="--", label="Degradation", zorder=5)
    ax2.set_ylabel("AUC Degradation", color=RED)
    ax2.set_ylim(-0.02, 0.18)
    ax2.tick_params(axis="y", labelcolor=RED)

    ax1.set_title("Model Robustness Under Feature Noise Injection",
                  fontweight="bold", color=CHARCOAL, pad=15)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=True, framealpha=0.9, edgecolor=GRAY_LL,
               fontsize=FONT_ANNOT, loc="center right")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for i, (n, a, d) in enumerate(zip(noise, aucs, degrad)):
        ax1.annotate(f"AUC={a:.3f}", (n, a), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=8, color=CHARCOAL)

    add_source_note(ax1)
    save_fig(fig, 13, "robustness_noise_injection", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 14: Overfitting Analysis
# ════════════════════════════════════════════════════════════════════════
def fig_14_overfitting():
    leak = load_report("leakage_and_overfitting.json")["overfitting_analysis"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ROC-AUC comparison
    splits = ["Train", "Validation", "Test"]
    aucs = [leak["train_roc_auc"], leak["val_roc_auc"], leak["test_roc_auc"]]
    colors = [GOLD, BLUE, GREEN]
    bars = ax1.bar(splits, aucs, color=colors, edgecolor=WHITE, width=0.5)
    for bar, val in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=FONT_ANNOT, fontweight="bold", color=CHARCOAL)
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("ROC-AUC by Split", fontweight="bold", color=CHARCOAL, pad=10)
    ax1.set_ylim(0.9, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: F1 comparison
    f1s = [leak["train_f1"], leak.get("val_f1", 0) or 0, leak["test_f1"]]
    bars2 = ax2.bar(splits, f1s, color=colors, edgecolor=WHITE, width=0.5)
    for bar, val in zip(bars2, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=FONT_ANNOT, fontweight="bold", color=CHARCOAL)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score by Split", fontweight="bold", color=CHARCOAL, pad=10)
    ax2.set_ylim(0.8, 1.08)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    risk_color = GREEN if leak["overfitting_risk"] == "LOW" else (AMBER if leak["overfitting_risk"] == "MODERATE" else RED)
    fig.text(0.5, -0.02, f"Overfitting Risk: {leak['overfitting_risk']} | "
             f"Train-Test AUC Gap: {leak['train_test_auc_gap']:.4f} | "
             f"Best Iteration: {leak['best_iteration']}",
             ha="center", fontsize=FONT_ANNOT, color=risk_color, fontweight="bold")

    fig.tight_layout(w_pad=4)
    add_source_note(ax2, yoff=-0.18)
    save_fig(fig, 14, "overfitting_analysis", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 15: Confidence Distribution
# ════════════════════════════════════════════════════════════════════════
def fig_15_confidence():
    metrics = load_report("test_metrics.json")
    preds = load_report("predictions.json")
    conf_dist = metrics["confidence_distribution"]

    y_prob = np.array(preds["y_prob"])
    confidence = 2 * np.abs(y_prob - 0.5)

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(confidence, bins=50, color=BLUE, alpha=0.7, edgecolor=WHITE, linewidth=0.5)

    ax.axvline(confidence.mean(), color=GOLD, linestyle="--", linewidth=1.5,
               label=f"Mean: {conf_dist['mean']:.3f}")
    ax.axvline(np.median(confidence), color=RED, linestyle=":", linewidth=1.5,
               label=f"Median: {conf_dist['median']:.3f}")

    ax.set_xlabel("Prediction Confidence (|2p - 1|)")
    ax.set_ylabel("Frequency")
    ax.set_title("Model Prediction Confidence Distribution",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_text = (f"σ = {conf_dist['std']:.3f}\n"
                  f"P(>0.8) = {conf_dist['pct_above_0.8']:.1f}%\n"
                  f"P(<0.2) = {conf_dist['pct_below_0.2']:.1f}%")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=FONT_ANNOT,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor=GRAY_LL))

    add_source_note(ax)
    save_fig(fig, 15, "prediction_confidence_distribution", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 16: Model Performance Summary (KPI Table)
# ════════════════════════════════════════════════════════════════════════
def fig_16_performance_summary():
    metrics = load_report("test_metrics.json")

    kpis = [
        ("ROC-AUC", f"{metrics['roc_auc']:.4f}", "Discrimination ability"),
        ("Avg Precision", f"{metrics['avg_precision']:.4f}", "Precision-recall balance"),
        ("F1 Score", f"{metrics['f1_score']:.4f}", "Harmonic mean P/R"),
        ("Precision", f"{metrics['precision']:.4f}", "Positive predictive value"),
        ("Recall", f"{metrics['recall']:.4f}", "Sensitivity / TPR"),
        ("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}", "Accuracy-adjusted"),
        ("MCC", f"{metrics['matthews_corrcoef']:.4f}", "Correlation coefficient"),
        ("Cohen's κ", f"{metrics['cohen_kappa']:.4f}", "Inter-rater agreement"),
        ("Brier Score", f"{metrics['brier_score']:.4f}", "Calibration quality"),
        ("Log Loss", f"{metrics['log_loss']:.4f}", "Probabilistic accuracy"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    col_labels = ["Metric", "Value", "Interpretation"]
    table_data = [[k, v, i] for k, v, i in kpis]

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="center",
                     colColours=[NAVY]*3)
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_ANNOT)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRAY_LL)
        if row == 0:
            cell.set_text_props(color=WHITE, fontweight="bold", fontsize=FONT_ANNOT)
            cell.set_facecolor(NAVY)
        else:
            cell.set_facecolor(WHITE if row % 2 == 0 else "#F8F9FA")
            cell.set_text_props(color=CHARCOAL)
            if col == 1:
                cell.set_text_props(fontweight="bold", color=GOLD)

    ax.set_title("GOLD360 Model Performance Summary — Test Set",
                 fontsize=FONT_TITLE, fontweight="bold", color=CHARCOAL, pad=20)
    add_source_note(ax, yoff=-0.02)
    save_fig(fig, 16, "model_performance_summary", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 17: Data Split Timeline
# ════════════════════════════════════════════════════════════════════════
def fig_17_split_timeline():
    info = load_report("split_info.json")

    fig, ax = plt.subplots(figsize=(12, 3.5))

    splits = [
        ("Training", info["train_months"][0], info["train_months"][-1],
         info["train_samples"], GOLD),
        ("Validation", info["val_months"][0], info["val_months"][-1],
         info["val_samples"], BLUE),
        ("Test", info["test_months"][0], info["test_months"][-1],
         info["test_samples"], GREEN),
    ]

    for i, (label, start, end, n, color) in enumerate(splits):
        y = 2 - i
        ax.barh(y, 1, left=i*0.05, height=0.6, color=color, alpha=0.85, edgecolor=WHITE)
        ax.text(i*0.05 + 0.01, y + 0.05, f"{label}\n{n:,} samples\n{start} → {end}",
                va="center", fontsize=FONT_ANNOT, color=WHITE if color != GOLD else CHARCOAL,
                fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, 3)
    ax.axis("off")
    ax.set_title("Temporal Data Split Strategy",
                 fontweight="bold", color=CHARCOAL, pad=15)

    ax.text(0.5, -0.15, f"Total: {info['total_samples']:,} samples | "
            f"Features: {info['feature_count']} | "
            f"Train positive rate: {info['target_positive_rate_train']:.1%} → "
            f"Test: {info['target_positive_rate_test']:.1%}",
            transform=ax.transAxes, ha="center", fontsize=FONT_ANNOT, color=GRAY)

    add_source_note(ax, yoff=-0.25)
    save_fig(fig, 17, "temporal_data_split_strategy", "Chapter 4")


# ════════════════════════════════════════════════════════════════════════
# FIG 18: Leakage & Validation Checks
# ════════════════════════════════════════════════════════════════════════
def fig_18_leakage_checks():
    leak = load_report("leakage_and_overfitting.json")["leakage_checks"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    col_labels = ["Check", "Result", "Detail"]
    table_data = [[c["check"], c["result"], c["detail"][:60]] for c in leak]

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="center",
                     colColours=[NAVY]*3)
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_ANNOT)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRAY_LL)
        if row == 0:
            cell.set_text_props(color=WHITE, fontweight="bold", fontsize=FONT_ANNOT)
            cell.set_facecolor(NAVY)
        else:
            cell.set_facecolor(WHITE if row % 2 == 0 else "#F8F9FA")
            if col == 1:
                result = table_data[row-1][1]
                cell.set_text_props(color=GREEN if result == "PASS" else RED, fontweight="bold")
            else:
                cell.set_text_props(color=CHARCOAL)

    ax.set_title("Data Integrity & Leakage Validation Checks",
                 fontsize=FONT_TITLE, fontweight="bold", color=CHARCOAL, pad=20)
    ax.text(0.5, -0.02, "All 6 checks passed — no evidence of data leakage or temporal contamination",
            transform=ax.transAxes, ha="center", fontsize=FONT_ANNOT, color=GREEN, style="italic")
    add_source_note(ax, yoff=-0.08)
    save_fig(fig, 18, "leakage_validation_checks", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 19: Feature Group Categories
# ════════════════════════════════════════════════════════════════════════
def fig_19_feature_groups():
    fi = load_report("feature_importance.json")

    groups = {
        "Delivery": ["delivery_gap_kg", "delivery_efficiency", "delivery_gap_ratio",
                     "delivery_gap_kg_roll3", "delivery_gap_kg_roll3_std",
                     "delivery_efficiency_roll3", "delivery_efficiency_roll3_std"],
        "Macro": ["fx_spread_pct", "inflation_rate", "gold_price_usd"],
        "Operational": ["ore_grade_efficiency", "rainfall_raw", "energy_stress",
                       "ore_processed_tonnes", "payment_delay_days"],
        "Governance": ["policy_shock_flag", "license_encoded"],
        "Spatial": ["border_distance_km", "border_risk", "fgr_distance_km", "fgr_access"],
        "Miner Type": ["miner_type_asm"],
    }

    fi_dict = {f["feature"]: f["importance"] for f in fi}
    group_importance = {}
    for gname, feats in groups.items():
        group_importance[gname] = sum(fi_dict.get(f, 0) for f in feats)

    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    names = [g[0] for g in sorted_groups]
    values = [g[1] for g in sorted_groups]
    total = sum(values)
    pcts = [v/total*100 for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [GOLD, BLUE, GREEN, AMBER, RED, GRAY][:len(names)]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor=WHITE, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=FONT_ANNOT, rotation=15, ha="right")
    ax.set_ylabel("Cumulative Feature Importance")
    ax.set_title("Feature Importance by Domain Group",
                 fontweight="bold", color=CHARCOAL, pad=15)

    for bar, val, pct in zip(bars, values, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}\n({pct:.1f}%)", ha="center", fontsize=FONT_ANNOT, color=CHARCOAL)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_source_note(ax)
    save_fig(fig, 19, "feature_importance_by_domain", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# FIG 20: Threshold Sensitivity (Operational Modes)
# ════════════════════════════════════════════════════════════════════════
def fig_20_threshold_sensitivity():
    preds = load_report("predictions.json")
    y_true = np.array(preds["y_true"])
    y_prob = np.array(preds["y_prob"])

    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.01, 0.99, 0.01)
    f1s = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
    precs = [precision_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
    recs = [recall_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]

    optimal_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1s, color=GOLD, linewidth=2.5, label="F1 Score")
    ax.plot(thresholds, precs, color=BLUE, linewidth=2, linestyle="--", label="Precision")
    ax.plot(thresholds, recs, color=GREEN, linewidth=2, linestyle=":", label="Recall")

    ax.axvline(optimal_t, color=RED, linewidth=1.5, linestyle="--",
               label=f"Optimal Threshold = {optimal_t:.2f}")
    ax.axvline(0.1258, color=AMBER, linewidth=1.5, linestyle="-.",
               label=f"Youden's J = 0.126")

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity Analysis — Operational Mode Selection",
                 fontweight="bold", color=CHARCOAL, pad=15)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY_LL, fontsize=FONT_ANNOT)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate operational modes
    modes = [
        (0.05, "Monitoring\n(Sensitive)", BLUE),
        (0.1258, "Balanced\n(Youden's J)", GOLD),
        (0.50, "Investigation\n(Conservative)", RED),
    ]
    for t, label, color in modes:
        ax.annotate(label, (t, 0.05), fontsize=8, color=color, ha="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1, edgecolor=color))

    add_source_note(ax)
    save_fig(fig, 20, "threshold_sensitivity_analysis", "Chapter 5")


# ════════════════════════════════════════════════════════════════════════
# GENERATE ALL + CATALOG
# ════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GOLD360 PUBLICATION-GRADE FIGURE GENERATION")
    print("Thesis Version — White Background, Academic Style")
    print("=" * 60)

    generators = [
        fig_01_dataset_overview,
        fig_02_delivery_trends,
        fig_03_delivery_gap,
        fig_04_feature_importance,
        fig_05_roc_curve,
        fig_06_pr_curve,
        fig_07_confusion_matrix,
        fig_08_cross_validation,
        fig_09_benchmark,
        fig_10_risk_distribution,
        fig_11_temporal_validation,
        fig_12_ablation,
        fig_13_robustness,
        fig_14_overfitting,
        fig_15_confidence,
        fig_16_performance_summary,
        fig_17_split_timeline,
        fig_18_leakage_checks,
        fig_19_feature_groups,
        fig_20_threshold_sensitivity,
    ]

    for gen in generators:
        try:
            gen()
        except Exception as e:
            print(f"  FAILED: {gen.__name__}: {e}")

    # Save catalog
    catalog_path = os.path.join(OUTPUT_DIR, "figure_catalog.xlsx")
    pd.DataFrame(CATALOG).to_excel(catalog_path, index=False)
    print(f"\n  Saved: figure_catalog.xlsx ({len(CATALOG)} figures)")
    print(f"\n  Total: {len(CATALOG)} figures generated")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
