# GOLD360 — Dashboard Requirements

## Overview

The GOLD360 dashboard is a **Streamlit** application with 10 pages, dark theme, and gold branding. It reads pre-computed JSON reports (not live models) and displays interactive visualizations.

## Page Requirements

### Page 0: Overview
- **Purpose:** Executive summary with KPIs
- **Data Sources:** test_metrics.json, split_info.json, feature_importance.json, leakage_and_overfitting.json, cross_validation.json
- **Components:**
  - Hero header: "Zimbabwe Gold Ecosystem — Economic Intelligence & Structural Risk Assessment"
  - 6 KPI metric cards (AUC, F1, Precision, Recall, MCC, Brier)
  - Risk distribution donut chart
  - Feature importance horizontal bar chart (top 9)
  - Temporal validation line chart
  - Model comparison table
- **Interactive:** None (read-only summary)

### Page 1: Data Pipeline
- **Purpose:** Display data sources and pipeline status
- **Data Sources:** Inline catalog (14 datasets)
- **Components:**
  - Dataset catalog cards (14 data sources)
  - Pipeline status indicators
  - Data quality metrics
- **Interactive:** Expandable dataset details

### Page 2: Feature Engineering
- **Purpose:** Show feature groups and definitions
- **Data Sources:** Inline feature registry
- **Components:**
  - 6 feature group cards (Delivery, Macro, Operational, Governance, Spatial, Trade)
  - Feature count per group
  - Feature list per group
- **Interactive:** Expandable feature details

### Page 3: Weak Supervision
- **Purpose:** Display labeling functions and pseudo-label generation
- **Data Sources:** Inline LF catalog
- **Components:**
  - 7 labeling function cards with signal strength bars
  - Pseudo-label distribution chart
  - Coverage metrics
- **Interactive:** LF detail expanders

### Page 4: Anomaly Detection
- **Purpose:** Show anomaly ensemble configuration
- **Data Sources:** Inline detector specs
- **Components:**
  - 3 detector cards (IsolationForest, ECOD, LOF)
  - Consensus scoring explanation
  - Anomaly distribution chart
- **Interactive:** Detector weight adjuster (display only)

### Page 5: Fusion Layer
- **Purpose:** Explain multi-signal fusion
- **Data Sources:** Inline signal catalog
- **Components:**
  - Signal source cards (Features, Pseudo-Labels, Anomaly Scores, Policy)
  - Fusion diagram
  - Intelligence tensor explanation
- **Interactive:** None

### Page 6: Model Performance
- **Purpose:** Comprehensive model evaluation
- **Data Sources:** All 12 JSON reports via report_loader.py
- **Components:**
  - Metric cards (14 metrics)
  - ROC curve (Plotly)
  - Precision-Recall curve
  - Confusion matrix heatmap
  - Cross-validation fold results
  - Ablation study bar chart
  - Robustness noise injection chart
  - Leakage validation checklist
  - **Threshold slider** (Monitoring/Balanced/Investigation modes)
- **Interactive:** Threshold slider updates all metrics dynamically

### Page 7: Explainability
- **Purpose:** Feature importance and model interpretation
- **Data Sources:** feature_importance.json, ablation_results.json
- **Components:**
  - Feature importance bar chart with domain coloring
  - Feature contribution table (importance %)
  - Ablation impact chart
  - Domain description cards
- **Interactive:** Domain filter

### Page 8: Scenario Analysis
- **Purpose:** Policy intervention simulation
- **Data Sources:** Raw data + YAML config + ScenarioEngine
- **Components:**
  - 5 preset scenario cards
  - What-If Builder (4 sliders: retention, fx_retention, tax_rate, compliance)
  - Preset quick-apply buttons
  - Comparison table
  - Elasticity sensitivity chart
- **Interactive:** Sliders, preset buttons, rerun-based state

### Page 9: Geospatial
- **Purpose:** Spatial risk visualization
- **Data Sources:** Raw data + Folium maps
- **Components:**
  - Interactive Folium map (dark tiles)
  - Mine risk circle markers (color-coded)
  - Province-level risk aggregation
  - Border post markers
  - FGR office markers
- **Interactive:** Map zoom, marker click

## Component Library

### metric_card(label, value, delta, help_text, color)
Renders a styled HTML card with:
- Label (small, secondary text)
- Value (large, primary color)
- Delta (green for +, red for -)
- Tooltip (help_text)

### risk_indicator(score, label)
Renders a colored badge:
- score >= 0.75 → Red badge
- score >= 0.50 → Orange badge
- score >= 0.25 → Amber badge
- score < 0.25 → Green badge

### risk_bar(score, height, width)
Renders a horizontal progress bar with risk-colored fill.

## Responsive Behavior

- **Mobile (< 640px):** Single column, stacked cards
- **Tablet (640-1024px):** 2-column grid
- **Desktop (> 1024px):** Full wide layout, sidebar navigation

## Data Loading

All pages use `report_loader.py` functions:
```python
from gold360.dashboard.report_loader import (
    get_test_metrics, get_feature_importance, get_cross_validation,
    get_benchmark, get_temporal_validation, get_ablation,
    get_robustness, get_leakage, get_split_info,
    get_roc_curve, get_pr_curve, get_predictions, load_report
)
```

Reports are loaded from `gold360_v3/reports/` directory.
