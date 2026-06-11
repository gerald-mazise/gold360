# GOLD360 — Executive Project Overview

## What is GOLD360?

GOLD360 is a **sovereign economic intelligence platform** for analyzing Zimbabwe's gold ecosystem. It combines:

- **Weakly supervised anomaly detection** — no ground truth labels needed
- **Ensemble ML classification** — CatBoost primary, XGBoost benchmark
- **Policy scenario simulation** — directional analysis of intervention impacts
- **Geospatial intelligence** — mine-level risk mapping with corridor analysis

**Academic positioning:** GOLD360 produces probabilistic intelligence signals, not forensic proof. All outputs are directional indicators for enhanced monitoring protocols.

## Current Status

| Attribute | Value |
|-----------|-------|
| **Version** | V3 (leakage-controlled) |
| **Dashboard** | Streamlit, 10 pages |
| **Model** | CatBoost (ROC-AUC 0.9817) |
| **Features** | 17 (5 excluded for leakage) |
| **Dataset** | 9,000 samples (125 mines × 72 months) |
| **Temporal Range** | 2020-01 to 2025-12 |
| **Dashboard Port** | 8503 (V3) |

## Key Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9817 |
| F1 Score | 0.9047 |
| Precision | 0.9916 |
| Recall | 0.8319 |
| MCC | 0.7423 |
| Brier Score | 0.0900 |
| CV AUC (5-fold) | 0.9739 ± 0.0097 |
| Overfitting Risk | LOW (gap: 0.0182) |

## Version History

| Version | Changes | AUC |
|---------|---------|-----|
| V1 | Initial prototype | — |
| V2 | Added MLflow, calibration, class weights | 0.9856 |
| **V3** | Removed 5 leaked features (delivery_efficiency, delivery_gap_ratio, etc.) | **0.9817** |

V3 is preferred for thesis validity despite the 0.39% AUC cost.

## Architecture (Simplified)

```
Data Sources → Feature Engineering → Weak Supervision → Anomaly Detection
     ↓                                                          ↓
  15 CSVs                                          Ensemble (IF + ECOD + LOF)
     ↓                                                          ↓
  Feature Store → Pseudo-Label Generation → Fusion Layer → CatBoost
                                                              ↓
                                                     Explainability (SHAP)
                                                              ↓
                                                     Policy Scenarios
                                                              ↓
                                                     Streamlit Dashboard
```

## Directory Structure

```
gold360_v3/
├── gold360/              # Python package
│   ├── data/             # Data loading, validation, harmonization
│   ├── features/         # Feature engineering (6 groups)
│   ├── weak_supervision/ # Labeling functions (7 LFs)
│   ├── anomaly/          # Isolation Forest, ECOD, LOF
│   ├── fusion/           # Multi-signal fusion layer
│   ├── models/           # CatBoost, XGBoost, training
│   ├── pipeline/         # Orchestrator, train, evaluate
│   ├── policy/           # Scenario engine, elasticities
│   ├── explainability/   # SHAP, plots, reports
│   ├── geospatial/       # Maps, clusters, corridors
│   ├── validation/       # Metrics, temporal, robustness
│   ├── evaluation/       # Self-contained eval script
│   ├── dashboard/        # Streamlit app
│   └── utils/            # Config, seeds, logging
├── config/               # YAML configs
├── data/raw/             # 15 CSV files
├── reports/              # 12 JSON report files
├── models/               # Saved models
├── outputs/figures/      # 80+ publication figures
└── mlflow.db             # MLflow tracking
```

## For the Next Developer

1. **Start with** `15_REPRODUCTION_GUIDE.md` to run the dashboard
2. **Read** `01_ARCHITECTURE.md` for system design
3. **Check** `05_MODEL_REGISTRY.md` for model details
4. **Review** `10_BRANDING_GUIDE.md` for visual identity
5. **See** `14_UX_IMPROVEMENTS.md` for recommended changes
