# GOLD360 — Architecture Summary

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (15 CSVs)                       │
│  Mine Ops │ Gold Price │ FX Market │ Rainfall │ Policy Events       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING (17 features)                 │
│  Delivery(3) │ Macro(3) │ Operational(3) │ Governance(2) │ Spatial(4)│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  WEAK SUPERVISION │ │ ANOMALY DETECTION│ │ POLICY OVERLAY   │
│  7 Labeling Funcs │ │ IF + ECOD + LOF  │ │ Scenario Engine  │
│  Weighted Fusion  │ │ Consensus Score  │ │ Elasticities     │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FUSION LAYER                                   │
│              Intelligence Tensor (multi-signal fusion)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRIMARY CLASSIFIER                               │
│              CatBoost (ROC-AUC: 0.9817)                             │
│              Isotonic Calibration │ Class Weights [1.0, 2.0]        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  EXPLAINABILITY  │ │ POLICY INTEL     │ │ GEOSPATIAL       │
│  SHAP Values     │ │ Scenario Sim     │ │ Risk Maps        │
│  Feature Imp.    │ │ Sensitivity      │ │ Cluster Analysis │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                               │
│              10 Pages │ Dark Theme │ Gold Branding                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Map

| Module | Path | Responsibility |
|--------|------|----------------|
| `data` | `gold360/data/` | Load, validate, harmonize raw CSVs |
| `features` | `gold360/features/` | Compute 17 engineered features |
| `weak_supervision` | `gold360/weak_supervision/` | 7 labeling functions, fusion, audit |
| `anomaly` | `gold360/anomaly/` | IF, ECOD, LOF ensemble |
| `fusion` | `gold360/fusion/` | IntelligenceTensor multi-signal fusion |
| `models` | `gold360/models/` | CatBoost, XGBoost, training, prediction |
| `pipeline` | `gold360/pipeline/` | Orchestrator, train, evaluate, simulate |
| `policy` | `gold360/policy/` | ScenarioEngine, elasticities, overlays |
| `explainability` | `gold360/explainability/` | SHAP explainer, plots, reports |
| `geospatial` | `gold360/geospatial/` | Maps, clusters, corridors, distances |
| `validation` | `gold360/validation/` | Metrics, temporal, robustness, ablation |
| `evaluation` | `gold360/evaluation/` | Self-contained 14-step eval script |
| `dashboard` | `gold360/dashboard/` | Streamlit app (10 pages) |
| `utils` | `gold360/utils/` | Config, seeds, logging, I/O |

## Data Flow

1. **Ingestion:** 5 raw CSVs loaded and merged onto mine_ops base (9,000 rows)
2. **Feature Engineering:** 17 features computed (delivery, macro, operational, governance, spatial)
3. **Pseudo-Label Generation:** 6-component weighted risk score → binary target
4. **Temporal Split:** 60% train / 20% val / 20% test (chronological)
5. **Training:** CatBoost with early stopping, isotonic calibration
6. **Evaluation:** 14-step comprehensive assessment
7. **Deployment:** Dashboard reads JSON reports, displays interactive visualizations

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.14 |
| ML Framework | CatBoost 1.2.10, XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Geospatial | Folium, streamlit-folium |
| Experiment Tracking | MLflow (SQLite backend) |
| Data Processing | Pandas, NumPy, SciPy |
| Anomaly Detection | PyOD (Isolation Forest, ECOD, LOF) |

## Key Design Decisions

1. **Self-contained evaluation** — `run_full_evaluation.py` bypasses the orchestrator pipeline for reliability
2. **JSON reports as interface** — Dashboard reads pre-computed JSON, not live model
3. **V3 parallel directory** — V2 preserved, V3 is isolated copy
4. **Pseudo-labels, not ground truth** — No verified labels available; model learns from weighted risk signals
5. **Temporal split, not random** — Prevents data leakage from future to past
6. **Academic language enforcement** — "delivery shortfall risk" not "smuggling"
