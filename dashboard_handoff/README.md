# GOLD360 — Dashboard Handoff Package

## Purpose

This package provides comprehensive documentation for recreating, improving, or extending the GOLD360 Intelligence Platform. It is designed for future AI agents or developers who need to understand the system without prior context.

## Contents

| File | Description |
|------|-------------|
| `00_EXECUTIVE_OVERVIEW.md` | System summary, key metrics, status |
| `01_DATA_SOURCES.md` | Data pipeline, 30 columns, 19 raw sources |
| `02_FEATURE_REGISTRY.md` | 22 engineered features with formulas |
| `03_LABELLING_SYSTEM.md` | Pseudo-label composition, 6 components |
| `04_MODEL_SPECIFICATION.md` | CatBoost parameters, calibration, weights |
| `05_SHAP_ANALYSIS.md` | Feature importance, direction, effects |
| `06_ANOMALY_SCORING.md` | Risk score computation, tiers, thresholds |
| `07_RISK_OUTPUTS.md` | Risk tiers, distribution, sensitivity |
| `08_VISUALIZATION_ENGINE.md` | Charts, maps, themes, styling |
| `09_GEOSPATIAL_OUTPUTS.md` | Maps, corridors, clusters, FGR offices |
| `10_BRANDING_GUIDE.md` | Colors, typography, CSS variables |
| `11_DASHBOARD_BLUEPRINT.md` | Layout structure, component inventory |
| `12_NAVIGATION_STRUCTURE.md` | 10 pages, icons, flow logic |
| `13_USER_PERSONAS.md` | 5 user personas with needs |
| `14_UX_IMPROVEMENTS.md` | 12 prioritized improvement suggestions |
| `15_REPRODUCTION_GUIDE.md` | Step-by-step reproduction instructions |
| `16_LEAKAGE_DISCLOSURE.md` | Feature-target overlap, audit results |

## Data Files

| File | Description |
|------|-------------|
| `data/feature_descriptions.json` | Structured feature registry (22 features) |
| `data/model_metrics.json` | All model performance metrics |
| `data/risk_distribution.json` | Risk tier data and formula |
| `data/shap_summary.json` | Feature importance + direction |
| `data/dashboard_config.json` | Theme, colors, layout config |

## Screenshots

| File | Page |
|------|------|
| `screenshots/page_00_overview.png` | Overview dashboard |
| `screenshots/page_01_data.png` | Data Pipeline |
| `screenshots/page_02_features.png` | Feature Engineering |
| `screenshots/page_03_weak_supervision.png` | Weak Supervision |
| `screenshots/page_04_anomaly.png` | Anomaly Detection |
| `screenshots/page_05_fusion.png` | Fusion Layer |
| `screenshots/page_06_model.png` | Model Performance |
| `screenshots/page_07_explainability.png` | Explainability |
| `screenshots/page_08_scenario.png` | Scenario Analysis |
| `screenshots/page_09_geospatial.png` | Geospatial |

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit catboost scikit-learn shap plotly folium

# 2. Launch dashboard
python -m streamlit run gold360_v3/dashboard/app.py --server.port 8503

# 3. Open browser
# http://localhost:8503
```

## Key Files in V3

| File | Purpose |
|------|---------|
| `gold360_v3/gold360/evaluation/run_full_evaluation.py` | Self-contained eval script |
| `gold360_v3/config/default.yaml` | Model config (version 0.3.0) |
| `gold360_v3/config/features.yaml` | Feature registry |
| `gold360_v3/dashboard/app.py` | Streamlit entry point |
| `gold360_v3/dashboard/pages/*.py` | 10 dashboard pages |
| `gold360_v3/dashboard/report_loader.py` | JSON report loader |
| `gold360_v3/reports/*.json` | 12 model reports |
| `gold360_v3/mlflow.db` | MLflow tracking DB |

## Model Performance (V3)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9817 |
| F1 | 0.9047 |
| Precision | 0.9916 |
| Recall | 0.8319 |
| Balanced Accuracy | 0.9065 |
| Brier Score | 0.0900 |
| MCC | 0.7423 |
| CV AUC (5-fold) | 0.9739 ± 0.0097 |
| Optimal Threshold | 0.1965 |
| Best Iteration | 664 |
| Feature Count | 17 |

## Architecture

```
Data Pipeline → Feature Engineering → Weak Supervision → Model Training → Fusion Layer → Dashboard
     ↓                    ↓                    ↓                  ↓              ↓            ↓
  19 sources      22 features         Pseudo-labels      CatBoost+XGBoost    Weighted    10 pages
  (synthetic)     (17 in V3)          (6 components)     (calibrated)        ensemble    (Streamlit)
```

## Known Limitations

1. **Synthetic data** — Not real Zimbabwe gold production data
2. **Pseudo-labels** — Not ground truth; probabilistic only
3. **Feature leakage** — Some features overlap with target definition (see `16_LEAKAGE_DISCLOSURE.md`)
4. **Static dashboard** — No real-time data refresh
5. **No authentication** — Open access (no login required)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| V1.0 | Initial | Base platform |
| V2.0 | +2 weeks | Bug fixes, calibration, UX, MLflow, figures |
| V3.0 | +4 weeks | Leakage control (5 features removed) |
