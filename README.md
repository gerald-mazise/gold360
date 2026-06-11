# GOLD360 — Hybrid Economic Intelligence Platform

A weakly supervised anomaly detection and policy intelligence platform for Zimbabwe's gold ecosystem.

## Architecture

```
Data Sources → Data Engineering → Feature Engineering → Weak Supervision
    → Anomaly Detection Ensemble → Fusion Intelligence Layer
        → Primary Classifier → Explainability Layer → Policy Intelligence Layer
```

## Key Results (V3)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9817 |
| F1 Score | 0.9047 |
| Precision | 0.9916 |
| Recall | 0.8319 |
| Features | 17 active, 5 excluded |
| Model | CatBoost 1.2.10 (1000 iter, depth 6) |
| Calibration | Isotonic regression (3-fold CV) |

## Project Structure

```
gold360_opencode_v1/
├── gold360_v3/              # V3 platform (latest)
│   ├── gold360/             # Python package
│   │   ├── anomaly/         # Isolation Forest, ECOD, LOF
│   │   ├── features/        # 17 engineered features
│   │   ├── models/          # CatBoost classifier
│   │   ├── fusion/          # Ensemble fusion layer
│   │   ├── explainability/  # SHAP analysis
│   │   ├── policy/          # Scenario Intelligence Engine
│   │   ├── weak_supervision/# Pseudo-label generation
│   │   └── evaluation/      # Full evaluation pipeline
│   ├── config/              # YAML configuration
│   ├── reports/             # JSON evaluation reports
│   ├── data/raw/            # 17 source datasets
│   └── outputs/figures/     # 26 thesis figures
├── dashboard_handoff/       # Dashboard documentation package
│   ├── 17 markdown docs     # Architecture, features, UX
│   ├── screenshots/         # 10 page screenshots
│   └── data/                # 5 JSON data extracts
└── research_audits/         # Ablation studies
    ├── audit_01/            # ore_grade_efficiency removal
    ├── audit_02/            # All component removal
    ├── audit_03/            # Weight sensitivity
    └── COMPARISON_REPORT.md
```

## Setup

```bash
pip install -e . --no-deps
pip install catboost scikit-learn shap pandas numpy pyyaml mlflow
```

## Usage

```bash
# Run full evaluation
python -m gold360.evaluation.run_full_evaluation

# Launch Streamlit dashboard
python run_dashboard.py
# or
streamlit run app/main.py --server.port 8503
```

## Feature Groups

| Group | Features | Importance |
|-------|----------|------------|
| Delivery | delivery_gap_kg, delivery_efficiency | 26.4% |
| Ore Grade | ore_grade_efficiency | 14.8% |
| Economic | fx_spread_pct, gold_price_lag_3m | 12.1% |
| Geographic | border_risk, fgr_distance_km | 9.7% |
| Temporal | month, quarter, year | 8.3% |
| Policy | policy_shock_flag | 5.2% |
| Operational | recovery_rate, mine_type_asm | 4.9% |

## Leakage Controls

- 5 features excluded for target leakage (delivery_efficiency r=-0.5365)
- Fixed pseudo-label threshold (0.5) instead of percentile-based
- Temporal train/test split (2020-2023 train, 2024-2025 test)
- Probability calibration via isotonic regression

## Academic Positioning

All outputs are probabilistic intelligence signals, not forensic proof. Terms used:
- "delivery shortfall risk" (not smuggling)
- "leakage-risk residual" (not illicit volumes)
- "probabilistic anomaly signal" (not criminal activity)
- "structural divergence" (not evidence)

## License

Private — all rights reserved.
