# GOLD360 Architecture

## Overview

GOLD360 is a research-grade hybrid economic intelligence + weakly supervised anomaly detection + policy intelligence platform for Zimbabwe's gold ecosystem. It combines 15 datasets, 31 engineered features, 7 labeling functions, 3 anomaly detectors, and a CatBoost classifier into a unified intelligence pipeline.

## Architecture (Frozen)

```
Data Sources ──► Data Engineering ──► Feature Engineering ──► Weak Supervision ──►
    Anomaly Detection Ensemble ──► Fusion Intelligence Layer ──► Primary Classifier ──►
        Explainability Layer ──► Policy Intelligence Layer ──► Streamlit Intelligence Platform
```

## Directory Structure

```
gold360/
├── gold360/                  # Main package
│   ├── data/                 # Data engineering (loader, validator, temporal, harmonizer, registry)
│   ├── features/             # Feature engineering (6 groups, 31 features, store)
│   ├── weak_supervision/     # Labeling functions, fusion, validation, audit
│   ├── anomaly/              # IsolationForest, ECOD, LOF, ensemble, calibration
│   ├── fusion/               # Fusion layer, intelligence tensor
│   ├── models/               # CatBoost, trainer, predictor, benchmark
│   ├── explainability/       # SHAP explainer, plots, reports
│   ├── policy/               # Elasticities, scenario engine, definitions, overlay
│   ├── geospatial/           # Distances, corridors, clusters, maps
│   ├── pipeline/             # Orchestrator, train, evaluate, simulation pipelines
│   ├── validation/           # Ablation, robustness, temporal validation, metrics
│   ├── dashboard/            # Streamlit dashboard (10 pages + components)
│   │   ├── pages/            # 10 dashboard pages
│   │   └── components/       # Metric cards, risk indicators, navbar
│   └── utils/                # Config, logging, seeds
├── tests/                    # Test suite (8+ test modules)
├── config/                   # YAML configuration files
├── data/                     # Data directory
└── models/                   # Trained model artifacts
```

## Key Design Decisions

1. **No neural networks** — unless they demonstrably outperform CatBoost
2. **Synthetic data visually distinguished** — never drives final conclusions
3. **Pseudo-labels are probabilistic** — not ground truth, fully traceable
4. **Temporal integrity** — no future leakage, walk-forward validation
5. **Approved terminology** — all outputs enforce "delivery shortfall risk" not "smuggling"
6. **Research-grade** — dashboard feels like sovereign economic intelligence, not generic BI

## Modules

### Data Engineering
- 15 datasets: FGR deliveries, ZIMSTAT production, synthetic mine ops, gold price, FX, inflation, CPI, rainfall, energy, policy events, FGR offices, mirror trade, exports, smuggling
- Pydantic v2 schemas, unified DataLoader, DataValidator, TemporalAligner, DataHarmonizer, DataLineage

### Feature Engineering
- 31 features across 6 domains: delivery, macro, operational, governance, spatial, trade
- Declarative FeatureRegistry with compute graph, versioned FeatureStore (Parquet)

### Weak Supervision
- 7 labeling functions: extreme_delivery_collapse, fx_arbitrage_stress, impossible_yield_contradiction, corridor_inconsistency, inventory_anomaly, policy_contradiction, operational_mismatch
- Fusion: majority_vote, weighted, mean methods
- Full audit trail per pseudo-label

### Anomaly Detection
- 3-detector ensemble: IsolationForest (0.40), ECOD (0.30), LOF (0.30)
- Weighted consensus with agreement scoring (Full/Majority/Single)
- Isotonic/logistic calibration

### Fusion & Classifier
- FusionLayer combines features + pseudo-labels + anomaly + policy signals
- CatBoost with ordered boosting, early stopping, temporal CV
- Risk categories: low (0.00-0.25), moderate (0.25-0.50), elevated (0.50-0.75), high (0.75-1.00)

### Explainability
- SHAP: global importance, local waterfall, summary beeswarm
- Publication-quality matplotlib figures
- Natural language reports with approved terminology

### Policy Intelligence
- Scenario Intelligence Engine: directional analysis only
- 5 default scenarios: FX Reform, Regulatory Tightening, Energy Stabilization, Combined, Baseline
- Elasticities estimated from historical response

### Geospatial
- Haversine distances, nearest FGR/border post
- Weighted corridor risk (Beitbridge, Plumtree, Mutare, Chirundu, Victoria Falls)
- DBSCAN spatial clustering
- Folium interactive maps with dark theme

## Academic Positioning

GOLD360 does not claim smuggling, criminal activity, illicit volumes, or legal conclusions. All outputs are probabilistic intelligence signals. Approved language is enforced programmatically.
