# GOLD360 — Intelligence Platform

Hybrid Economic Intelligence + Weakly Supervised Anomaly Detection + Policy Intelligence Platform for Zimbabwe's Gold Ecosystem.

## Quick Start

```bash
# Install
pip install -e .

# Run full pipeline
python -c "from gold360.pipeline.orchestrator import PipelineOrchestrator; p = PipelineOrchestrator(); results = p.run_full(seed=42); print(results)"

# Launch dashboard
streamlit run gold360/dashboard/app.py
```

## Project Status

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Project Scaffolding | ✅ Complete |
| 2 | Data Engineering | ✅ Complete |
| 3 | Feature Engineering | ✅ Complete |
| 4 | Weak Supervision | ✅ Complete |
| 5 | Anomaly Detection | ✅ Complete |
| 6 | Fusion Layer | ✅ Complete |
| 7 | Primary Classifier | ✅ Complete |
| 8 | Explainability | ✅ Complete |
| 9 | Scenario Intelligence Engine | ✅ Complete |
| 10 | Geospatial Intelligence | ✅ Complete |
| 11 | Pipeline Orchestration | ✅ Complete |
| 12 | Validation Framework | ✅ Complete |
| 13 | Streamlit Dashboard | ✅ Complete |
| 14 | Testing | ✅ Complete |
| 15 | Documentation | ✅ Complete |

## Key Features

- **15 datasets** harmonized into unified intelligence table
- **31 features** across 6 intelligence domains
- **7 labeling functions** for probabilistic weak supervision
- **3-detector anomaly ensemble** with weighted consensus
- **CatBoost classifier** with temporal walk-forward validation
- **SHAP explainability** with approved terminology reports
- **Scenario Intelligence Engine** for policy micro-simulation
- **Geospatial analysis** with corridors, clusters, and interactive maps
- **Streamlit dashboard** with dark slate/gold theme (10 pages)

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full architectural documentation.

## Terminology

GOLD360 enforces approved academic terminology:

| Preferred Term | Avoid |
|----------------|-------|
| Delivery shortfall risk | Smuggling |
| Leakage-risk residual | Illegal flows |
| Probabilistic anomaly signal | Certain detection |
| Structural divergence | Systemic fraud |
| Delivery gap | Theft |

## License

MIT
