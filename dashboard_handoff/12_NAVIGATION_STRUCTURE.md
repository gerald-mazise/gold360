# GOLD360 — Navigation Structure

## Page Order

| Index | Page Name | Icon | Module |
|-------|-----------|------|--------|
| 0 | Overview | `:material/home:` | `00_overview.py` |
| 1 | Data Pipeline | `:material/database:` | `01_data_pipeline.py` |
| 2 | Feature Engineering | `:material/engineering:` | `02_feature_engineering.py` |
| 3 | Weak Supervision | `:material/label:` | `03_weak_supervision.py` |
| 4 | Anomaly Detection | `:material/warning:` | `04_anomaly_detection.py` |
| 5 | Fusion Layer | `:material/account_tree:` | `05_fusion_layer.py` |
| 6 | Model Performance | `:material/monitoring:` | `06_model_performance.py` |
| 7 | Explainability | `:material/search:` | `07_explainability.py` |
| 8 | Scenario Analysis | `:material/science:` | `08_scenario_analysis.py` |
| 9 | Geospatial | `:material/public:` | `09_geospatial.py` |

## Navigation Implementation

```python
# navbar.py
import streamlit as st

def get_navigation():
    pages = [
        st.Page("pages/00_overview.py", title="Overview", icon=":material/home:"),
        st.Page("pages/01_data_pipeline.py", title="Data Pipeline", icon=":material/database:"),
        st.Page("pages/02_feature_engineering.py", title="Feature Engineering", icon=":material/engineering:"),
        st.Page("pages/03_weak_supervision.py", title="Weak Supervision", icon=":material/label:"),
        st.Page("pages/04_anomaly_detection.py", title="Anomaly Detection", icon=":material/warning:"),
        st.Page("pages/05_fusion_layer.py", title="Fusion Layer", icon=":material/account_tree:"),
        st.Page("pages/06_model_performance.py", title="Model Performance", icon=":material/monitoring:"),
        st.Page("pages/07_explainability.py", title="Explainability", icon=":material/search:"),
        st.Page("pages/08_scenario_analysis.py", title="Scenario Analysis", icon=":material/science:"),
        st.Page("pages/09_geospatial.py", title="Geospatial", icon=":material/public:"),
    ]
    return st.navigation(pages, position="sidebar")
```

## Sidebar Layout

```
┌─────────────────┐
│                 │
│    GOLD360      │  ← Logo text, gold color
│  Intelligence   │  ← Subtitle, secondary text
│  Platform       │
│                 │
│ ─────────────── │
│                 │
│  🏠 Overview    │  ← Active: gold color
│  📂 Data Pipeline│
│  🔧 Feature Eng │
│  🏷️ Weak Super. │
│  ⚠️ Anomaly Det │
│  🌳 Fusion Layer│
│  📊 Model Perf  │
│  🔍 Explain.    │
│  🔬 Scenario    │
│  🌍 Geospatial  │
│                 │
└─────────────────┘
```

## Active State Styling

```css
/* Active nav item */
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    color: #D4AF37;
    font-weight: 600;
}

/* Nav item hover */
[data-testid="stSidebarNav"] a:hover {
    color: #D4AF37;
    background-color: rgba(212, 175, 55, 0.1);
}
```

## Page Flow Logic

The navigation follows the **data pipeline flow**:

1. **Overview** — Executive summary
2. **Data Pipeline** — What data we have
3. **Feature Engineering** — How we transform data
4. **Weak Supervision** — How we create labels
5. **Anomaly Detection** — How we find anomalies
6. **Fusion Layer** — How we combine signals
7. **Model Performance** — How well the model works
8. **Explainability** — Why the model makes predictions
9. **Scenario Analysis** — What-if policy testing
10. **Geospatial** — Where risks are located

## Icon Reference

All icons use Streamlit's Material Icons syntax:
- `:material/home:` — Home/Overview
- `:material/database:` — Data
- `:material/engineering:` — Features
- `:material/label:` — Labels
- `:material/warning:` — Anomaly
- `:material/account_tree:` — Fusion
- `:material/monitoring:` — Performance
- `:material/search:` — Explainability
- `:material/science:` — Scenarios
- `:material/public:` — Geospatial
