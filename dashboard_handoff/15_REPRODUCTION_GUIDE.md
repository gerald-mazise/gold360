# GOLD360 — Reproduction Guide

## Environment Setup

### Prerequisites
- Python 3.11+
- pip
- Git

### Installation

```bash
# Clone repository
git clone <repository-url>
cd gold360_opencode_v1

# Create virtual environment
python -m venv gold360_v3/.venv
gold360_v3/.venv/Scripts/activate  # Windows
# source gold360_v3/.venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r gold360_v3/requirements.lock
```

### Alternative: System Python (if venv install times out)

```bash
# Use system Python directly
python -m pip install streamlit catboost scikit-learn shap plotly folium
python -m pip install -r gold360_v3/requirements.lock --no-deps
```

## Data Requirements

### Input Data
- `data/raw/synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv`
- 9,000 rows × 30 columns
- Monthly mine-level operations data (2020-2025)
- Synthetic but realistic

### Data Format

| Column | Type | Description |
|--------|------|-------------|
| mine_id | string | Unique mine identifier |
| month | date | Month (YYYY-MM-01) |
| province | string | Province name |
| latitude | float | Mine latitude |
| longitude | float | Mine longitude |
| ore_processed_tonnes | float | Monthly ore processed |
| recovery_rate_pct | float | Recovery rate (%) |
| gold_produced_kg | float | Gold produced |
| delivery_to_fidelity_kg | float | Gold delivered to FGR |
| ... | ... | 26 more columns |

## Model Reproduction

### Step 1: Feature Engineering

```bash
cd gold360_v3
python -c "
from gold360.evaluation.run_full_evaluation import EngineerFeatures
import pandas as pd

raw = pd.read_csv('data/raw/synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv')
engineer = EngineerFeatures()
features, feature_cols = engineer.run(raw)
print(f'Features: {len(feature_cols)}')
print(f'Rows: {len(features)}')
"
```

### Step 2: Pseudo-Label Generation

```bash
python -c "
from gold360.evaluation.run_full_evaluation import generate_pseudo_labels
import pandas as pd

features = pd.read_pickle('features.pkl')  # From step 1
labels = generate_pseudo_labels(features)
print(f'Positive rate: {labels.mean():.2%}')
"
```

### Step 3: Model Training

```bash
python -c "
from gold360.evaluation.run_full_evaluation import run_evaluation
run_evaluation()  # Trains CatBoost, evaluates, saves reports
"
```

### Step 4: Dashboard Launch

```bash
python -m streamlit run dashboard/app.py --server.port 8503
```

## Key Configuration Files

| File | Purpose |
|------|---------|
| `config/default.yaml` | Model parameters, feature groups, weights |
| `config/features.yaml` | Feature registry with formulas |
| `requirements.lock` | Pinned dependency versions |

## Known Issues

1. **pip install -e . times out** — Heavy dependencies. Use `--no-deps` or system Python.
2. **MLflow requires SQLite** — File store deprecated in MLflow 3.x. Use `sqlite:///mlflow.db`.
3. **Pseudo-label threshold** — Must use fixed threshold (0.5), not percentile, to avoid data leakage.
4. **Feature leakage** — `delivery_efficiency` and `delivery_gap_ratio` are removed from V3. Do not re-add.

## Validation Checklist

- [ ] AUC ≥ 0.98
- [ ] F1 ≥ 0.90
- [ ] Brier ≤ 0.10
- [ ] No feature leakage (delivery_efficiency, delivery_gap_ratio excluded)
- [ ] MLflow tracking active
- [ ] All 12 reports generated
- [ ] Dashboard loads all pages
