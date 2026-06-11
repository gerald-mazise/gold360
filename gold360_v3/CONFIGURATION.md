# GOLD360 — Configuration Reference

## Default Configuration (`config/default.yaml`)

```yaml
project:
  name: GOLD360
  version: 0.1.0
  seed: 42

logging:
  level: INFO
  format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
  file: logs/gold360.log

data:
  raw_dir: data/raw
  processed_dir: data/processed
  cache_dir: data/cache

models:
  catboost:
    iterations: 500
    depth: 4
    learning_rate: 0.05
    early_stopping_rounds: 30
    loss_function: Logloss
  xgboost:
    n_estimators: 500
    max_depth: 4
    learning_rate: 0.05
    early_stopping_rounds: 30

validation:
  temporal_split:
    train_fraction: 0.7
    val_fraction: 0.15
    test_fraction: 0.15
  walk_forward:
    n_splits: 3
    min_train_months: 12

anomaly:
  ensemble_weights:
    IsolationForest: 0.4
    ECOD: 0.3
    LOF: 0.3
  calibration: isotonic

features:
  store_dir: data/features
  store_format: parquet

dashboard:
  host: localhost
  port: 8501
  theme: dark
```

## Data Sources (`config/data_sources.yaml`)

```yaml
datasets:
  fgr_deliveries:
    path: data/raw/fgr_deliveries.csv
    type: economic
    frequency: quarterly
    required: true

  zimstat_production:
    path: data/raw/zimstat_production.csv
    type: economic
    frequency: quarterly
    required: true

  synthetic_mine_ops:
    path: data/raw/synthetic_mine_ops.csv
    type: operational
    frequency: monthly
    required: true
    synthetic: true

  gold_price:
    path: data/raw/gold_price.csv
    type: market
    frequency: monthly
    required: true

  fx_distortion:
    path: data/raw/fx_distortion.csv
    type: macro
    frequency: annual
    required: true

  inflation:
    path: data/raw/inflation.csv
    type: macro
    frequency: annual
    required: true

  cpi:
    path: data/raw/cpi.csv
    type: macro
    frequency: annual
    required: false

  rainfall:
    path: data/raw/rainfall.csv
    type: environmental
    frequency: monthly
    required: false

  energy:
    path: data/raw/energy.csv
    type: operational
    frequency: annual
    required: false

  policy_events:
    path: data/raw/policy_events.csv
    type: governance
    frequency: event
    required: true

  fgr_offices:
    path: data/raw/fgr_offices.csv
    type: spatial
    frequency: static
    required: false

  mirror_trade:
    path: data/raw/mirror_trade.csv
    type: trade
    frequency: annual
    required: true

  gold_exports:
    path: data/raw/gold_exports.csv
    type: trade
    frequency: annual
    required: false

  smuggling_incidents:
    path: data/raw/smuggling_incidents.csv
    type: enforcement
    frequency: event
    required: false
    sensitive: true
```

## Feature Registry (`config/features.yaml`)

See `gold360/features/registry.py` for the declarative feature registry with 6 groups, ~31 features.
