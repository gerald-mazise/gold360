# GOLD360 — Feature Registry

## Overview

31 features across 6 intelligence domains, managed by the declarative `FeatureRegistry` with versioned Parquet persistence.

## Feature Groups

### Delivery Features (5)

| Feature | Type | Description |
|---------|------|-------------|
| `delivery_gap_kg` | Continuous | Absolute gap between expected and actual deliveries (kg) |
| `delivery_efficiency` | Percentage | Ratio of actual to expected deliveries |
| `delivery_volatility` | Continuous | Rolling standard deviation of delivery efficiency |
| `delivery_trend` | Continuous | Directional momentum in delivery volumes |
| `delivery_seasonality` | Continuous | Calendar-based seasonal component |

### Macro Features (6)

| Feature | Type | Description |
|---------|------|-------------|
| `fx_spread_pct` | Percentage | Premium of parallel market over official rate |
| `macro_instability_index` | Composite | Weighted combination of inflation, FX, GDP indicators |
| `gold_price_momentum` | Continuous | 3-month gold price rate of change |
| `inflation_pressure` | Continuous | Inflation rate with exponential decay weighting |
| `real_fx_rate` | Index | Inflation-adjusted exchange rate index |
| `economic_activity_proxy` | Composite | Composite economic activity estimate |

### Operational Features (6)

| Feature | Type | Description |
|---------|------|-------------|
| `ore_grade_efficiency` | Ratio | Recovery rate / ore grade |
| `rainfall_disruption` | Z-score | Deviation from historical rainfall |
| `energy_stress` | Index | Composite energy availability index |
| `operational_composite` | Composite | Weighted combination of operational factors |
| `cost_pressure_index` | Continuous | Input cost pressure estimate |
| `labor_disruption_proxy` | Binary | Strike and labor action indicator |

### Governance Features (6)

| Feature | Type | Description |
|---------|------|-------------|
| `policy_shock_flag` | Binary | Major policy event indicator |
| `compliance_pressure` | Index | Composite regulatory enforcement index |
| `policy_volatility` | Count | Rolling count of policy events |
| `regulatory_sentiment` | Continuous | Regulatory tone indicator |
| `enforcement_intensity` | Continuous | Estimated enforcement resource allocation |
| `governance_composite` | Composite | Combined governance risk score |

### Spatial Features (5)

| Feature | Type | Description |
|---------|------|-------------|
| `border_proximity_km` | Distance | Kilometers to nearest international border post |
| `fgr_proximity_km` | Distance | Kilometers to nearest FGR purchase office |
| `corridor_risk_index` | Index | Weighted corridor risk based on border route |
| `province_risk` | Baseline | Province-level historical risk baseline |
| `spatial_isolation` | Composite | Remoteness and accessibility composite |

### Trade Features (3)

| Feature | Type | Description |
|---------|------|-------------|
| `mirror_trade_asymmetry` | Percentage | Discrepancy in trade mirror data |
| `export_declaration_gap` | Percentage | Gap between declared exports and partner imports |
| `trade_flow_anomaly` | Z-score | Deviation from historical trade flow patterns |

## Usage

```python
from gold360.features.registry import FeatureRegistry
from gold360.features.delivery import DeliveryFeatures

registry = FeatureRegistry()
DeliveryFeatures(registry)

# List features in a group
registry.list_features(group="delivery")

# Compute a specific feature
registry.compute("delivery_gap_kg", unified_df)

# Describe a feature
registry.describe("delivery_gap_kg")
```

## Storage

Features are persisted via `FeatureStore` in Parquet format with versioning:

```python
store = FeatureStore()
store.save(feature_df, version="v001")
loaded = store.load(version="v001")
```
