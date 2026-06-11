# GOLD360 — Risk Outputs

## Risk Score Formula

The risk score is a **weighted combination of 6 signal components**, normalized to [0, 1]:

```
risk_score = clip(
    delivery_signal * 0.30 +
    fx_signal * 0.20 +
    border_signal * 0.15 +
    policy_signal * 0.15 +
    ore_grade_signal * 0.10 +
    asm_signal * 0.10,
    0, 1
)
```

### Components

| Component | Weight | Formula | Signal |
|-----------|--------|---------|--------|
| Delivery gap | 30% | `clip(gap_ratio * 3.0, 0, 1)` | Delivery shortfall ratio |
| FX spread | 20% | `clip(fx / fx_max, 0, 1)` | FX arbitrage pressure |
| Border risk | 15% | `border_risk` (sigmoid) | Border proximity |
| Policy shock | 15% | `policy_shock_flag` | Policy event indicator |
| Ore grade efficiency | 10% | `clip(\|ore_eff - median\| / std, 0, 1)` | Operational anomaly |
| ASM miner type | 10% | `miner_type_asm` | Artisanal mining flag |

## Risk Tiers

| Tier | Range | Color | Label | Interpretation |
|------|-------|-------|-------|----------------|
| Low | [0.0, 0.25] | Green (#38A169) | Low Risk | Normal operations, no intervention |
| Moderate | [0.25, 0.50] | Amber (#D69E2E) | Moderate Risk | Minor anomalies, monitor trends |
| Elevated | [0.50, 0.75] | Orange (#DD6B20) | Elevated Risk | Multiple signals, investigate |
| High | [0.75, 1.0] | Red (#E53E3E) | High Risk | Strong signal, priority investigation |

## Current Distribution (Test Set)

| Tier | Percentage | Count |
|------|------------|-------|
| Low | 33.31% | ~583 |
| Moderate | 5.66% | ~99 |
| Elevated | 5.77% | ~101 |
| High | 55.26% | ~967 |

## Model Prediction Distribution

| Metric | Value |
|--------|-------|
| Mean confidence | 0.844 |
| Median confidence | 0.9677 |
| Std confidence | 0.2426 |
| Predictions > 0.8 | 75.26% |
| Predictions > 0.6 | 85.03% |
| Predictions < 0.2 | 4.17% |

## Risk Score Interpretation

### For Analysts
- **Low (green):** Mine operates within normal parameters. No action needed.
- **Moderate (amber):** Minor deviations detected. Continue standard monitoring.
- **Elevated (orange):** Multiple risk signals present. Schedule investigation within 30 days.
- **High (red):** Strong risk signal. Priority investigation recommended within 7 days.

### Academic Language
| Risk Level | Preferred Term |
|------------|---------------|
| High | "elevated delivery shortfall risk" |
| Anomaly | "probabilistic anomaly signal" |
| Gap | "structural divergence between estimated and reported output" |
| Investigation | "enhanced monitoring protocol" |

## Binary Classification Threshold

| Threshold | Mode | Use Case |
|-----------|------|----------|
| 0.1965 | **Monitoring** (Youden's J) | Maximum recall, catches most anomalies |
| 0.50 | **Balanced** (default) | Standard classification |
| 0.80 | **Investigation** | High precision, only strongest signals |

The dashboard provides an interactive threshold slider for switching between these modes.

## Risk Tier Function (Detailed)

```python
def risk_tier(score, threshold=0.5):
    if score < threshold * 0.1:
        return "Clear"      # Green
    elif score < threshold:
        return "Monitoring"  # Amber
    elif score < 0.5:
        return "Elevated"    # Orange
    else:
        return "Critical"    # Red
```

## Related Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Risk scores | `predictions.json` | Per-sample risk probabilities |
| Risk distribution | `test_metrics.json` | Aggregate risk tier percentages |
| Risk heatmap | `FIG_53_anomaly_heatmap` | Visual risk by mine and month |
| Risk ranking | `FIG_52_high_risk_mine_ranking` | Top high-risk mines |
| Risk evolution | `FIG_54_temporal_evolution` | Risk trends over time |
