# GOLD360 — SHAP Outputs

## Method

CatBoost native feature importance (PredictionValuesChange). Full SHAP TreeExplainer values available in figure files (FIG 59-62).

## Feature Importance with Direction

| Rank | Feature | Importance | Direction | Domain | Explanation |
|------|---------|------------|-----------|--------|-------------|
| 1 | delivery_gap_kg | 26.42% | Positive | Delivery | Higher delivery gaps → higher risk |
| 2 | ore_grade_efficiency | 19.30% | Nonlinear | Operational | Extreme values (high/low) → higher risk |
| 3 | ore_processed_tonnes | 16.87% | Negative | Operational | Higher volumes → lower risk |
| 4 | miner_type_asm | 9.66% | Positive | Governance | ASM mines → higher risk |
| 5 | fx_spread_pct | 8.86% | Positive | Macro | Wider spreads → higher risk |
| 6 | inflation_rate | 3.16% | Positive | Macro | Higher inflation → higher risk |
| 7 | license_encoded | 2.44% | Positive | Governance | Informal/Pending → higher risk |
| 8 | border_distance_km | 2.07% | Negative | Spatial | Closer to border → higher risk |
| 9 | border_risk | 2.00% | Positive | Spatial | Higher border risk → higher risk |
| 10 | policy_shock_flag | 1.57% | Positive | Governance | Policy events → higher risk |
| 11 | gold_price_usd | 1.43% | Nonlinear | Macro | Extreme prices → higher risk |
| 12 | delivery_gap_kg_roll3 | 1.32% | Positive | Delivery | Sustained gaps → higher risk |
| 13 | payment_delay_days | 1.09% | Positive | Operational | Longer delays → higher risk |
| 14 | delivery_gap_kg_roll3_std | 1.08% | Positive | Delivery | Volatile gaps → higher risk |
| 15 | fgr_access | 0.94% | Negative | Spatial | Better access → lower risk |
| 16 | rainfall_raw | 0.93% | Nonlinear | Operational | Extreme rainfall → higher risk |
| 17 | fgr_distance_km | 0.84% | Positive | Spatial | Farther from FGR → higher risk |

## Direction Summary

| Direction | Count | Total Importance | Features |
|-----------|-------|------------------|----------|
| Positive | 11 | 73.61% | delivery_gap_kg, miner_type_asm, fx_spread_pct, inflation_rate, license_encoded, border_risk, policy_shock_flag, delivery_gap_kg_roll3, payment_delay_days, delivery_gap_kg_progress3_std, fgr_distance_km |
| Negative | 4 | 20.72% | ore_processed_tonnes, border_distance_km, fgr_access, rainfall_raw |
| Nonlinear | 3 | 21.66% | ore_grade_efficiency, gold_price_usd, rainfall_raw |

## Ablation Impact

| Group | AUC Drop | Interpretation |
|-------|----------|----------------|
| delivery | -0.1287 | Most critical — delivery signals are primary risk indicators |
| operational | -0.0803 | Secondary — ore grade, rainfall, payment data |
| spatial | -0.0145 | Minor — border/FGR proximity |
| governance | -0.0116 | Minor — policy and license data |
| macro | -0.0070 | Minimal — FX, inflation, gold price |

## Approved Academic Language

| Concept | Preferred Term | Avoid |
|---------|---------------|-------|
| High risk score | "elevated delivery shortfall risk" | "smuggling" |
| Anomaly detection | "probabilistic anomaly signal" | "certain detection" |
| Delivery gap | "structural divergence" | "theft" |
| Investigation | "enhanced monitoring protocol" | "criminal investigation" |
| Risk classification | "risk-tier classification" | "guilty/innocent" |

## Regenerating SHAP Values

```python
from gold360.explainability.shap_explainer import SHAPExplainer
from gold360.explainability.plots import ExplanationPlots

explainer = SHAPExplainer(model, X_train, feature_cols)
shap_values = explainer.explain(X_test)
ExplanationPlots.summary(shap_values, feature_cols)
ExplanationPlots.waterfall(shap_values[0], feature_cols)
```

## Figure Files

| Figure | File |
|--------|------|
| SHAP Summary | `FIG_59_shap_summary_plot.svg/png` |
| SHAP Beeswarm | `FIG_60_shap_beeswarm.svg/png` |
| SHAP Waterfall | `FIG_61_shap_waterfall_example.svg/png` |
| SHAP Force | `FIG_62_shap_force_plot.svg/png` |
| Feature Contributions | `FIG_63_top_feature_contributions.svg/png` |
