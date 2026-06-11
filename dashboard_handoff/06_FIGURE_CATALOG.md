# GOLD360 — Figure Catalog

## Summary

- **Total Figures:** 80 (SVG + PNG formats)
- **Location:** `gold360_v3/outputs/figures/`
- **Catalog:** `figure_catalog.xlsx`
- **Resolution:** 600 DPI
- **Style:** White background, gold/navy palette, academic formatting

## Pack A: Core Model Performance (FIG 01-20)

| # | Filename | Description |
|---|----------|-------------|
| 01 | `FIG_01_dataset_overview` | Dataset statistics and composition |
| 02 | `FIG_02_monthly_delivery_trends` | Monthly gold delivery trends over time |
| 03 | `FIG_03_delivery_gap_distribution` | Distribution of delivery gaps across mines |
| 04 | `FIG_04_feature_importance_ranking` | Top 17 features by CatBoost importance |
| 05 | `FIG_05_roc_curve` | ROC curve with AUC = 0.9817 |
| 06 | `FIG_06_precision_recall_curve` | Precision-recall curve |
| 07 | `FIG_07_confusion_matrix` | Confusion matrix heatmap (TN=468, FP=9, FN=214, TP=1059) |
| 08 | `FIG_08_cross_validation_performance` | 5-fold CV AUC results |
| 09 | `FIG_09_catboost_vs_xgboost` | Benchmark comparison |
| 10 | `FIG_10_risk_category_distribution` | Risk tier distribution donut chart |
| 11 | `FIG_11_walk_forward_validation` | Walk-forward temporal validation results |
| 12 | `FIG_12_ablation_study` | Feature group ablation impact |
| 13 | `FIG_13_robustness_noise_injection` | Robustness under noise injection |
| 14 | `FIG_14_overfitting_analysis` | Train/val/test performance comparison |
| 15 | `FIG_15_prediction_confidence` | Prediction confidence distribution |
| 16 | `FIG_16_model_performance_summary` | Combined metrics summary |
| 17 | `FIG_17_temporal_data_split` | Data split visualization |
| 18 | `FIG_18_leakage_validation` | Leakage check results |
| 19 | `FIG_19_feature_importance_by_domain` | Importance grouped by domain |
| 20 | `FIG_20_threshold_sensitivity` | Performance vs classification threshold |

## Pack B: Macro Intelligence (FIG 21-29)

| # | Filename | Description |
|---|----------|-------------|
| 21 | `FIG_21_gold_price_trend` | Gold price trend 2020-2025 |
| 22 | `FIG_22_gold_price_volatility` | Rolling volatility of gold price |
| 23 | `FIG_23_inflation_trend` | Zimbabwe inflation trend |
| 24 | `FIG_24_fx_distortion_trend` | FX market distortion over time |
| 25 | `FIG_25_inflation_vs_fx_distortion` | Correlation between inflation and FX |
| 26 | `FIG_26_macro_instability_index` | Composite macro instability |
| 27 | `FIG_27_gold_price_vs_deliveries` | Gold price vs delivery volumes |
| 28 | `FIG_28_inflation_vs_deliveries` | Inflation vs delivery volumes |
| 29 | `FIG_29_fx_distortion_vs_deliveries` | FX distortion vs deliveries |

## Pack C: Mining Operations (FIG 30-37)

| # | Filename | Description |
|---|----------|-------------|
| 30 | `FIG_30_production_by_province` | Provincial production distribution |
| 31 | `FIG_31_production_trend` | Monthly production trend |
| 32 | `FIG_32_recovery_rate_distribution` | Recovery rate distribution |
| 33 | `FIG_33_ore_grade_distribution` | Ore grade distribution |
| 34 | `FIG_34_mine_type_distribution` | ASM vs Formal mine counts |
| 35 | `FIG_35_delivery_efficiency` | Delivery efficiency by mine |
| 36 | `FIG_36_yield_by_province` | Gold yield by province |
| 37 | `FIG_37_operational_state` | Operational state classification |

## Pack D: Policy Intelligence (FIG 38-47)

| # | Filename | Description |
|---|----------|-------------|
| 38 | `FIG_38_policy_event_timeline` | Policy events over time |
| 39 | `FIG_39_policy_event_heatmap` | Policy events by type and month |
| 40 | `FIG_40_shock_matrix` | Policy shock impact matrix |
| 41 | `FIG_41_pre_post_comparison` | Before/after policy comparison |
| 42 | `FIG_42_impact_ranking` | Policy impact ranking |
| 43 | `FIG_43_policy_vs_deliveries` | Policy events vs delivery volumes |
| 44 | `FIG_44_policy_vs_risk` | Policy events vs risk scores |
| 45 | `FIG_45_policy_response_simulation` | Scenario simulation results |
| 46 | `FIG_46_scenario_comparison` | Multiple scenario comparison |
| 47 | `FIG_47_sensitivity_analysis` | Parameter sensitivity |

## Pack E: Anomaly & Risk (FIG 48-58)

| # | Filename | Description |
|---|----------|-------------|
| 48 | `FIG_48_pseudo_label_distribution` | Pseudo-label distribution |
| 49 | `FIG_49_anomaly_score_distribution` | Anomaly score distribution |
| 50 | `FIG_50_consensus_scores` | Ensemble consensus scoring |
| 51 | `FIG_51_risk_probability_distribution` | Risk probability distribution |
| 52 | `FIG_52_high_risk_mine_ranking` | Top high-risk mines |
| 53 | `FIG_53_anomaly_heatmap` | Anomaly scores by mine and month |
| 54 | `FIG_54_temporal_evolution` | Risk evolution over time |
| 55 | `FIG_55_anomaly_risk_clustering` | Anomaly vs risk scatter |
| 56 | `FIG_56_risk_transition_matrix` | Risk tier transitions |
| 57 | `FIG_57_risk_ranking` | Mine risk ranking |
| 58 | `FIG_58_risk_summary` | Risk overview |

## Pack F: Explainability / SHAP (FIG 59-66)

| # | Filename | Description |
|---|----------|-------------|
| 59 | `FIG_59_shap_summary_plot` | SHAP summary (beeswarm) |
| 60 | `FIG_60_shap_beeswarm` | SHAP beeswarm plot |
| 61 | `FIG_61_shap_waterfall_example` | SHAP waterfall for single prediction |
| 62 | `FIG_62_shap_force_plot` | SHAP force plot |
| 63 | `FIG_63_top_feature_contributions` | Feature contribution bar chart |
| 64 | `FIG_64_feature_interaction` | Feature interaction heatmap |
| 65 | `FIG_65_partial_dependence_1` | Partial dependence plot (feature 1) |
| 66 | `FIG_66_partial_dependence_2` | Partial dependence plot (feature 2) |

## Pack G: Geospatial (FIG 67-74)

| # | Filename | Description |
|---|----------|-------------|
| 67 | `FIG_67_province_risk_map` | Province-level risk choropleth |
| 68 | `FIG_68_mine_risk_heatmap` | Mine-level risk heatmap |
| 69 | `FIG_69_fgr_office_coverage` | FGR office coverage areas |
| 70 | `FIG_70_border_risk_map` | Border proximity risk |
| 71 | `FIG_71_corridor_risk_map` | Corridor risk analysis |
| 72 | `FIG_72_spatial_cluster_map` | DBSCAN spatial clusters |
| 73 | `FIG_73_risk_density_surface` | Risk density surface |
| 74 | `FIG_74_hotspot_analysis` | Hotspot analysis |

## Pack H: Forecast (FIG 75-80)

| # | Filename | Description |
|---|----------|-------------|
| 75 | `FIG_75_expected_vs_actual` | Expected vs actual deliveries |
| 76 | `FIG_76_residual_distribution` | Residual distribution |
| 77 | `FIG_77_residual_heatmap` | Residuals by mine and month |
| 78 | `FIG_78_residual_evolution` | Residual trends over time |
| 79 | `FIG_79_forecast_accuracy` | Forecast accuracy metrics |
| 80 | `FIG_80_forecast_uncertainty` | Forecast uncertainty bounds |
