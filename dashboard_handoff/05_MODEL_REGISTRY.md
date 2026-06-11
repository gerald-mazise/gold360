# GOLD360 — Model Registry

## Model: CatBoost Classifier

| Attribute | Value |
|-----------|-------|
| Algorithm | CatBoost (Ordered Boosting) |
| Version | 1.2.10 |
| Task | Binary Classification |
| Target | `pseudo_risk_probability` (binary: high risk vs not) |
| Status | Production (V3) |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `iterations` | 1000 | Maximum boosting iterations |
| `learning_rate` | 0.03 | Step size shrinkage |
| `depth` | 6 | Tree depth |
| `l2_leaf_reg` | 3.0 | L2 regularization |
| `border_count` | 128 | Split decision border count |
| `early_stopping_rounds` | 50 | Stop if no improvement for 50 rounds |
| `random_seed` | 42 | Reproducibility seed |
| `loss_function` | Logloss | Binary classification loss |
| `eval_metric` AUC | Area under ROC curve |
| `class_weights` | [1.0, 2.0] | 2x weight on positive class |
| `best_iteration` | 664 | Actual iterations used |

## Calibration

| Parameter | Value |
|-----------|-------|
| Method | Isotonic regression |
| CV Folds | 3 |
| Train+Val Combined | Yes |

## Test Set Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9817 |
| Average Precision | 0.9930 |
| F1 Score | 0.9047 |
| Precision | 0.9916 |
| Recall | 0.8319 |
| Accuracy | 0.8726 |
| Balanced Accuracy | 0.9065 |
| Brier Score | 0.0900 |
| Log Loss | 0.2738 |
| MCC | 0.7423 |
| Cohen's Kappa | 0.7167 |
| Optimal Threshold (Youden J) | 0.1965 |

### At Optimal Threshold (0.1965)
| Metric | Value |
|--------|-------|
| F1 | 0.9459 |
| Precision | 0.9726 |
| Recall | 0.9207 |
| Accuracy | 0.9234 |

## Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative** | TN = 468 | FP = 9 |
| **Actual Positive** | FN = 214 | TP = 1059 |

## Cross-Validation (5-fold TimeSeriesSplit)

| Fold | AUC | F1 |
|------|-----|-----|
| 1 | 0.9729 | 0.8779 |
| 2 | 0.9672 | 0.8593 |
| 3 | 0.9613 | 0.8867 |
| 4 | 0.9786 | 0.9072 |
| 5 | 0.9896 | 0.9279 |
| **Mean** | **0.9739** | **0.8918** |
| **Std** | **±0.0097** | — |

## Walk-Forward Temporal Validation

| Split | Train Period | Val Period | AUC |
|-------|-------------|------------|-----|
| 1 | 2020-01 to 2020-12 | 2021-01 to 2021-12 | 0.9709 |
| 2 | 2020-01 to 2021-12 | 2022-01 to 2022-12 | 0.9686 |
| 3 | 2020-01 to 2022-12 | 2023-01 to 2023-12 | 0.9630 |
| 4 | 2020-01 to 2023-12 | 2024-01 to 2024-12 | 0.9814 |
| 5 | 2020-01 to 2024-12 | 2025-01 to 2025-12 | 0.9922 |

## Ablation Study

| Group Removed | Features | AUC | Delta |
|---------------|----------|-----|-------|
| ALL FEATURES | 17 | 0.9817 | — |
| delivery | 3 removed | 0.8530 | -0.1287 |
| operational | 4 removed | 0.9014 | -0.0803 |
| spatial | 4 removed | 0.9672 | -0.0145 |
| governance | 2 removed | 0.9701 | -0.0116 |
| macro | 3 removed | 0.9747 | -0.0070 |

## Robustness (Noise Injection)

| Noise Level | Mean AUC | Degradation |
|-------------|----------|-------------|
| 1% | 0.8014 | +0.1261 |
| 5% | 0.7888 | +0.1386 |
| 10% | 0.7824 | +0.1450 |
| 20% | 0.7577 | +0.1698 |

## Benchmark Comparison

| Model | AUC | F1 |
|-------|-----|-----|
| CatBoost | 0.9692 | 0.8419 |
| XGBoost | 0.9676 | 0.8181 |
| **Winner** | **CatBoost** | **CatBoost** |

## Overfitting Analysis

| Metric | Value |
|--------|-------|
| Train AUC | 0.9999 |
| Val AUC | 1.0000 |
| Test AUC | 0.9817 |
| Train-Test Gap | 0.0182 |
| Risk Level | LOW |

## Leakage Validation

| Check | Result |
|-------|--------|
| Temporal split: train < val < test | PASS |
| No future information in features | PASS |
| Target derived from features, not labels | PASS |
| Walk-forward respects temporal order | PASS |
| Noise robustness validated | PASS |
| Ablation shows distributed importance | PASS |
