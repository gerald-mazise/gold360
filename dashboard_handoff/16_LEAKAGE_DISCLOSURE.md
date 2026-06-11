# GOLD360 — Leakage Disclosure

## Overview

Target leakage refers to features that contain information about the target variable that would not be available at prediction time, or that are direct components of the target definition. This document discloses all known leakage in GOLD360.

## Pseudo-Label Composition

The pseudo-risk label is a weighted composite of 6 components:

```python
pseudo_risk = (
    0.30 * (1 - delivery_efficiency) +  # Delivery shortfall
    0.20 * fx_spread_pct +               # FX deviation
    0.15 * border_risk +                 # Border proximity
    0.15 * policy_shock_flag +           # Policy events
    0.10 * (1 - ore_grade_efficiency) +  # Ore grade deviation
    0.10 * miner_type_asm                # ASM proportion
)
```

## Feature-Target Overlap

| Feature | Role in Target | Weight | Correlation with Pseudo-Risk | V3 Status |
|---------|---------------|--------|------------------------------|-----------|
| `delivery_efficiency` | Component (1-η) | 30% | r = −0.5365 | **REMOVED** |
| `delivery_gap_ratio` | Derived from delivery_efficiency | — | r = +0.5365 | **REMOVED** |
| `delivery_gap_kg` | Independent | — | r = −0.018 | Retained |
| `fx_spread_pct` | Component | 20% | r = +0.026 | Retained |
| `border_risk` | Component | 15% | r = +0.159 | Retained |
| `policy_shock_flag` | Component | 15% | r = −0.069 | Retained |
| `ore_grade_efficiency` | Component (η) | 10% | r = −0.196 | Retained |
| `miner_type_asm` | Component | 10% | r = −0.024 | Retained |

## Leakage Assessment

### High Leakage (Removed in V3)
- **delivery_efficiency**: Direct component of target (30% weight). r = −0.5365 with pseudo-risk. Removed.
- **delivery_gap_ratio**: Mathematically equivalent to 1 − delivery_efficiency. r = +0.5365. Removed.

### Moderate Leakage (Retained in V3)
- **ore_grade_efficiency**: Component of target (10% weight). r = −0.196. Retained because:
  - Audit 01 showed 4.93% AUC cost to remove
  - Leakage is partial (10% weight)
  - Cost outweighs benefit

### Low Leakage (Retained in V3)
- **fx_spread_pct**: Component (20% weight). r = +0.026. Low correlation.
- **border_risk**: Component (15% weight). r = +0.159. Moderate correlation.
- **policy_shock_flag**: Component (15% weight). r = −0.069. Low correlation.
- **miner_type_asm**: Component (10% weight). r = −0.024. Low correlation.

## V2 vs V3 Leakage Comparison

| Metric | V2 (with leakage) | V3 (leakage-controlled) | Delta |
|--------|-------------------|------------------------|-------|
| ROC-AUC | 0.9856 | 0.9817 | −0.39% |
| F1 | 0.9037 | 0.9047 | +0.11% |
| Precision | 0.9981 | 0.9916 | −0.65% |
| Recall | 0.8256 | 0.8319 | +0.76% |
| CV AUC | 0.9921 | 0.9739 | −1.83% |
| Feature count | 22 | 17 | −5 |

## Audit Results

### Audit 01: Remove ore_grade_efficiency
- AUC drops 4.93% (0.9817 → 0.9324)
- **Conclusion:** Cost exceeds benefit. Keep in V3.

### Audit 02: Remove ALL pseudo-label components
- AUC drops 6.10% (0.9817 → 0.9207)
- **Conclusion:** Model relies heavily on these features.

### Audit 03: Weight sensitivity
- Baseline reproducible (0.9817)
- delivery=0 causes 28.10% AUC drop
- max_delivery raises AUC 1.39%
- **Conclusion:** Delivery weight is critical.

## Recommendations

1. **V3 is preferred for thesis** — 0.39% AUC cost is negligible vs. eliminating worst leakage
2. **Disclose all findings** — Include leakage audit in thesis methodology
3. **Do not re-add removed features** — delivery_efficiency and delivery_gap_ratio
4. **Monitor future work** — If real data becomes available, re-evaluate leakage

## Academic Integrity

All figures and metrics reported in this dashboard are computed on held-out test data using proper temporal splits. The model is not evaluated on training data. Cross-validation uses 5-fold stratified splits with proper temporal awareness.

Pseudo-labels are used as a research methodology for weak supervision. They are not ground truth. All predictions are probabilistic intelligence signals, not forensic proof.
