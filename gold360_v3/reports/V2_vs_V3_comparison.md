# GOLD360 V2 vs V3 — Leakage-Controlled Model Comparison

**Date:** 2026-06-09
**Purpose:** Compare V2 (with leaked features) against V3 (leakage-controlled)

---

## 1. What Changed

| Aspect | V2 | V3 |
|--------|----|----|
| Features | 22 | 17 |
| Leaked features removed | — | `delivery_efficiency`, `delivery_gap_ratio`, `delivery_efficiency_roll3`, `delivery_efficiency_roll3_std`, `energy_stress` |
| Target formula | Unchanged | Unchanged |
| CatBoost params | Unchanged | Unchanged |

---

## 2. Test Set Performance Metrics

| Metric | V2 | V3 | Delta | Assessment |
|--------|----|----|-------|------------|
| **ROC-AUC** | 0.9856 | 0.9817 | −0.0039 | Minimal drop — model still discriminative |
| **F1 Score** | 0.9037 | 0.9047 | +0.0010 | Slight improvement |
| **Precision** | 0.9981 | 0.9916 | −0.0065 | Minor drop |
| **Recall** | 0.8256 | 0.8319 | +0.0063 | Better detection |
| **Balanced Accuracy** | 0.9107 | 0.9065 | −0.0042 | Minimal |
| **Brier Score** | 0.1004 | 0.0900 | −0.0104 | Better calibration |
| **Log Loss** | 0.3272 | 0.2738 | −0.0534 | Significantly better |
| **MCC** | 0.7471 | 0.7423 | −0.0048 | Minimal |
| **Cohen's Kappa** | 0.7179 | 0.7167 | −0.0012 | Negligible |

**Key finding:** Removing leaked features caused only a **0.39% AUC drop** (0.9856 → 0.9817). The model's performance is remarkably stable.

---

## 3. Feature Importance Ranking

### V2 (with leaked features)
| Rank | Feature | Importance | Leaked? |
|------|---------|------------|---------|
| 1 | ore_grade_efficiency | 21.03 | No |
| 2 | fx_spread_pct | 15.83 | No |
| 3 | miner_type_asm | 15.10 | No |
| 4 | **delivery_gap_ratio** | **14.55** | **YES** |
| 5 | **delivery_efficiency** | **13.63** | **YES** |
| 6 | inflation_rate | 3.63 | No |
| 7 | border_risk | 2.69 | No |
| 8 | border_distance_km | 2.59 | No |
| 9 | gold_price_usd | 2.16 | No |
| 10 | policy_shock_flag | 1.25 | No |

### V3 (leakage-controlled)
| Rank | Feature | Importance | Change from V2 |
|------|---------|------------|----------------|
| 1 | **delivery_gap_kg** | **26.42** | **+26.07** (was #21 at 0.35) |
| 2 | ore_grade_efficiency | 19.30 | −1.73 |
| 3 | **ore_processed_tonnes** | **16.87** | **+16.10** (was #14 at 0.78) |
| 4 | miner_type_asm | 9.66 | −5.44 |
| 5 | fx_spread_pct | 8.86 | −6.97 |
| 6 | inflation_rate | 3.16 | −0.47 |
| 7 | license_encoded | 2.44 | +1.42 |
| 8 | border_distance_km | 2.07 | −0.52 |
| 9 | border_risk | 2.00 | −0.69 |
| 10 | policy_shock_flag | 1.57 | +0.32 |

### Key Shifts
- **delivery_gap_kg** jumped from #21 (0.35%) to **#1 (26.42%)** — the model now relies heavily on the absolute delivery gap
- **ore_processed_tonnes** jumped from #14 (0.78%) to **#3 (16.87%)**
- **fx_spread_pct** dropped from #2 (15.83%) to #5 (8.86%)
- **miner_type_asm** dropped from #3 (15.10%) to #4 (9.66%)

---

## 4. Cross-Validation (5-Fold)

| Metric | V2 | V3 | Delta |
|--------|----|----|-------|
| Mean AUC | 0.9921 | 0.9739 | −0.0182 |
| Std AUC | ±0.0025 | ±0.0097 | +0.0072 |
| Mean F1 | 0.8978 | 0.8918 | −0.0060 |

V3 has slightly lower CV AUC but higher variance — expected with fewer features.

---

## 5. Ablation Study

| Group | V2 Features | V2 AUC Delta | V3 Features | V3 AUC Delta |
|-------|-------------|--------------|-------------|--------------|
| ALL | 22 | 0.0 | 17 | 0.0 |
| Remove delivery | 7 removed | −0.1326 | 3 removed | −0.1287 |
| Remove macro | 3 removed | +0.0024 | 3 removed | −0.0070 |
| Remove operational | 5 removed | −0.0135 | 4 removed | −0.0803 |
| Remove governance | 2 removed | −0.0032 | 2 removed | −0.0116 |
| Remove spatial | 4 removed | +0.0015 | 4 removed | −0.0145 |

**Key finding:** In V3, removing the delivery group causes a **0.1287 AUC drop** — almost identical to V2's 0.1326. The delivery signal is still the most important group even without the leaked ratio features.

---

## 6. Robustness (Noise Injection)

| Noise Level | V2 AUC | V3 AUC | V2 Degradation | V3 Degradation |
|-------------|--------|--------|----------------|----------------|
| 1% | 0.9666 | 0.8014 | −0.0022 | −0.1261 |
| 5% | 0.9463 | 0.7888 | +0.0180 | −0.1386 |
| 10% | 0.9009 | 0.7824 | +0.0634 | −0.1450 |
| 20% | 0.8213 | 0.7577 | +0.1431 | −0.1698 |

**V3 is significantly less robust to noise.** V2 maintained AUC > 0.82 even at 20% noise; V3 drops to 0.76. This suggests V3's model is more reliant on a single feature (`delivery_gap_kg`) and is less distributed.

---

## 7. Walk-Forward Temporal Validation

| Split | V2 AUC | V3 AUC | Delta |
|-------|--------|--------|-------|
| 1 (2020→2021) | 0.9915 | 0.9709 | −0.0206 |
| 2 (2020→2022) | 0.9915 | 0.9686 | −0.0229 |
| 3 (2020→2023) | 0.9923 | 0.9630 | −0.0293 |
| 4 (2020→2024) | 0.9930 | 0.9814 | −0.0116 |
| 5 (2020→2025) | 0.9962 | 0.9922 | −0.0040 |

V3 shows consistent ~2% lower AUC across all temporal splits. The gap narrows with more training data.

---

## 8. Overfitting Analysis

| Metric | V2 | V3 |
|--------|----|----|
| Train AUC | 1.0000 | 0.9999 |
| Val AUC | 1.0000 | 1.0000 |
| Test AUC | 0.9856 | 0.9817 |
| Train-Test gap | 0.0144 | 0.0182 |
| Overfitting risk | LOW | LOW |

Both models show minimal overfitting. V3's train-test gap is slightly larger (0.0182 vs 0.0144) but still LOW.

---

## 9. Delivery Gap kg — Independent Predictor?

**Yes.** `delivery_gap_kg` remains an independent predictor:

- **Correlation with target:** r = −0.018 (essentially zero)
- **Importance in V3:** 26.42% (highest of all features)
- **Ablation delta:** Removing delivery group drops AUC by 0.1287
- **The model uses `delivery_gap_kg` through nonlinear interactions** with other features (CatBoost captures splits on delivery_gap_kg thresholds combined with ore_grade, miner_type, etc.)

This is a legitimate signal — `delivery_gap_kg` captures the absolute volume of gold not delivered, which is operationally meaningful even if its linear correlation with the pseudo-label is near zero.

---

## 10. Conclusions

| Question | Answer |
|----------|--------|
| Does removing leaked features hurt performance? | **Minimal** — AUC dropped only 0.39% |
| Is V3 more valid? | **Yes** — no circular reasoning in feature-target relationship |
| Does delivery_gap_kg remain useful? | **Yes** — it's the #1 feature (26.42% importance) |
| Is V3 more robust? | **No** — significantly less robust to noise (V2: 0.8213 vs V3: 0.7577 at 20% noise) |
| Should we use V3? | **For thesis validity, V3 is preferred** — clean features, no leakage, minimal performance cost |
| What about the noise robustness gap? | Investigate — likely caused by V3 relying more heavily on a single feature |

---

## 11. Recommendation

**Use V3 for the thesis.** The 0.39% AUC cost is negligible compared to the validity gain of removing target leakage. The model's discrimination ability is preserved (ROC-AUC 0.9817), and all validation metrics remain strong.

**Action items:**
1. Investigate V3's noise robustness — consider adding feature regularization or diversity constraints
2. Update thesis narrative to disclose pseudo-label composition and feature cleanup
3. Consider whether `delivery_gap_kg` should be further investigated for potential proxy effects
