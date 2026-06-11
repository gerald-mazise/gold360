# GOLD360 — Feature Registry

## Summary

| Attribute | Value |
|-----------|-------|
| Total Features | 17 |
| Feature Groups | 5 (Delivery, Macro, Operational, Governance, Spatial) |
| Excluded Features | 5 (delivery_efficiency, delivery_gap_ratio, delivery_efficiency_roll3, delivery_efficiency_roll3_std, energy_stress) |
| Top Feature | delivery_gap_kg (26.42%) |
| Total Feature Importance | 100% |

## Feature Groups

### Delivery (3 features, 28.82% importance)
| Feature | Formula | Importance | Type |
|---------|---------|------------|------|
| `delivery_gap_kg` | `max(estimated_gold_yield_kg - official_delivery_kg, 0)` | 26.42% | continuous |
| `delivery_gap_kg_roll3` | `rolling_3m_mean(delivery_gap_kg)` | 1.32% | continuous |
| `delivery_gap_kg_roll3_std` | `rolling_3m_std(delivery_gap_kg)` | 1.08% | continuous |

### Macro (3 features, 13.45% importance)
| Feature | Formula | Importance | Type |
|---------|---------|------------|------|
| `fx_spread_pct` | `fx_market_rate` | 8.86% | continuous |
| `inflation_rate` | `inflation_rate` | 3.16% | continuous |
| `gold_price_usd` | `gold_price_usd` | 1.43% | continuous |

### Operational (3 features, 21.26% importance)
| Feature | Formula | Importance | Type |
|---------|---------|------------|------|
| `ore_grade_efficiency` | `ore_grade_gpt * recovery_rate_pct / 100` | 19.30% | continuous |
| `ore_processed_tonnes` | `ore_processed_tonnes` | 16.87% | continuous |
| `payment_delay_days` | `payment_delay_days` | 1.09% | continuous |
| `rainfall_raw` | `rainfall_mm` | 0.93% | continuous |

### Governance (2 features, 12.10% importance)
| Feature | Formula | Importance | Type |
|---------|---------|------------|------|
| `miner_type_asm` | `(miner_type == "ASM") ? 1 : 0` | 9.66% | binary |
| `license_encoded` | `{"Licensed": 0, "Cooperative": 1, "Informal": 2, "Pending": 3}` | 2.44% | ordinal |
| `policy_shock_flag` | `policy_shock_flag` | 1.57% | binary |

### Spatial (4 features, 5.85% importance)
| Feature | Formula | Importance | Type |
|---------|---------|------------|------|
| `border_distance_km` | `distance_to_border_km` | 2.07% | continuous |
| `border_risk` | `1 / (1 + distance_to_border_km / 50)` | 2.00% | index |
| `fgr_access` | `1 / (1 + distance_to_fidelity_km / 200)` | 0.94% | index |
| `fgr_distance_km` | `distance_to_fidelity_km` | 0.84% | continuous |

## Excluded Features

| Feature | Reason | Correlation with Target |
|---------|--------|------------------------|
| `delivery_efficiency` | Target leakage — inverse of delivery_gap_ratio | r = -0.5365 |
| `delivery_gap_ratio` | Target leakage — used in pseudo-label (30% weight) | r = +0.5365 |
| `delivery_efficiency_roll3` | Derived from excluded feature | — |
| `delivery_efficiency_roll3_std` | Derived from excluded feature | — |
| `energy_stress` | Dead feature — placeholder constant (0.5) | 0.0 importance |

## Dual-Role Features

Some features are ALSO components of the pseudo-label target:

| Feature | Target Weight | Importance | Leakage Risk |
|---------|---------------|------------|--------------|
| `fx_spread_pct` | 20% | 8.86% | **Monitor** |
| `border_risk` | 15% | 2.00% | **Monitor** |
| `policy_shock_flag` | 15% | 1.57% | **Monitor** |
| `ore_grade_efficiency` | 10% | 19.30% | **Monitor** |
| `miner_type_asm` | 10% | 9.66% | **Monitor** |

**Research Audit Finding:** Removing `ore_grade_efficiency` causes 4.93% AUC drop. The cost outweighs the leakage risk.

## Feature Importance Ranking

| Rank | Feature | Importance | Cumulative |
|------|---------|------------|------------|
| 1 | delivery_gap_kg | 26.42% | 26.42% |
| 2 | ore_grade_efficiency | 19.30% | 45.72% |
| 3 | ore_processed_tonnes | 16.87% | 62.59% |
| 4 | miner_type_asm | 9.66% | 72.25% |
| 5 | fx_spread_pct | 8.86% | 81.11% |
| 6 | inflation_rate | 3.16% | 84.27% |
| 7 | license_encoded | 2.44% | 86.71% |
| 8 | border_distance_km | 2.07% | 88.78% |
| 9 | border_risk | 2.00% | 90.78% |
| 10 | policy_shock_flag | 1.57% | 92.35% |
| 11 | gold_price_usd | 1.43% | 93.78% |
| 12 | delivery_gap_kg_roll3 | 1.32% | 95.10% |
| 13 | payment_delay_days | 1.09% | 96.19% |
| 14 | delivery_gap_kg_roll3_std | 1.08% | 97.27% |
| 15 | fgr_access | 0.94% | 98.21% |
| 16 | rainfall_raw | 0.93% | 99.14% |
| 17 | fgr_distance_km | 0.84% | 100.00% |
