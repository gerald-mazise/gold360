# GOLD360 — Data Dictionary

## Primary Dataset: Mine Operations (Monthly)

**File:** `synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv`
**Shape:** 9,000 rows × 30 columns (125 mines × 72 months)
**Temporal Range:** 2020-01 to 2025-12

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `month` | string | Month identifier (YYYY-MM) | "2023-06" |
| `mine_id` | string | Unique mine identifier | "MINE_001" |
| `mine_name` | string | Human-readable mine name | "Chevron Mine" |
| `province` | string | Zimbabwe province | "Mashonaland West" |
| `district` | string | District within province | "Chegutu" |
| `miner_type` | string | Mining type: ASM or Formal | "ASM" |
| `license_status` | string | License: Licensed/Cooperative/Informal/Pending | "Licensed" |
| `mine_latitude` | float | Mine latitude | -18.5392 |
| `mine_longitude` | float | Mine longitude | 29.9167 |
| `nearest_fidelity_office` | string | Nearest FGR buying office | "Kwekwe" |
| `distance_to_fidelity_km` | float | Distance to FGR (km) | 45.2 |
| `nearest_border_post` | string | Nearest border post | "Beitbridge" |
| `distance_to_border_km` | float | Distance to border (km) | 120.5 |
| `ore_processed_tonnes` | float | Ore processed in tonnes | 1250.0 |
| `ore_grade_gpt` | float | Ore grade (grams per tonne) | 3.2 |
| `recovery_rate_pct` | float | Recovery rate percentage | 85.0 |
| `estimated_gold_yield_kg` | float | Estimated gold yield (kg) | 3.40 |
| `official_delivery_kg` | float | Official delivery to FGR (kg) | 2.10 |
| `delivery_gap_kg` | float | Delivery gap (yield - delivery) | 1.30 |
| `payment_delay_days` | float | Payment delay in days | 15.0 |
| `compliance_score` | float | Compliance score (0-100) | 72.0 |
| `gold_price_usd` | float | Gold price (USD/oz) | 1950.0 |
| `fx_market_rate` | float | FX market rate | 0.025 |
| `inflation_rate` | float | Inflation rate (%) | 120.0 |
| `rainfall_mm` | float | Monthly rainfall (mm) | 85.0 |
| `rainfall_anomaly_mm` | float | Rainfall anomaly from mean | -12.0 |
| `macro_energy_availability_index` | float | Energy availability index (0-1) | 0.65 |
| `national_gap_ratio` | float | National delivery gap ratio | 0.35 |
| `mirror_trade_gap_ratio` | float | Mirror trade discrepancy ratio | 0.12 |
| `policy_shock_flag` | int | Policy event indicator (0/1) | 0 |

## Additional Data Sources (15 CSVs)

| # | File | Frequency | Key Columns |
|---|------|-----------|-------------|
| 1 | `synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv` | monthly | mine_id, month, all 30 columns |
| 2 | `gold_price_monthly.csv` | monthly | month, gold_price_usd |
| 3 | `fx_market_annual.csv` | annual | year, fx_market_rate |
| 4 | `rainfall_province_monthly_zimbabwe_2020_2025.csv` | monthly | month, province, rainfall_mm |
| 5 | `zimbabwe_gold_policy_event_intelligence_2020_2025.csv` | event | date, event_type, description |
| 6 | `fgr_gold_deliveries_quarterly_verified_zimbabwe_2020_2021.csv` | quarterly | quarter, delivery_kg |
| 7 | `zimstat_gold_production_quarterly_zimbabwe_2020_2025.csv` | quarterly | quarter, production_kg |
| 8 | `inflation_annual.csv` | annual | year, inflation_rate |
| 9 | `zimbabwe_inflation_imf_avg_cpi_annual_2010_2025.csv` | annual | year, inflation_rate |
| 10 | `zimbabwe_primary_energy_consumption_annual_2020_2024.csv` | annual | year, energy_index |
| 11 | `fgr_buying_offices.csv` | static | office_name, latitude, longitude |
| 12 | `gold_mirror_trade_discrepancy_zimbabwe_2020_2024.csv` | annual | year, gap_ratio |
| 13 | `gold_exports_annual.csv` | annual | year, export_value |
| 14 | `gold_smuggling_incident_log_zimbabwe_2020_2025.csv` | event | date, incident_type |
| 15 | `gold360_synthetic_data_dictionary.csv` | metadata | column_name, description |

## Feature Engineering Formulas

| Feature | Formula | Source Columns |
|---------|---------|----------------|
| `delivery_gap_kg` | `max(estimated_gold_yield_kg - official_delivery_kg, 0)` | estimated_gold_yield_kg, official_delivery_kg |
| `delivery_gap_ratio` | `delivery_gap_kg / estimated_gold_yield_kg` | delivery_gap_kg, estimated_gold_yield_kg |
| `delivery_efficiency` | `official_delivery_kg / estimated_gold_yield_kg` | official_delivery_kg, estimated_gold_yield_kg |
| `ore_grade_efficiency` | `ore_grade_gpt * recovery_rate_pct / 100` | ore_grade_gpt, recovery_rate_pct |
| `border_risk` | `1 / (1 + distance_to_border_km / 50)` | distance_to_border_km |
| `fgr_access` | `1 / (1 + distance_to_fidelity_km / 200)` | distance_to_fidelity_km |
| `miner_type_asm` | `(miner_type == "ASM") ? 1 : 0` | miner_type |
| `license_encoded` | `{"Licensed": 0, "Cooperative": 1, "Informal": 2, "Pending": 3}` | license_status |

## Temporal Split

| Split | Months | Samples | Positive Rate |
|-------|--------|---------|---------------|
| Train | 2020-01 to 2023-07 | 5,500 | 40.87% |
| Val | 2023-08 to 2024-09 | 1,750 | 62.80% |
| Test | 2024-10 to 2025-12 | 1,750 | 72.74% |

**Note:** Positive rate drifts across splits due to temporal changes in gold market conditions.
