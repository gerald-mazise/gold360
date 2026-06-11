# GOLD360 — Data Lineage

## Dataset Overview

| # | Dataset | Type | Frequency | Source | Synthetic |
|---|---------|------|-----------|--------|-----------|
| 1 | FGR Deliveries | Economic | Quarterly | Fidelity Gold Refinery | No |
| 2 | ZIMSTAT Production | Economic | Quarterly | ZIMSTAT | No |
| 3 | Synthetic Mine Ops | Operational | Monthly | Simulated | **Yes** |
| 4 | Gold Price | Market | Monthly | LBMA | No |
| 5 | FX Distortion | Macro | Annual | RBZ / Parallel Market | No |
| 6 | Inflation | Macro | Annual | ZIMSTAT | No |
| 7 | CPI | Macro | Annual | ZIMSTAT | No |
| 8 | Rainfall | Environmental | Monthly | Meteorological Services | No |
| 9 | Energy | Operational | Annual | ZESA | No |
| 10 | Policy Events | Governance | Event-based | Legislative Record | No |
| 11 | FGR Offices | Spatial | Static | Geolocated | No |
| 12 | Mirror Trade | Trade | Annual | UN Comtrade | No |
| 13 | Gold Exports | Trade | Annual | ZIMSTAT | No |
| 14 | Smuggling Incidents | Enforcement | Event-based | Media / Reports | No |

## Data Engineering Pipeline

1. **Loading** — `DataLoader.load_all()` loads all 14+ datasets from CSV with caching
2. **Validation** — `DataValidator` checks missing rates, temporal integrity, duplicates, outliers
3. **Temporal Alignment** — `TemporalAligner` expands quarterly→monthly and annual→monthly
4. **Harmonization** — `DataHarmonizer` fuses all sources into unified intelligence table
5. **Registration** — `DataLineage` tracks every dataset with path, type, frequency, row/column count

## Synthetic Data Governance

- `synthetic_mine_ops.csv` is explicitly tagged with `synthetic: true` in config
- All outputs must visually distinguish synthetic from observed data
- Synthetic data must not drive final research conclusions
- Dashboard marks synthetic data with distinct visual treatment

## Temporal Integrity

- No future leakage permitted
- Temporal train/val/test split enforced
- Walk-forward cross-validation respects time ordering
- All quarterly/annual data expanded to monthly via TemporalAligner
