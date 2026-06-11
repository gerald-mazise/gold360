"""Audit 01: Remove ore_grade_efficiency from feature set."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audit_engine import run_audit

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# V3 baseline features MINUS ore_grade_efficiency
FEATURE_COLS = [
    "delivery_gap_kg", "delivery_gap_kg_roll3", "delivery_gap_kg_roll3_std",
    "fx_spread_pct", "rainfall_raw",
    "policy_shock_flag", "border_risk", "fgr_access",
    "border_distance_km", "fgr_distance_km",
    "miner_type_asm", "license_encoded",
    "gold_price_usd", "inflation_rate", "payment_delay_days", "ore_processed_tonnes",
]

if __name__ == "__main__":
    results = run_audit(
        audit_name="Audit 01: Remove ore_grade_efficiency",
        output_dir=OUTPUT_DIR,
        feature_cols_override=FEATURE_COLS,
        skip_robustness=False,
        skip_ablation=False,
    )
