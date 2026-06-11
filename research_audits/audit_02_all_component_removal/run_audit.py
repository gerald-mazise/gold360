"""Audit 02: Remove ALL pseudo-label component features from feature set."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audit_engine import run_audit

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# Pseudo-label components to REMOVE:
# delivery_gap_ratio (30%) - already removed in V3
# fx_spread_pct (20%) - REMOVE
# border_risk (15%) - REMOVE
# policy_shock_flag (15%) - REMOVE
# ore_grade_efficiency (10%) - REMOVE
# miner_type_asm (10%) - REMOVE
#
# Remaining features: only those NOT used in pseudo-label generation
FEATURE_COLS = [
    "delivery_gap_kg", "delivery_gap_kg_roll3", "delivery_gap_kg_roll3_std",
    "rainfall_raw", "fgr_access",
    "border_distance_km", "fgr_distance_km",
    "license_encoded",
    "gold_price_usd", "inflation_rate", "payment_delay_days", "ore_processed_tonnes",
]

if __name__ == "__main__":
    results = run_audit(
        audit_name="Audit 02: Remove ALL pseudo-label components",
        output_dir=OUTPUT_DIR,
        feature_cols_override=FEATURE_COLS,
        skip_robustness=False,
        skip_ablation=False,
    )
