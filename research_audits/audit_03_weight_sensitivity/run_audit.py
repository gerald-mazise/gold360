"""Audit 03: Sensitivity analysis on pseudo-label component weights."""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audit_engine import run_audit

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# 12 weight combinations to test
WEIGHT_EXPERIMENTS = [
    {"delivery": 0.30, "fx": 0.20, "border": 0.15, "policy": 0.15, "ore_grade": 0.10, "asm": 0.10, "label": "baseline"},
    {"delivery": 0.40, "fx": 0.20, "border": 0.10, "policy": 0.10, "ore_grade": 0.10, "asm": 0.10, "label": "delivery_heavy"},
    {"delivery": 0.20, "fx": 0.30, "border": 0.15, "policy": 0.15, "ore_grade": 0.10, "asm": 0.10, "label": "fx_heavy"},
    {"delivery": 0.30, "fx": 0.20, "border": 0.25, "policy": 0.15, "ore_grade": 0.05, "asm": 0.05, "label": "border_heavy"},
    {"delivery": 0.30, "fx": 0.20, "border": 0.15, "policy": 0.25, "ore_grade": 0.05, "asm": 0.05, "label": "policy_heavy"},
    {"delivery": 0.25, "fx": 0.25, "border": 0.25, "policy": 0.25, "ore_grade": 0.00, "asm": 0.00, "label": "no_ore_asm"},
    {"delivery": 0.50, "fx": 0.10, "border": 0.10, "policy": 0.10, "ore_grade": 0.10, "asm": 0.10, "label": "max_delivery"},
    {"delivery": 0.20, "fx": 0.20, "border": 0.20, "policy": 0.20, "ore_grade": 0.10, "asm": 0.10, "label": "equal_excl_ore_asm"},
    {"delivery": 0.30, "fx": 0.20, "border": 0.15, "policy": 0.15, "ore_grade": 0.00, "asm": 0.20, "label": "no_ore_double_asm"},
    {"delivery": 0.15, "fx": 0.15, "border": 0.15, "policy": 0.15, "ore_grade": 0.20, "asm": 0.20, "label": "ore_asm_heavy"},
    {"delivery": 0.30, "fx": 0.20, "border": 0.15, "policy": 0.15, "ore_grade": 0.10, "asm": 0.10, "label": "baseline_repeat"},
    {"delivery": 0.00, "fx": 0.25, "border": 0.25, "policy": 0.25, "ore_grade": 0.15, "asm": 0.10, "label": "no_delivery"},
]

# V3 baseline feature set (all 17)
FEATURE_COLS = [
    "delivery_gap_kg", "delivery_gap_kg_roll3", "delivery_gap_kg_roll3_std",
    "fx_spread_pct", "ore_grade_efficiency", "rainfall_raw",
    "policy_shock_flag", "border_risk", "fgr_access",
    "border_distance_km", "fgr_distance_km",
    "miner_type_asm", "license_encoded",
    "gold_price_usd", "inflation_rate", "payment_delay_days", "ore_processed_tonnes",
]


if __name__ == "__main__":
    all_results = []

    for i, exp in enumerate(WEIGHT_EXPERIMENTS):
        label = exp.pop("label")
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {i+1}/12: {label}")
        print(f"{'#'*70}\n")

        exp_output = os.path.join(OUTPUT_DIR, f"weight_{i+1:02d}_{label}")

        results = run_audit(
            audit_name=f"Weight sensitivity: {label}",
            output_dir=exp_output,
            feature_cols_override=FEATURE_COLS,
            risk_weights=exp,
            skip_robustness=True,
            skip_ablation=True,
            skip_benchmark=True,
        )

        all_results.append({
            "experiment": i + 1,
            "label": label,
            "weights": exp,
            "roc_auc": results["test_metrics"]["roc_auc"],
            "f1": results["test_metrics"]["f1_score"],
            "precision": results["test_metrics"]["precision"],
            "recall": results["test_metrics"]["recall"],
            "mcc": results["test_metrics"]["matthews_corrcoef"],
            "cv_mean_auc": results["cv_summary"]["mean_auc"],
            "positive_rate_test": results["test_metrics"].get("positive_rate_test"),
        })

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "weight_sensitivity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS COMPLETE")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")

    # Print comparison table
    print(f"\n{'Label':<25s} {'AUC':>8s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'MCC':>8s} {'CV AUC':>8s}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['label']:<25s} {r['roc_auc']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['mcc']:>8.4f} {r['cv_mean_auc']:>8.4f}")
