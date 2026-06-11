"""
GOLD360 Research Audit Engine
Modular evaluation script for controlled experiments.
Based on gold360_v3/gold360/evaluation/run_full_evaluation.py
Does NOT modify V3 — reads data from V3/data/raw/, saves reports locally.
"""
import os
import sys
import json
import warnings
import yaml
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    log_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, average_precision_score,
)
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier as _CatBoost

# Path to V3 data (never modify)
V3_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "gold360_v3", "data", "raw")
V3_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "gold360_v3", "config", "default.yaml")


def save_json(data, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_config():
    with open(V3_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_risk_score(df, weights):
    """
    Compute pseudo-label risk score with configurable weights.
    weights dict keys: delivery, fx, border, policy, ore_grade, asm
    """
    risk_score = np.zeros(len(df))

    # Delivery gap (30% default)
    gap_ratio = df["delivery_gap_ratio"].fillna(0).values
    risk_score += np.clip(gap_ratio * 3.0, 0, 1) * weights.get("delivery", 0.30)

    # FX spread (20% default)
    fx = df["fx_spread_pct"].fillna(0).values
    fx_max = fx.max() if fx.max() > 0 else 1
    risk_score += np.clip(fx / fx_max, 0, 1) * weights.get("fx", 0.20)

    # Border proximity (15% default)
    border = df["border_risk"].fillna(0).values
    risk_score += border * weights.get("border", 0.15)

    # Policy shock (15% default)
    policy = df["policy_shock_flag"].fillna(0).values
    risk_score += policy * weights.get("policy", 0.15)

    # Ore grade efficiency anomaly (10% default)
    ore_eff = df["ore_grade_efficiency"].fillna(0).values
    eff_score = np.abs(ore_eff - np.median(ore_eff)) / (np.std(ore_eff) + 1e-8)
    risk_score += np.clip(eff_score, 0, 1) * weights.get("ore_grade", 0.10)

    # ASM miner type (10% default)
    asm = df["miner_type_asm"].values
    risk_score += asm * weights.get("asm", 0.10)

    risk_score = np.clip(risk_score, 0, 1)
    return risk_score


def engineer_features(df):
    """All feature engineering from V3 eval script."""
    # Delivery features
    df["delivery_gap_kg"] = (df.get("estimated_gold_yield_kg", 0) - df.get("official_delivery_kg", 0)).clip(lower=0)
    df["delivery_efficiency"] = (df.get("official_delivery_kg", 0) / df.get("estimated_gold_yield_kg", 1).replace(0, 1)).clip(0, 2)
    df["delivery_gap_ratio"] = (df["delivery_gap_kg"] / df.get("estimated_gold_yield_kg", 1).replace(0, 1)).clip(0, 1)

    # Rolling features
    for col in ["delivery_gap_kg", "delivery_efficiency"]:
        df[f"{col}_roll3"] = df.groupby("mine_id")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_roll3_std"] = df.groupby("mine_id")[col].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))

    # Macro
    df["fx_spread_pct"] = df.get("fx_market_rate", 1.0)

    # Operational
    df["ore_grade_efficiency"] = df.get("ore_grade_gpt", 1.0) * df.get("recovery_rate_pct", 100.0) / 100.0
    df["rainfall_raw"] = df.get("rainfall_mm", 0.0)

    # Governance
    df["policy_shock_flag"] = df.get("policy_shock_flag", 0).astype(float)

    # Spatial
    df["border_distance_km"] = df.get("distance_to_border_km", 100.0)
    df["fgr_distance_km"] = df.get("distance_to_fidelity_km", 100.0)
    df["border_risk"] = 1.0 / (1.0 + df["border_distance_km"] / 50.0)
    df["fgr_access"] = 1.0 / (1.0 + df["fgr_distance_km"] / 200.0)

    # Encode categorical
    df["miner_type_asm"] = (df.get("miner_type", "ASM") == "ASM").astype(float)
    df["license_encoded"] = df.get("license_status", "Licensed").map(
        {"Licensed": 0, "Cooperative": 1, "Informal": 2, "Pending": 3}
    ).fillna(2).astype(float)

    return df


def load_and_merge_data():
    """Load all 5 raw CSVs and merge onto mine_ops."""
    mine_ops = pd.read_csv(os.path.join(V3_DATA_DIR, "synthetic_mine_ops_monthly_zimbabwe_2020_2025.csv"))
    gold_price = pd.read_csv(os.path.join(V3_DATA_DIR, "gold_price_monthly.csv"))
    fx_market = pd.read_csv(os.path.join(V3_DATA_DIR, "fx_market_annual.csv"))
    rainfall = pd.read_csv(os.path.join(V3_DATA_DIR, "rainfall_province_monthly_zimbabwe_2020_2025.csv"))
    policy_events = pd.read_csv(os.path.join(V3_DATA_DIR, "zimbabwe_gold_policy_event_intelligence_2020_2025.csv"))

    df = mine_ops.copy()
    df["month"] = df["month"].astype(str)
    return df


def run_audit(
    audit_name,
    output_dir,
    feature_cols_override=None,
    risk_weights=None,
    skip_robustness=False,
    skip_benchmark=False,
    skip_ablation=False,
):
    """
    Run a single audit experiment.

    Args:
        audit_name: Human-readable name for this audit
        output_dir: Where to save reports
        feature_cols_override: If provided, use this list instead of default
        risk_weights: If provided, use these weights for pseudo-label generation
        skip_robustness: Skip noise injection test (slow)
        skip_benchmark: Skip CatBoost vs XGBoost comparison
        skip_ablation: Skip ablation study
    """
    os.makedirs(output_dir, exist_ok=True)
    config = load_config()
    pseudo_threshold = config.get("weak_supervision", {}).get("pseudo_label_threshold", 0.5)

    print("=" * 70)
    print(f"GOLD360 RESEARCH AUDIT: {audit_name}")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/8] Loading data...")
    df = load_and_merge_data()
    print(f"  Loaded: {df.shape}")

    # Step 2: Feature engineering
    print("\n[2/8] Engineering features...")
    df = engineer_features(df)

    # Step 3: Generate pseudo-labels with custom weights
    print("\n[3/8] Generating pseudo-labels...")
    weights = risk_weights or {
        "delivery": 0.30, "fx": 0.20, "border": 0.15,
        "policy": 0.15, "ore_grade": 0.10, "asm": 0.10
    }
    print(f"  Weights: {weights}")
    risk_score = compute_risk_score(df, weights)
    df["target_risk"] = risk_score
    df["target_binary"] = (risk_score > pseudo_threshold).astype(int)
    n_high = df["target_binary"].sum()
    print(f"  Target distribution: {n_high}/{len(df)} ({n_high/len(df)*100:.1f}%)")

    # Step 4: Define features
    print("\n[4/8] Defining feature set...")
    if feature_cols_override is not None:
        feature_cols = feature_cols_override
    else:
        feature_cols = [
            "delivery_gap_kg",
            "delivery_gap_kg_roll3", "delivery_gap_kg_roll3_std",
            "fx_spread_pct", "ore_grade_efficiency", "rainfall_raw",
            "policy_shock_flag", "border_risk", "fgr_access",
            "border_distance_km", "fgr_distance_km",
            "miner_type_asm", "license_encoded",
        ]
        for col in ["gold_price_usd", "inflation_rate", "payment_delay_days", "ore_processed_tonnes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                feature_cols.append(col)

    # Ensure all features exist and are numeric
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    feature_cols = sorted(set(feature_cols))
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Save feature list
    save_json({"feature_cols": feature_cols, "n_features": len(feature_cols),
               "audit_name": audit_name, "risk_weights": weights},
              "audit_config.json", output_dir)

    # Step 5: Temporal split
    print("\n[5/8] Temporal split...")
    df["month_sort"] = df["month"].astype(str)
    df = df.sort_values("month_sort").reset_index(drop=True)

    months_sorted = sorted(df["month"].unique())
    n_months = len(months_sorted)
    train_cutoff = months_sorted[int(n_months * 0.6)]
    val_cutoff = months_sorted[int(n_months * 0.8)]

    train_mask = df["month"] <= train_cutoff
    val_mask = (df["month"] > train_cutoff) & (df["month"] <= val_cutoff)
    test_mask = df["month"] > val_cutoff

    X_train = df.loc[train_mask, feature_cols].fillna(0).values
    y_train = df.loc[train_mask, "target_binary"].values
    X_val = df.loc[val_mask, feature_cols].fillna(0).values
    y_val = df.loc[val_mask, "target_binary"].values
    X_test = df.loc[test_mask, feature_cols].fillna(0).values
    y_test = df.loc[test_mask, "target_binary"].values

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Positive rate: Train={y_train.mean():.1%}, Val={y_val.mean():.1%}, Test={y_test.mean():.1%}")

    # Step 6: Train CatBoost
    print("\n[6/8] Training CatBoost...")
    cb_config = config.get("model", {}).get("catboost", {})
    model = _CatBoost(
        iterations=cb_config.get("iterations", 1000),
        learning_rate=cb_config.get("learning_rate", 0.03),
        depth=cb_config.get("depth", 6),
        l2_leaf_reg=cb_config.get("l2_leaf_reg", 3.0),
        border_count=cb_config.get("border_count", 128),
        early_stopping_rounds=cb_config.get("early_stopping_rounds", 50),
        random_seed=42, verbose=False,
        loss_function="Logloss", eval_metric="AUC",
        class_weights=cb_config.get("class_weight", [1.0, 2.0]),
    )

    use_val = len(np.unique(y_val)) > 1 and len(y_val) > 10
    if use_val:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        best_iter = model.get_best_iteration()
    else:
        model.fit(X_train, y_train)
        best_iter = None
    print(f"  Best iteration: {best_iter}")

    # Calibration
    cal_config = config.get("model", {}).get("calibration", {"enabled": True})
    if cal_config.get("enabled", True):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import KFold as KFoldCV
        cal_model = _CatBoost(
            iterations=cb_config.get("iterations", 1000),
            learning_rate=cb_config.get("learning_rate", 0.03),
            depth=cb_config.get("depth", 6),
            l2_leaf_reg=cb_config.get("l2_leaf_reg", 3.0),
            border_count=cb_config.get("border_count", 128),
            early_stopping_rounds=cb_config.get("early_stopping_rounds", 50),
            random_seed=42, verbose=False,
            loss_function="Logloss", eval_metric="AUC",
        )
        cv_splitter = KFoldCV(n_splits=3, shuffle=False)
        calibrated = CalibratedClassifierCV(cal_model, method="isotonic", cv=cv_splitter)
        if use_val:
            X_cal = np.vstack([X_train, X_val])
            y_cal = np.concatenate([y_train, y_val])
            calibrated.fit(X_cal, y_cal)
        else:
            calibrated.fit(X_train, y_train)
        model = calibrated
        print("  Applied isotonic calibration")

    # Step 7: Test metrics
    print("\n[7/8] Computing test metrics...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    y_bin = y_test

    roc_auc = roc_auc_score(y_bin, y_prob)
    f1 = f1_score(y_bin, y_pred, zero_division=0)
    precision = precision_score(y_bin, y_pred, zero_division=0)
    recall = recall_score(y_bin, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_bin, y_pred)
    brier = brier_score_loss(y_bin, y_prob)
    kappa = cohen_kappa_score(y_bin, y_pred)
    balanced_acc = balanced_accuracy_score(y_bin, y_pred)
    logloss = log_loss(y_bin, np.column_stack([1 - y_prob, y_prob]))
    avg_precision = average_precision_score(y_bin, y_prob)

    cm = confusion_matrix(y_bin, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Youden's J
    fpr, tpr, roc_thresholds = roc_curve(y_bin, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(roc_thresholds[optimal_idx])

    test_metrics = {
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "matthews_corrcoef": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "avg_precision": round(avg_precision, 4),
        "optimal_threshold_youden": round(optimal_threshold, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    print(f"\n  {'Metric':<25s} {'Value':>10s}")
    print(f"  {'-'*35}")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:<25s} {v:>10.4f}")
    print(f"\n  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    save_json(test_metrics, "test_metrics.json", output_dir)

    # Feature importance
    print("\n  Feature importance:")
    base_model = model.calibrated_classifiers_[0].estimator if hasattr(model, 'calibrated_classifiers_') else model
    importances = base_model.get_feature_importance()
    fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    fi_list = fi.to_dict("records")
    for i, row in enumerate(fi_list[:10]):
        print(f"  {i+1:2d}. {row['feature']:<35s} {row['importance']:.4f}")
    save_json(fi_list, "feature_importance.json", output_dir)

    # Cross-validation
    print("\n  Cross-validation (5-fold)...")
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {"fold": [], "auc": [], "f1": []}
    for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        if len(np.unique(y_va)) < 2:
            continue
        m = _CatBoost(iterations=300, learning_rate=0.05, depth=5, verbose=False, random_seed=42,
                       class_weights=cb_config.get("class_weight", [1.0, 2.0]))
        m.fit(X_tr, y_tr)
        yp = m.predict_proba(X_va)[:, 1]
        yb = (yp > 0.5).astype(int)
        cv_results["fold"].append(fold_i + 1)
        cv_results["auc"].append(round(roc_auc_score(y_va, yp), 4))
        cv_results["f1"].append(round(f1_score(y_va, yb, zero_division=0), 4))

    cv_summary = {
        "mean_auc": round(float(np.mean(cv_results["auc"])), 4),
        "std_auc": round(float(np.std(cv_results["auc"])), 4),
        "mean_f1": round(float(np.mean(cv_results["f1"])), 4),
    }
    print(f"  CV AUC: {cv_summary['mean_auc']:.4f} +/- {cv_summary['std_auc']:.4f}")
    save_json({**cv_summary, "folds": cv_results}, "cross_validation.json", output_dir)

    # Step 8: Optional tests
    results = {
        "audit_name": audit_name,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "feature_importance": fi_list,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "risk_weights": weights,
    }

    # Ablation
    if not skip_ablation:
        print("\n  Ablation study...")
        fgroup_map = {
            "delivery": [f for f in feature_cols if "delivery" in f or "gap" in f],
            "macro": [f for f in feature_cols if "fx" in f or "inflation" in f or "price" in f],
            "operational": [f for f in feature_cols if "ore" in f or "rain" in f or "payment" in f],
            "governance": [f for f in feature_cols if "policy" in f or "license" in f],
            "spatial": [f for f in feature_cols if "border" in f or "fgr" in f or "distance" in f],
        }
        baseline_auc = roc_auc_score(y_bin, y_prob)
        ablation_results = [{"group": "ALL", "auc": round(baseline_auc, 4), "delta": 0.0}]
        for gname, gfeatures in fgroup_map.items():
            remaining = [f for f in feature_cols if f not in gfeatures]
            if len(remaining) < 3:
                continue
            X_abl_train = df.loc[train_mask, remaining].fillna(0).values
            X_abl_test = df.loc[test_mask, remaining].fillna(0).values
            abm = _CatBoost(iterations=300, depth=5, verbose=False, random_seed=42,
                             class_weights=cb_config.get("class_weight", [1.0, 2.0]))
            abm.fit(X_abl_train, y_train)
            abl_prob = abm.predict_proba(X_abl_test)[:, 1]
            try:
                abl_auc = roc_auc_score(y_bin, abl_prob)
            except ValueError:
                abl_auc = 0.5
            delta = baseline_auc - abl_auc
            ablation_results.append({"group": gname, "auc": round(abl_auc, 4), "delta": round(delta, 4)})
            print(f"    Remove {gname:15s}: AUC={abl_auc:.4f}, delta={delta:+.4f}")
        save_json(ablation_results, "ablation_results.json", output_dir)
        results["ablation"] = ablation_results

    # Robustness
    if not skip_robustness:
        print("\n  Robustness test...")
        from sklearn.model_selection import cross_val_score
        baseline_cv = cross_val_score(
            _CatBoost(iterations=200, depth=4, verbose=False, random_seed=42),
            X_train, y_train, cv=3, scoring="roc_auc"
        ).mean()
        robustness = []
        for nl in [0.01, 0.05, 0.10, 0.20]:
            trial_aucs = []
            for trial in range(3):
                np.random.seed(trial)
                X_noisy = X_train + np.random.normal(0, nl, X_train.shape)
                score = cross_val_score(
                    _CatBoost(iterations=200, depth=4, verbose=False, random_seed=42),
                    X_noisy, y_train, cv=3, scoring="roc_auc"
                ).mean()
                trial_aucs.append(score)
            deg = float(baseline_cv - np.mean(trial_aucs))
            robustness.append({
                "noise_level": nl,
                "mean_auc": round(float(np.mean(trial_aucs)), 4),
                "degradation": round(deg, 4),
            })
            print(f"    Noise {nl:.0%}: AUC={np.mean(trial_aucs):.4f}, deg={deg:+.4f}")
        save_json(robustness, "robustness_results.json", output_dir)
        results["robustness"] = robustness

    # Overfitting analysis
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_roc = roc_auc_score(y_train, y_train_prob)
    overfitting = {
        "train_roc_auc": round(train_roc, 4),
        "test_roc_auc": test_metrics["roc_auc"],
        "train_test_gap": round(train_roc - test_metrics["roc_auc"], 4),
        "risk": "LOW" if (train_roc - test_metrics["roc_auc"]) < 0.1 else "MODERATE",
    }
    save_json(overfitting, "overfitting_analysis.json", output_dir)
    results["overfitting"] = overfitting

    print(f"\n  Overfitting gap: {overfitting['train_test_gap']:.4f} ({overfitting['risk']})")

    # Save full results
    save_json(results, "full_results.json", output_dir)

    print(f"\n{'='*70}")
    print(f"AUDIT COMPLETE: {audit_name}")
    print(f"Reports saved to: {output_dir}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GOLD360 Research Audit Engine")
    parser.add_argument("--name", default="default", help="Audit name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--remove-features", nargs="*", default=[], help="Features to remove")
    parser.add_argument("--weight-delivery", type=float, default=0.30)
    parser.add_argument("--weight-fx", type=float, default=0.20)
    parser.add_argument("--weight-border", type=float, default=0.15)
    parser.add_argument("--weight-policy", type=float, default=0.15)
    parser.add_argument("--weight-ore-grade", type=float, default=0.10)
    parser.add_argument("--weight-asm", type=float, default=0.10)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    # Build feature list with removals
    default_features = [
        "delivery_gap_kg", "delivery_gap_kg_roll3", "delivery_gap_kg_roll3_std",
        "fx_spread_pct", "ore_grade_efficiency", "rainfall_raw",
        "policy_shock_flag", "border_risk", "fgr_access",
        "border_distance_km", "fgr_distance_km",
        "miner_type_asm", "license_encoded",
        "gold_price_usd", "inflation_rate", "payment_delay_days", "ore_processed_tonnes",
    ]
    feature_cols = [f for f in default_features if f not in args.remove_features]

    weights = {
        "delivery": args.weight_delivery,
        "fx": args.weight_fx,
        "border": args.weight_border,
        "policy": args.weight_policy,
        "ore_grade": args.weight_ore_grade,
        "asm": args.weight_asm,
    }

    run_audit(
        audit_name=args.name,
        output_dir=args.output,
        feature_cols_override=feature_cols,
        risk_weights=weights,
        skip_robustness=args.skip_robustness,
        skip_benchmark=args.skip_benchmark,
        skip_ablation=args.skip_ablation,
    )
