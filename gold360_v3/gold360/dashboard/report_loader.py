"""Load evaluation reports from JSON files."""
import json
import os
from pathlib import Path


def _reports_dir():
    """Find the reports directory relative to the project root."""
    # From dashboard/report_loader.py -> dashboard -> gold360 (inner) -> gold360 (outer, project root)
    return Path(__file__).resolve().parent.parent.parent / "reports"


def load_report(filename):
    """Load a single JSON report file. Returns {} if not found."""
    path = _reports_dir() / filename
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_test_metrics():
    return load_report("test_metrics.json")

def get_feature_importance():
    return load_report("feature_importance.json")

def get_cross_validation():
    return load_report("cross_validation.json")

def get_benchmark():
    return load_report("benchmark_results.json")

def get_temporal_validation():
    return load_report("temporal_validation.json")

def get_ablation():
    return load_report("ablation_results.json")

def get_robustness():
    return load_report("robustness_results.json")

def get_leakage():
    return load_report("leakage_and_overfitting.json")

def get_split_info():
    return load_report("split_info.json")

def get_roc_curve():
    return load_report("roc_curve.json")

def get_pr_curve():
    return load_report("pr_curve.json")

def get_predictions():
    return load_report("predictions.json")
