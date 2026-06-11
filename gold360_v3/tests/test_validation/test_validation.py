import numpy as np
import pandas as pd
import pytest

from gold360.validation.ablation import AblationStudy
from gold360.validation.robustness import RobustnessTest
from gold360.validation.temporal_validation import TemporalValidator
from gold360.validation.metrics import IntelligenceMetrics


class TestAblationStudy:
    def test_run(self):
        study = AblationStudy()
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        groups = {"group_a": [0, 1, 2], "group_b": [3, 4, 5]}
        result = study.run(X, y, groups)
        assert len(result) > 0
        assert "roc_auc" in result.columns


class TestRobustnessTest:
    def test_evaluate(self):
        rt = RobustnessTest(n_trials=3, noise_levels=[0.01, 0.05])
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        result = rt.evaluate(X, y)
        assert len(result) == 2
        assert "noise_level" in result.columns
        assert "mean_auc" in result.columns


class TestTemporalValidator:
    def test_walk_forward(self):
        validator = TemporalValidator(n_splits=2, min_train_months=6)
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        df = pd.DataFrame({
            "month_year": dates,
            "f1": np.random.randn(24),
            "f2": np.random.randn(24),
            "pseudo_risk_probability": (np.random.rand(24) > 0.5).astype(float),
        })
        results = validator.walk_forward_evaluate(df)
        assert len(results) >= 1
        assert "roc_auc" in results[0]

    def test_temporal_stability(self):
        validator = TemporalValidator()
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        df = pd.DataFrame({
            "month_year": dates,
            "pseudo_risk_probability": np.random.rand(12),
        })
        result = validator.temporal_stability(df)
        assert len(result) == 12
        assert "mean_risk" in result.columns


class TestIntelligenceMetrics:
    def test_pseudo_label_confidence(self):
        scores = np.random.rand(100)
        result = IntelligenceMetrics.pseudo_label_confidence(scores)
        assert "mean_confidence" in result
        assert "high_conf_pct" in result

    def test_anomaly_agreement(self):
        scores = np.random.uniform(0, 100, 100)
        result = IntelligenceMetrics.anomaly_agreement(scores)
        assert "flagged_pct" in result
        assert "mean_score" in result

    def test_risk_distribution(self):
        scores = np.random.rand(100)
        result = IntelligenceMetrics.risk_distribution(scores)
        assert "low" in result
        assert "high" in result
        assert sum(result.values()) == 100
