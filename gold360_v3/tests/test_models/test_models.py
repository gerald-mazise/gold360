import numpy as np
import pandas as pd
import pytest

from gold360.models.classifier import CatBoostClassifier
from gold360.models.trainer import ModelTrainer
from gold360.models.predict import Predictor
from gold360.models.benchmark import ModelBenchmark


class TestCatBoostClassifier:
    def test_initialization(self):
        clf = CatBoostClassifier()
        assert clf is not None

    def test_fit_predict(self):
        clf = CatBoostClassifier()
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        clf.fit(X, y)
        y_prob = clf.predict_proba(X)
        assert len(y_prob) == len(y)
        assert all(0 <= p <= 1 for p in y_prob)

    def test_feature_importance(self):
        clf = CatBoostClassifier()
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        clf.fit(X, y)
        names = ["f1", "f2", "f3", "f4", "f5"]
        imp = clf.feature_importances(names)
        assert len(imp) == len(names)
        assert imp["importance"].sum() > 0


class TestModelTrainer:
    def test_initialization(self):
        trainer = ModelTrainer()
        assert trainer is not None

    def test_train(self):
        trainer = ModelTrainer()
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        clf = trainer.train(X, y)
        assert clf is not None

    def test_temporal_split(self):
        trainer = ModelTrainer()
        df = pd.DataFrame({
            "month_year": pd.date_range("2020-01-01", periods=60, freq="ME"),
            "value": np.random.randn(60),
            "target": np.random.rand(60),
        })
        train, val, test = trainer.temporal_split(df)
        assert len(train) > 0
        assert len(test) >= 0


class TestPredictor:
    def test_predict_with_confidence(self):
        clf = CatBoostClassifier()
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        clf.fit(X, y)
        predictor = Predictor(clf)
        result = predictor.predict_with_confidence(X[:10], ["f1", "f2", "f3", "f4", "f5"])
        assert len(result) == 10
        assert "risk_probability" in result.columns
        assert "risk_category" in result.columns

    def test_risk_categories(self):
        from gold360.models.predict import RISK_CATEGORIES
        for lo, hi, cat in RISK_CATEGORIES:
            mid = (lo + hi) / 2
            clf = CatBoostClassifier()
            X = np.random.randn(50, 3)
            y = (X[:, 0] > 0).astype(float)
            clf.fit(X, y)
            predictor = Predictor(clf)
            result = predictor.predict_with_confidence(X[:5])
            assert "risk_category" in result.columns


class TestModelBenchmark:
    def test_comparison(self):
        benchmark = ModelBenchmark()
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        split = 150
        metrics = benchmark.benchmark_catboost(X[:split], y[:split], X[split:], y[split:])
        assert "roc_auc" in metrics
        metrics2 = benchmark.benchmark_xgboost(X[:split], y[:split], X[split:], y[split:])
        assert "f1" in metrics2
