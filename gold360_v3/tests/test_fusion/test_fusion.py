import numpy as np
import pandas as pd
import pytest

from gold360.fusion.layer import FusionLayer
from gold360.fusion.tensor import IntelligenceTensor


class TestFusionLayer:
    def test_initialization(self):
        layer = FusionLayer()
        assert layer is not None

    def test_fuse(self):
        layer = FusionLayer()
        df = pd.DataFrame({
            "feature_1": np.random.randn(50),
            "feature_2": np.random.randn(50),
        })
        pseudo_labels = pd.DataFrame({"pseudo_risk_probability": np.random.rand(1),
                                       "pseudo_confidence": np.random.rand(1)})
        anomaly = pd.DataFrame({"anomaly_consensus_score": np.random.rand(50)})
        result = layer.fuse(df, pseudo_labels, anomaly)
        assert result is not None


class TestIntelligenceTensor:
    def test_initialization(self):
        df = pd.DataFrame({
            "month_year": pd.date_range("2020-01-01", periods=12, freq="ME"),
            "f1": np.random.randn(12),
            "f2": np.random.randn(12),
            "target": np.random.rand(12),
        })
        tensor = IntelligenceTensor(df)
        assert tensor is not None

    def test_set_features(self):
        df = pd.DataFrame({
            "month_year": pd.date_range("2020-01-01", periods=12, freq="ME"),
            "f1": np.random.randn(12),
            "f2": np.random.randn(12),
            "target": np.random.rand(12),
        })
        tensor = IntelligenceTensor(df)
        tensor.set_features(["f1", "f2"]).set_target("target")
        X = tensor.get_X()
        y = tensor.get_y()
        assert X.shape == (12, 2)
        assert y.shape == (12,)

    def test_no_future_leakage(self):
        df = pd.DataFrame({
            "month_year": pd.date_range("2020-01-01", periods=12, freq="ME"),
            "f1": np.random.randn(12),
            "f2": np.random.randn(12),
            "target": np.random.rand(12),
        })
        tensor = IntelligenceTensor(df)
        tensor.set_features(["f1", "f2"]).set_target("target")
        splits = tensor.extract_temporal_safe(train_end="2020-06-30", val_end="2020-09-30")
        assert splits["train"][0].shape[0] <= 6
