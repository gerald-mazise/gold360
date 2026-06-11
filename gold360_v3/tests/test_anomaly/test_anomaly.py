import numpy as np
import pandas as pd
import pytest

from gold360.anomaly.isolation_forest import IsolationForestDetector
from gold360.anomaly.ecod import ECODDetector
from gold360.anomaly.lof import LOFDetector
from gold360.anomaly.ensemble import AnomalyEnsemble
from gold360.anomaly.calibration import Calibrator


class TestIsolationForest:
    def test_fit_predict(self):
        detector = IsolationForestDetector()
        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        df.iloc[:10] += 5
        detector.fit(df)
        result = detector.predict(df)
        assert "if_anomaly_score" in result.columns
        assert result["if_anomaly_score"].iloc[:10].mean() > result["if_anomaly_score"].iloc[10:].mean()


class TestECOD:
    def test_fit_predict(self):
        detector = ECODDetector()
        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        df.iloc[:10] += 5
        detector.fit(df)
        result = detector.predict(df)
        assert "ecod_anomaly_score" in result.columns


class TestLOF:
    def test_fit_predict(self):
        detector = LOFDetector()
        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        df.iloc[:10] += 5
        result = detector.fit_predict(df)
        assert "lof_anomaly_score" in result.columns


class TestAnomalyEnsemble:
    def test_initialization(self):
        ensemble = AnomalyEnsemble()
        assert ensemble is not None
        assert "isolation_forest" in ensemble.weights

    def test_fit_predict(self):
        ensemble = AnomalyEnsemble()
        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        df.iloc[:10] += 5
        ensemble.fit(df, df.columns.tolist())
        result = ensemble.predict(df)
        assert "anomaly_consensus_score" in result.columns
        assert "anomaly_agreement" in result.columns

    def test_agreement_levels(self):
        ensemble = AnomalyEnsemble()
        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        ensemble.fit(df, df.columns.tolist())
        result = ensemble.predict(df)
        assert result["anomaly_agreement"].between(0, 1).all()


class TestCalibrator:
    def test_isotonic(self):
        scores = np.random.uniform(0, 1, 200)
        y = (scores > 0.5).astype(int)
        calibrator = Calibrator(method="isotonic")
        calibrator.fit(scores[:100], y[:100])
        calibrated = calibrator.transform(scores[100:])
        assert all(0 <= c <= 1 for c in calibrated)
