from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.anomaly.isolation_forest import IsolationForestDetector
from gold360.anomaly.ecod import ECODDetector
from gold360.anomaly.lof import LOFDetector
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class AnomalyEnsemble:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "isolation_forest": 0.40,
            "ecod": 0.30,
            "lof": 0.30,
        }
        self.if_detector = IsolationForestDetector()
        self.ecod_detector = ECODDetector()
        self.lof_detector = LOFDetector()
        self._fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
        self.if_detector.fit(df, feature_cols)
        self.ecod_detector.fit(df, feature_cols)
        self._feature_cols = feature_cols
        self._fitted = True
        logger.info("Anomaly ensemble fitted: IF + ECOD + LOF")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        logger.info("Computing ensemble anomaly scores...")

        if_scores = self._normalize(self.if_detector.score(df))
        ecod_scores = self._normalize(self.ecod_detector.score(df))

        lof_result = self.lof_detector.fit_predict(df, self._feature_cols)
        lof_scores = self._normalize(lof_result["lof_anomaly_score"].values)

        consensus = (
            self.weights["isolation_forest"] * if_scores +
            self.weights["ecod"] * ecod_scores +
            self.weights["lof"] * lof_scores
        )

        agreement = np.column_stack([
            if_scores > 50,
            ecod_scores > 50,
            lof_scores > 50,
        ]).mean(axis=1)

        result = df.copy()
        result["if_score_normalized"] = if_scores
        result["ecod_score_normalized"] = ecod_scores
        result["lof_score_normalized"] = lof_scores
        result["anomaly_consensus_score"] = consensus
        result["anomaly_agreement"] = agreement
        result["anomaly_confidence"] = np.clip(agreement * 2, 0, 1)

        flagged = (consensus > 50).sum()
        logger.info(
            f"Ensemble complete: {flagged}/{len(result)} flagged "
            f"(mean consensus={consensus.mean():.1f})"
        )
        return result

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        s_min, s_max = float(scores.min()), float(scores.max())
        if s_max - s_min < 1e-8:
            return np.ones_like(scores) * 50.0
        return (scores - s_min) / (s_max - s_min) * 100.0
