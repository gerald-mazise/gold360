from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class IsolationForestDetector:
    def __init__(self, n_estimators: int = 200, contamination: float = 0.05, random_state: int = 42):
        self.params = {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
            "bootstrap": False,
            "n_jobs": -1,
        }
        self.model: Optional[IsolationForest] = None
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "IsolationForestDetector":
        data, cols = self._prepare(df, feature_cols)
        self.feature_names = cols
        self.model = IsolationForest(**self.params)
        self.model.fit(data)
        logger.info(f"Trained IsolationForest on {len(cols)} features, {len(data)} samples")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        data, _ = self._prepare(df, self.feature_names)
        scores = self.model.decision_function(data)
        anomaly_labels = self.model.predict(data)
        result = df.copy()
        result["if_anomaly_score"] = -scores
        result["if_anomaly_label"] = (anomaly_labels == -1).astype(int)
        return result

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        data, _ = self._prepare(df, self.feature_names)
        scores = self.model.decision_function(data)
        return -scores

    def _prepare(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> tuple:
        if cols:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            data = df[cols].copy()
        else:
            data = df.select_dtypes(include=[np.number]).copy()
            cols = list(data.columns)
        data = data.fillna(data.median())
        return data.values, cols
