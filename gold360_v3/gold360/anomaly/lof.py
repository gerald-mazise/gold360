from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class LOFDetector:
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.params = {
            "n_neighbors": n_neighbors,
            "contamination": contamination,
            "n_jobs": -1,
        }
        self.model: Optional[LocalOutlierFactor] = None
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "LOFDetector":
        data, cols = self._prepare(df, feature_cols)
        self.feature_names = cols
        self.model = LocalOutlierFactor(**self.params, novelty=False)
        self.model.fit(data)
        logger.info(f"Fitted LOF on {len(cols)} features, {len(data)} samples")
        return self

    def fit_predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        data, cols = self._prepare(df, feature_cols)
        self.feature_names = cols
        self.model = LocalOutlierFactor(**self.params, novelty=False)
        labels = self.model.fit_predict(data)
        scores = -self.model.negative_outlier_factor_

        result = df.copy()
        result["lof_anomaly_score"] = scores
        result["lof_anomaly_label"] = (labels == -1).astype(int)
        logger.info(f"LOF fit_predict complete: {len(data)} samples")
        return result

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        data, _ = self._prepare(df, self.feature_names)
        scores = -self.model.score_samples(data)
        return scores

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
