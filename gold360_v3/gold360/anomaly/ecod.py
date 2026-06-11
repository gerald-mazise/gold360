from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class ECODDetector:
    """Empirical Cumulative Distribution-based outlier detection.

    Implements an efficient ECOD approximation using scipy stats.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self._means: Dict[int, float] = {}
        self._stds: Dict[int, float] = {}
        self._tails: Dict[int, str] = {}
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "ECODDetector":
        data, cols = self._prepare(df, feature_cols)
        self.feature_names = cols
        n_features = data.shape[1]
        for i in range(n_features):
            col_data = data[:, i]
            self._means[i] = float(np.mean(col_data))
            self._stds[i] = float(np.std(col_data)) + 1e-8
            skew = float(np.mean(((col_data - self._means[i]) / self._stds[i]) ** 3))
            self._tails[i] = "right" if skew > 0.5 else ("left" if skew < -0.5 else "both")
        logger.info(f"Fitted ECOD on {n_features} features")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = self.score(df)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        result = df.copy()
        result["ecod_anomaly_score"] = scores
        result["ecod_anomaly_label"] = (scores > threshold).astype(int)
        return result

    def score(self, df: pd.DataFrame) -> np.ndarray:
        data, _ = self._prepare(df, self.feature_names)
        n_samples, n_features = data.shape
        scores = np.zeros(n_samples)

        for i in range(n_features):
            col_data = data[:, i]
            mu, sigma = self._means[i], self._stds[i]
            z = (col_data - mu) / sigma

            tail = self._tails[i]
            if tail in ("right", "both"):
                p_tail = 1.0 - _approx_cdf(z)
                scores += -np.log(np.clip(p_tail, 1e-10, 1.0))
            if tail in ("left", "both"):
                p_tail = _approx_cdf(z)
                scores += -np.log(np.clip(p_tail, 1e-10, 1.0))

        return scores / n_features

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


def _approx_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    return norm.cdf(x)
