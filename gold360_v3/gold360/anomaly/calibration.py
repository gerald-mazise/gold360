from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._model = None

    def fit(self, anomaly_scores: np.ndarray, pseudo_labels: np.ndarray):
        scores = anomaly_scores.reshape(-1, 1)
        labels = (pseudo_labels > 0.5).astype(int)

        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(anomaly_scores.flatten(), labels)
        elif self.method == "logistic":
            self._model = LogisticRegression()
            self._model.fit(scores, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        logger.info(f"Calibrated using {self.method} regression")
        return self

    def transform(self, anomaly_scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator not fitted")
        if self.method == "isotonic":
            return self._model.transform(anomaly_scores.flatten())
        return self._model.predict_proba(anomaly_scores.reshape(-1, 1))[:, 1]
