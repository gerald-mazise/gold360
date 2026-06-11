from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class SHAPExplainer:
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self._explainer: Optional[shap.TreeExplainer] = None
        self._global_shap: Optional[pd.DataFrame] = None

    def fit(self, X_background: np.ndarray):
        self._explainer = shap.TreeExplainer(self.model.model)
        self._global_shap = None
        logger.info(f"SHAP explainer initialized with {len(self.feature_names)} features")
        return self

    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        if self._explainer is None:
            self.fit(X[:100])
        shap_values = self._explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        result = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs)],
            "importance": mean_abs,
        }).sort_values("importance", ascending=False)
        self._global_shap = result
        logger.info("Computed global SHAP importance")
        return result

    def local_explanation(self, X: np.ndarray, idx: int = 0) -> pd.DataFrame:
        if self._explainer is None:
            raise RuntimeError("Explainer not fitted")
        shap_values = self._explainer.shap_values(X[idx:idx+1])
        values = shap_values[0]
        result = pd.DataFrame({
            "feature": self.feature_names[:len(values)],
            "shap_value": values,
            "directional_effect": ["positive" if v > 0 else "negative" for v in values],
            "abs_contribution": np.abs(values),
        }).sort_values("abs_contribution", ascending=False)
        return result

    def waterfall_data(self, X: np.ndarray, idx: int = 0) -> Dict:
        if self._explainer is None:
            raise RuntimeError("Explainer not fitted")
        shap_values = self._explainer.shap_values(X[idx:idx+1])
        expected = self._explainer.expected_value
        if isinstance(expected, np.ndarray):
            expected = float(expected[0]) if expected.shape else float(expected)
        values = shap_values[0]
        top_k = min(10, len(values))
        top_idx = np.argsort(np.abs(values))[-top_k:]

        return {
            "expected_value": float(expected),
            "output_value": float(expected + values.sum()),
            "features": [
                {
                    "name": self.feature_names[i],
                    "value": float(X[idx, i]),
                    "shap_value": float(values[i]),
                    "contribution": "increases" if values[i] > 0 else "decreases",
                }
                for i in reversed(top_idx)
            ],
        }
