from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from gold360.models.trainer import ModelTrainer
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class AblationStudy:
    def __init__(self, config: Optional[Dict] = None):
        self.trainer = ModelTrainer(config or {})
        self.results: Dict[str, Dict] = {}

    def run(self, X: np.ndarray, y: np.ndarray,
             feature_groups: Dict[str, List[int]], baseline_name: str = "all") -> pd.DataFrame:
        baseline_metrics = self._evaluate(X, y)
        self.results[baseline_name] = {"features": X.shape[1], "metrics": baseline_metrics}

        for group_name, indices in feature_groups.items():
            keep_mask = np.ones(X.shape[1], dtype=bool)
            keep_mask[indices] = False
            X_ablated = X[:, keep_mask]
            metrics = self._evaluate(X_ablated, y)
            delta = {k: baseline_metrics[k] - metrics[k] for k in baseline_metrics}
            self.results[group_name] = {
                "features": X_ablated.shape[1], "metrics": metrics, "delta": delta,
            }
            logger.info(f"Ablation '{group_name}': AUC delta={delta.get('roc_auc', 0):.4f}")

        return self.summary()

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        from catboost import CatBoostClassifier
        from sklearn.model_selection import KFold, cross_val_score
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        clf = CatBoostClassifier(verbose=0, random_seed=42, iterations=200, depth=4)
        aucs = cross_val_score(clf, X, (y > 0.5).astype(int), cv=kf, scoring="roc_auc")
        return {"roc_auc": float(aucs.mean()), "roc_auc_std": float(aucs.std())}

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, data in self.results.items():
            row = {"group": name, "features": data["features"]}
            row.update(data["metrics"])
            if "delta" in data:
                row.update({f"delta_{k}": v for k, v in data["delta"].items()})
            rows.append(row)
        return pd.DataFrame(rows)
