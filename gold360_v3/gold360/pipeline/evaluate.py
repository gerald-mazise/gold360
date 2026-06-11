from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from gold360.models.classifier import CatBoostClassifier
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class EvaluationPipeline:
    def __init__(self, model: CatBoostClassifier):
        self.model = model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        y_prob = self.model.predict_proba(X_test)
        y_pred = (y_prob > 0.5).astype(int)
        y_bin = (y_test > 0.5).astype(int)

        results = {
            "roc_auc": float(roc_auc_score(y_bin, y_prob)),
            "brier_score": float(brier_score_loss(y_bin, y_prob)),
            "classification_report": classification_report(y_bin, y_pred, output_dict=True),
        }

        precision, recall, _ = precision_recall_curve(y_bin, y_prob)
        results["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}

        logger.info(
            f"Evaluation: AUC={results['roc_auc']:.4f}, "
            f"Brier={results['brier_score']:.4f}"
        )
        return results

    def feature_importance_report(self, feature_names: List[str]) -> pd.DataFrame:
        return self.model.feature_importances(feature_names)
