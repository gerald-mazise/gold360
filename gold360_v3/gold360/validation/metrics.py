from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class IntelligenceMetrics:
    @staticmethod
    def pseudo_label_confidence(confidence_scores: np.ndarray) -> Dict[str, float]:
        return {
            "mean_confidence": float(np.mean(confidence_scores)),
            "median_confidence": float(np.median(confidence_scores)),
            "high_conf_pct": float((confidence_scores > 0.7).mean() * 100),
            "low_conf_pct": float((confidence_scores < 0.3).mean() * 100),
        }

    @staticmethod
    def anomaly_agreement(scores: np.ndarray, threshold: float = 50.0) -> Dict[str, float]:
        flagged = (scores > threshold).sum()
        return {
            "flagged_pct": float(flagged / len(scores) * 100),
            "max_score": float(scores.max()),
            "min_score": float(scores.min()),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        }

    @staticmethod
    def feature_stability(df: pd.DataFrame, feature_cols: List[str],
                           date_col: str = "month_year") -> pd.DataFrame:
        if date_col not in df.columns:
            return pd.DataFrame()
        numeric_features = [c for c in feature_cols if c in df.columns and df[c].dtype in (np.float64, np.int64)]
        return df.groupby(date_col)[numeric_features].mean().std().to_frame("temporal_stability").T

    @staticmethod
    def risk_distribution(scores: np.ndarray, bins: List[float] = None) -> Dict[str, float]:
        if bins is None:
            bins = [0.0, 0.25, 0.50, 0.75, 1.0]
        labels = ["low", "moderate", "elevated", "high"]
        counts, _ = np.histogram(np.clip(scores, 0, 1), bins=bins)
        return {labels[i]: int(counts[i]) for i in range(len(labels))}
