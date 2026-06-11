from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging
from gold360.weak_supervision.labeling_functions import (
    LabelingFunctionResult, LabelingFunctionRegistry,
)

logger = setup_logging(__name__)


class PseudoLabelFusion:
    def __init__(self, lf_registry: LabelingFunctionRegistry):
        self.registry = lf_registry

    def fuse(self, df: pd.DataFrame, method: str = "majority_vote") -> pd.DataFrame:
        results = self.registry.apply_all(df)
        if not results:
            logger.warning("No labeling function results to fuse")
            result_df = df.copy()
            result_df["pseudo_risk_probability"] = 0.0
            result_df["pseudo_confidence"] = 1.0
            result_df["num_active_lfs"] = 0
            return result_df

        strengths = np.array([r.signal_strength for r in results.values()])
        confidences = np.array([r.confidence for r in results.values()])

        if method == "max":
            prob = float(np.max(strengths))
        elif method == "mean":
            prob = float(np.mean(strengths))
        elif method == "weighted":
            weights = confidences / (confidences.sum() + 1e-8)
            prob = float(np.sum(strengths * weights))
        elif method == "majority_vote":
            above_threshold = (strengths > 0.5).sum()
            total = len(strengths)
            prob = float(above_threshold / max(total, 1))
        else:
            prob = float(np.median(strengths))

        confidence = float(np.mean(confidences))
        active_lfs = int((strengths > 0.1).sum())

        result_df = df.copy()
        result_df["pseudo_risk_probability"] = prob
        result_df["pseudo_confidence"] = confidence
        result_df["num_active_lfs"] = active_lfs

        for lf_name, result in results.items():
            result_df[f"lf_{lf_name}_signal"] = result.signal_strength
            result_df[f"lf_{lf_name}_confidence"] = result.confidence

        logger.info(
            f"Fused pseudo-label: prob={prob:.3f}, confidence={confidence:.3f}, "
            f"active_lfs={active_lfs}/{len(results)}"
        )
        return result_df
