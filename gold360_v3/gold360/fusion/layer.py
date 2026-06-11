from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class FusionLayer:
    """Combines engineered features, pseudo-labels, anomaly scores, and policy signals."""

    def __init__(self, master_key: str = "month_year"):
        self.master_key = master_key

    def fuse(self, feature_df: pd.DataFrame,
             pseudo_label_df: Optional[pd.DataFrame] = None,
             anomaly_df: Optional[pd.DataFrame] = None,
             drop_na: bool = True) -> pd.DataFrame:

        result = feature_df.copy()

        if pseudo_label_df is not None:
            pl_cols = [c for c in pseudo_label_df.columns if c.startswith("pseudo_") or c.startswith("lf_")]
            for col in pl_cols:
                if col in pseudo_label_df.columns:
                    result[col] = pseudo_label_df[col].iloc[0] if len(pseudo_label_df) == 1 else pseudo_label_df[col].values[0] if len(col) > 0 else 0

            if "pseudo_risk_probability" in pseudo_label_df.columns:
                result["pseudo_risk_probability"] = float(pseudo_label_df["pseudo_risk_probability"].iloc[0]) if len(pseudo_label_df) == 1 else pseudo_label_df["pseudo_risk_probability"].values[0]

        if anomaly_df is not None:
            anomaly_cols = ["anomaly_consensus_score", "anomaly_agreement",
                            "anomaly_confidence", "if_score_normalized",
                            "ecod_score_normalized", "lof_score_normalized"]
            for col in anomaly_cols:
                if col in anomaly_df.columns:
                    result[col] = anomaly_df[col].values

        if drop_na:
            initial = len(result)
            result = result.dropna(axis=1, how="all")
            dropped = initial - len(result.columns)
            if dropped > 0:
                logger.warning(f"Dropped {dropped} all-null columns during fusion")

        logger.info(f"Fusion complete: {len(result)} rows, {len(result.columns)} columns")
        return result


class IntelligenceTensor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._feature_cols: List[str] = []
        self._target_col: Optional[str] = None

    def set_features(self, columns: Optional[List[str]] = None):
        if columns:
            self._feature_cols = [c for c in columns if c in self.df.columns]
        else:
            self._feature_cols = [
                c for c in self.df.columns
                if c not in ("month_year", "mine_id", self._target_col)
                and self.df[c].dtype in (np.float64, np.int64, np.float32, np.int32, bool)
            ]
        logger.info(f"Intelligence tensor: {len(self._feature_cols)} features")
        return self

    def set_target(self, column: str):
        if column not in self.df.columns:
            raise ValueError(f"Target column '{column}' not found")
        self._target_col = column
        return self

    def get_X(self) -> np.ndarray:
        return self.df[self._feature_cols].fillna(0).values

    def get_y(self) -> np.ndarray:
        if self._target_col is None:
            raise RuntimeError("Target column not set")
        return self.df[self._target_col].values

    def get_feature_names(self) -> List[str]:
        return self._feature_cols

    def get_metadata(self) -> pd.DataFrame:
        meta_cols = [c for c in ["month_year", "mine_id", "province", "mine_name"]
                     if c in self.df.columns]
        return self.df[meta_cols].copy() if meta_cols else pd.DataFrame()
