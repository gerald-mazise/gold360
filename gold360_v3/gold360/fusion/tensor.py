from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class IntelligenceTensor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._feature_cols: List[str] = []
        self._target_col: Optional[str] = None
        self._date_col: Optional[str] = None
        self._validate_date_column()

    def _validate_date_column(self):
        for candidate in ["month_year", "date", "period", "quarter", "year"]:
            if candidate in self.df.columns:
                self._date_col = candidate
                break

    def set_features(self, feature_cols: List[str]) -> "IntelligenceTensor":
        missing = [c for c in feature_cols if c not in self.df.columns]
        if missing:
            logger.warning(f"Missing feature columns: {missing}")
        self._feature_cols = [c for c in feature_cols if c in self.df.columns]
        return self

    def set_target(self, target_col: str) -> "IntelligenceTensor":
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame")
        self._target_col = target_col
        return self

    def get_feature_names(self) -> List[str]:
        return self._feature_cols.copy()

    def get_X(self) -> np.ndarray:
        if not self._feature_cols:
            raise ValueError("No feature columns set. Call set_features() first.")
        return self.df[self._feature_cols].fillna(0).values

    def get_y(self) -> np.ndarray:
        if self._target_col is None:
            raise ValueError("No target column set. Call set_target() first.")
        return self.df[self._target_col].values

    def extract_temporal_safe(self, train_end: str = "2023-12-31",
                               val_end: str = "2024-06-30") -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        if self._date_col is None:
            logger.warning("No date column found; returning full dataset")
            return {
                "train": (self.get_X(), self.get_y()),
                "val": (np.array([]), np.array([])),
                "test": (np.array([]), np.array([])),
            }

        train_mask = self.df[self._date_col] <= pd.Timestamp(train_end)
        val_mask = (self.df[self._date_col] > pd.Timestamp(train_end)) & \
                   (self.df[self._date_col] <= pd.Timestamp(val_end))
        test_mask = self.df[self._date_col] > pd.Timestamp(val_end)

        return {
            "train": (
                self.df.loc[train_mask, self._feature_cols].fillna(0).values,
                self.df.loc[train_mask, self._target_col].values,
            ),
            "val": (
                self.df.loc[val_mask, self._feature_cols].fillna(0).values,
                self.df.loc[val_mask, self._target_col].values,
            ),
            "test": (
                self.df.loc[test_mask, self._feature_cols].fillna(0).values,
                self.df.loc[test_mask, self._target_col].values,
            ),
        }
