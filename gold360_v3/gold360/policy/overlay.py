from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class PolicyOverlay:
    def __init__(self):
        self._overlays: List[Dict[str, Any]] = []

    def compute_overlay(self, df: pd.DataFrame, policy_col: str = "policy_shock_flag",
                         target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if policy_col not in df.columns:
            logger.warning(f"Policy column '{policy_col}' not found")
            return df

        result = df.copy()
        targets = target_cols or [c for c in df.columns if c != policy_col and df[c].dtype in (np.float64, np.int64)]

        for col in targets:
            if col not in df.columns:
                continue
            pre_mean = df[df[policy_col] == 0][col].mean()
            post_mean = df[df[policy_col] > 0][col].mean()
            if pd.notna(pre_mean) and pd.notna(post_mean) and pre_mean != 0:
                result[f"{col}_policy_effect"] = (post_mean - pre_mean) / pre_mean
            else:
                result[f"{col}_policy_effect"] = 0.0

        return result

    def get_policy_periods(self, df: pd.DataFrame, policy_col: str = "policy_shock_flag") -> List[Dict]:
        if policy_col not in df.columns:
            return []
        periods = []
        in_period = False
        start = None
        for i, row in df.iterrows():
            if row[policy_col] > 0 and not in_period:
                in_period = True
                start = row.get("month_year", i)
            elif row[policy_col] == 0 and in_period:
                in_period = False
                end = df.loc[df.index[df.index.get_loc(i) - 1], "month_year"] if i != df.index[0] else row.get("month_year", i)
                periods.append({"start": start, "end": end})
        if in_period:
            periods.append({"start": start, "end": df.iloc[-1].get("month_year", "present")})
        return periods
