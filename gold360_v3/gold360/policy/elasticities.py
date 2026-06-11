from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class PolicyElasticities:
    def __init__(self):
        self.elasticities: Dict[str, float] = {}

    def estimate(self, df: pd.DataFrame, policy_col: str = "policy_shock_flag",
                 target_col: str = "official_delivery_kg") -> Dict[str, float]:
        if policy_col not in df.columns or target_col not in df.columns:
            logger.warning("Cannot estimate elasticities: columns missing")
            return {}

        pre = df[df[policy_col] == 0][target_col].mean()
        post = df[df[policy_col] > 0][target_col].mean()

        if pre > 0 and not np.isnan(post):
            pct_change = (post - pre) / pre
            self.elasticities["delivery_response"] = float(pct_change)
            logger.info(f"Estimated delivery response: {pct_change:+.4f}")
        else:
            self.elasticities["delivery_response"] = 0.0

        return self.elasticities

    def corridor_elasticity(self, df: pd.DataFrame, policy_col: str = "policy_shock_flag",
                             target_col: str = "delivery_gap_kg") -> float:
        if policy_col not in df.columns or target_col not in df.columns:
            return 0.0
        pre = df[df[policy_col] == 0][target_col].mean()
        post = df[df[policy_col] > 0][target_col].mean()
        if pre > 0 and not np.isnan(post):
            return float((post - pre) / pre)
        return 0.0

    def report(self) -> str:
        lines = ["POLICY ELASTICITY ESTIMATES", "=" * 60]
        for key, val in self.elasticities.items():
            lines.append(f"  {key}: {val:+.4f}")
        if not self.elasticities:
            lines.append("  No estimates available")
        return "\n".join(lines)
