from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class CorridorAnalyzer:
    CORRIDOR_WEIGHTS = {
        "Beitbridge": 0.30,
        "Plumtree": 0.25,
        "Mutare": 0.20,
        "Chirundu": 0.15,
        "Victoria Falls": 0.10,
    }

    def __init__(self):
        self.corridor_scores: Dict[str, float] = {}

    def compute_corridor_risk(self, df: pd.DataFrame,
                               border_col: str = "nearest_border_post",
                               distance_col: str = "distance_to_border_km") -> pd.Series:
        if border_col not in df.columns:
            return pd.Series(0.0, index=df.index)

        weights = []
        for _, row in df.iterrows():
            corridor = str(row.get(border_col, ""))
            base_weight = 0.15
            for name, w in self.CORRIDOR_WEIGHTS.items():
                if name.lower() in corridor.lower():
                    base_weight = w
                    break
            dist = row.get(distance_col, 100)
            if pd.notna(dist) and dist < 50:
                dist_factor = 1.5
            elif pd.notna(dist) and dist < 100:
                dist_factor = 1.2
            else:
                dist_factor = 0.8
            weights.append(min(base_weight * dist_factor, 1.0))

        return pd.Series(weights, index=df.index) * 100

    def corridor_summary(self, df: pd.DataFrame,
                          border_col: str = "nearest_border_post") -> pd.DataFrame:
        if border_col not in df.columns:
            return pd.DataFrame()

        risk = self.compute_corridor_risk(df)
        summary = df.groupby(border_col).agg(
            count=("mine_id", "count"),
            mean_risk=("mine_id", lambda x: risk.loc[x.index].mean()),
        ).reset_index()
        summary.columns = ["corridor", "mine_count", "mean_risk_score"]
        return summary.sort_values("mean_risk_score", ascending=False)
