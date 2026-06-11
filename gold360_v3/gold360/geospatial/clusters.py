from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class SpatialCluster:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model: Optional[DBSCAN] = None
        self._scaler = StandardScaler()

    def fit_predict(self, df: pd.DataFrame, lat_col: str = "mine_latitude",
                     lon_col: str = "mine_longitude") -> pd.DataFrame:
        coords = df[[lat_col, lon_col]].values
        coords_scaled = self._scaler.fit_transform(coords)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(coords_scaled)

        result = df.copy()
        result["spatial_cluster"] = labels
        result["is_anomaly_cluster"] = (labels == -1).astype(int)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        return result

    def cluster_summary(self, df: pd.DataFrame, cluster_col: str = "spatial_cluster",
                          value_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if cluster_col not in df.columns:
            return pd.DataFrame()
        group_cols = [cluster_col]
        if value_cols:
            agg = {c: ["mean", "std", "count"] for c in value_cols if c in df.columns}
            return df.groupby(cluster_col).agg(agg).round(3)
        return df.groupby(cluster_col).size().reset_index(name="count")
