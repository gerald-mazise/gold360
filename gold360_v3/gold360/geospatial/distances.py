from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class GeoDistances:
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * \
            np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    @staticmethod
    def compute_distance_matrix(lats: List[float], lons: List[float]) -> np.ndarray:
        n = len(lats)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = GeoDistances.haversine(lats[i], lons[i], lats[j], lons[j])
        return matrix

    @staticmethod
    def nearest_fgr(mine_lat: float, mine_lon: float,
                    fgr_df: pd.DataFrame) -> Tuple[str, float]:
        min_dist = float("inf")
        nearest = ""
        for _, row in fgr_df.iterrows():
            dist = GeoDistances.haversine(
                mine_lat, mine_lon, row["latitude"], row["longitude"]
            )
            if dist < min_dist:
                min_dist = dist
                nearest = row["office_name"]
        return nearest, min_dist

    @staticmethod
    def nearest_border_post(mine_lat: float, mine_lon: float) -> Tuple[str, float]:
        border_posts = {
            "Beitbridge": (-22.2167, 30.0000),
            "Plumtree": (-20.4833, 27.8167),
            "Mutare": (-18.9667, 32.6667),
            "Chirundu": (-16.0500, 28.8500),
            "Victoria Falls": (-17.9333, 25.8333),
            "Nyamapanda": (-17.0667, 32.8833),
            "Forbes": (-18.9667, 32.6667),
            "Kazungula": (-17.7833, 25.2667),
        }
        min_dist = float("inf")
        nearest = ""
        for name, (lat, lon) in border_posts.items():
            dist = GeoDistances.haversine(mine_lat, mine_lon, lat, lon)
            if dist < min_dist:
                min_dist = dist
                nearest = name
        return nearest, min_dist
