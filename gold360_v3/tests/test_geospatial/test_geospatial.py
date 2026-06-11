import numpy as np
import pandas as pd
import pytest

from gold360.geospatial.distances import GeoDistances
from gold360.geospatial.corridors import CorridorAnalyzer
from gold360.geospatial.clusters import SpatialCluster


class TestGeoDistances:
    def test_haversine(self):
        dist = GeoDistances.haversine(-17.8252, 31.0335, -20.1500, 28.5833)
        assert dist > 300
        assert dist < 500

    def test_distance_matrix(self):
        lats = [-17.8, -20.1, -19.0]
        lons = [31.0, 28.6, 29.8]
        matrix = GeoDistances.compute_distance_matrix(lats, lons)
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0
        assert matrix[0, 1] > 0

    def test_nearest_fgr(self, sample_fgr_data):
        nearest, dist = GeoDistances.nearest_fgr(-17.8252, 31.0335, sample_fgr_data)
        assert nearest == "Harare"
        assert dist < 1

    def test_nearest_border_post(self):
        nearest, dist = GeoDistances.nearest_border_post(-20.2, 28.5)
        assert nearest in ["Beitbridge", "Plumtree"]
        assert dist > 0


class TestCorridorAnalyzer:
    def test_compute_risk(self):
        analyzer = CorridorAnalyzer()
        df = pd.DataFrame({
            "mine_id": ["M1", "M2", "M3"],
            "nearest_border_post": ["Beitbridge", "Mutare", "Victoria Falls"],
            "distance_to_border_km": [30.0, 80.0, 15.0],
        })
        risk = analyzer.compute_corridor_risk(df)
        assert len(risk) == 3

    def test_summary(self):
        analyzer = CorridorAnalyzer()
        df = pd.DataFrame({
            "mine_id": ["M1", "M2", "M3", "M4"],
            "nearest_border_post": ["Beitbridge", "Beitbridge", "Mutare", "Plumtree"],
        })
        summary = analyzer.corridor_summary(df)
        assert len(summary) > 0
        assert "corridor" in summary.columns


class TestSpatialCluster:
    def test_fit_predict(self):
        cluster = SpatialCluster(eps=0.5, min_samples=2)
        df = pd.DataFrame({
            "mine_latitude": np.random.uniform(-22, -16, 50),
            "mine_longitude": np.random.uniform(25, 33, 50),
        })
        result = cluster.fit_predict(df)
        assert "spatial_cluster" in result.columns
        assert "is_anomaly_cluster" in result.columns

    def test_cluster_summary(self):
        cluster = SpatialCluster()
        df = pd.DataFrame({
            "mine_latitude": np.random.uniform(-22, -16, 30),
            "mine_longitude": np.random.uniform(25, 33, 30),
        })
        result = cluster.fit_predict(df)
        summary = cluster.cluster_summary(result)
        assert len(summary) > 0
