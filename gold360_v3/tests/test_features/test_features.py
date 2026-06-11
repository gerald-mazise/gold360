import numpy as np
import pandas as pd
import pytest

from gold360.features.delivery import DeliveryFeatures
from gold360.features.macro import MacroFeatures
from gold360.features.operational import OperationalFeatures
from gold360.features.governance import GovernanceFeatures
from gold360.features.spatial import SpatialFeatures
from gold360.features.trade import TradeFeatures
from gold360.features.registry import FeatureRegistry
from gold360.features.store import FeatureStore


@pytest.fixture
def registry():
    return FeatureRegistry()


class TestDeliveryFeatures:
    def test_registration(self, registry):
        features = DeliveryFeatures(registry)
        assert len(features.registry.list_features(group="delivery")) > 0

    def test_feature_names(self, registry):
        features = DeliveryFeatures(registry)
        names = [f.name for f in features.registry.list_features(group="delivery")]
        assert "delivery_gap_kg" in names


class TestMacroFeatures:
    def test_registration(self, registry):
        features = MacroFeatures(registry)
        assert len(features.registry.list_features(group="macro")) > 0

    def test_feature_names(self, registry):
        features = MacroFeatures(registry)
        names = [f.name for f in features.registry.list_features(group="macro")]
        assert "fx_spread_pct" in names


class TestOperationalFeatures:
    def test_registration(self, registry):
        features = OperationalFeatures(registry)
        assert len(features.registry.list_features(group="operational")) > 0

    def test_feature_names(self, registry):
        features = OperationalFeatures(registry)
        names = [f.name for f in features.registry.list_features(group="operational")]
        assert "ore_grade_efficiency" in names


class TestGovernanceFeatures:
    def test_registration(self, registry):
        features = GovernanceFeatures(registry)
        assert len(features.registry.list_features(group="governance")) > 0

    def test_feature_names(self, registry):
        features = GovernanceFeatures(registry)
        names = [f.name for f in features.registry.list_features(group="governance")]
        assert "policy_shock_flag" in names


class TestSpatialFeatures:
    def test_registration(self, registry):
        features = SpatialFeatures(registry)
        assert len(features.registry.list_features(group="spatial")) > 0

    def test_feature_names(self, registry):
        features = SpatialFeatures(registry)
        names = [f.name for f in features.registry.list_features(group="spatial")]
        assert "border_risk" in names


class TestTradeFeatures:
    def test_registration(self, registry):
        features = TradeFeatures(registry)
        assert len(features.registry.list_features(group="trade")) > 0

    def test_feature_names(self, registry):
        features = TradeFeatures(registry)
        names = [f.name for f in features.registry.list_features(group="trade")]
        assert "mirror_trade_asymmetry" in names


class TestFeatureStore:
    def test_initialization(self):
        store = FeatureStore()
        assert store is not None

    def test_write_read(self, sample_mine_data):
        store = FeatureStore(version="test_v001")
        meta = {"rows": len(sample_mine_data), "cols": len(sample_mine_data.columns)}
        store.write_features("test_features", sample_mine_data, meta)
        loaded = store.read_features("test_features")
        assert loaded is not None
        assert len(loaded) == len(sample_mine_data)
