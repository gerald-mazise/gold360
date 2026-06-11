import numpy as np
import pandas as pd

from gold360.features.registry import FeatureDefinition, FeatureRegistry
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class TradeFeatures:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._register()

    def _register(self):
        specs = [
            FeatureDefinition(
                name="mirror_trade_asymmetry",
                group="trade",
                description="Absolute gap between partner imports and reported exports",
                formula="partner_imports_usd - reported_exports_usd",
                dtype="continuous", category="indicator",
                dependencies=["uae_imports_from_zimbabwe_usd_hs710812", "gold_export_value_usd"],
                economic_rationale="Persistent positive gap suggests unreported outflows",
            ),
            FeatureDefinition(
                name="export_discrepancy_ratio",
                group="trade",
                description="Asymmetry normalized by reported exports",
                formula="mirror_trade_asymmetry / reported_exports_usd",
                dtype="ratio", category="indicator",
                dependencies=["mirror_trade_asymmetry", "gold_export_value_usd"],
            ),
            FeatureDefinition(
                name="trade_gap_pressure",
                group="trade",
                description="Z-score of export discrepancy over available history",
                formula="zscore(export_discrepancy_ratio, 5y)",
                dtype="zscore", category="risk",
                dependencies=["export_discrepancy_ratio"],
            ),
        ]
        for spec in specs:
            self.registry.register(spec, getattr(self, f"_compute_{spec.name}"))

    def _compute_mirror_trade_asymmetry(self, df: pd.DataFrame) -> pd.Series:
        if "uae_imports_from_zimbabwe_usd_hs710812" not in df.columns:
            return pd.Series(0.0, index=df.index)
        partner = df["uae_imports_from_zimbabwe_usd_hs710812"].fillna(0)
        exports = df.get("gold_export_value_usd", pd.Series(0.0, index=df.index)).fillna(0)
        return (partner - exports).clip(lower=0)

    def _compute_export_discrepancy_ratio(self, df: pd.DataFrame) -> pd.Series:
        asymmetry = self._compute_mirror_trade_asymmetry(df)
        exports = df.get("gold_export_value_usd", pd.Series(1.0, index=df.index)).fillna(1e-6)
        return (asymmetry / exports.replace(0, 1e-6)).clip(0, 10)

    def _compute_trade_gap_pressure(self, df: pd.DataFrame) -> pd.Series:
        ratio = self._compute_export_discrepancy_ratio(df)
        mu = ratio.mean()
        sigma = ratio.std()
        if sigma > 0:
            return ((ratio - mu) / sigma).clip(-3, 3)
        return pd.Series(0.0, index=df.index)
