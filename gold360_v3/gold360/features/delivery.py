import numpy as np
import pandas as pd

from gold360.features.registry import FeatureDefinition, FeatureRegistry
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class DeliveryFeatures:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._register()

    def _register(self):
        specs = [
            FeatureDefinition(
                name="delivery_gap_kg",
                group="delivery",
                description="Difference between estimated yield and formal delivery",
                formula="estimated_gold_yield_kg - official_delivery_kg",
                dtype="continuous", category="core",
                dependencies=["estimated_gold_yield_kg", "official_delivery_kg"],
                economic_rationale="Positive gap indicates production not reaching formal channels",
            ),
            FeatureDefinition(
                name="delivery_efficiency",
                group="delivery",
                description="Ratio of formal delivery to estimated yield",
                formula="official_delivery_kg / estimated_gold_yield_kg",
                dtype="ratio", category="core",
                dependencies=["official_delivery_kg", "estimated_gold_yield_kg"],
                economic_rationale="Efficiency of formal channel utilization",
            ),
            FeatureDefinition(
                name="delivery_gap_ratio",
                group="delivery",
                description="Delivery gap normalized by estimated yield",
                formula="delivery_gap_kg / estimated_gold_yield_kg",
                dtype="ratio", category="core",
                dependencies=["delivery_gap_kg", "estimated_gold_yield_kg"],
            ),
            FeatureDefinition(
                name="rolling_delivery_change_3m",
                group="delivery",
                description="3-month rolling change in delivery vs expected",
                formula="rolling_3m_delivery - rolling_3m_expected",
                dtype="continuous", category="trend",
                dependencies=["official_delivery_kg", "estimated_gold_yield_kg"],
            ),
            FeatureDefinition(
                name="delivery_volatility_6m",
                group="delivery",
                description="6-month rolling std of delivery gap",
                formula="std(delivery_gap_kg, 6m)",
                dtype="continuous", category="risk",
                dependencies=["delivery_gap_kg"],
            ),
        ]
        for spec in specs:
            self.registry.register(spec, getattr(self, f"_compute_{spec.name}"))

    def _compute_delivery_gap_kg(self, df: pd.DataFrame) -> pd.Series:
        return (df["estimated_gold_yield_kg"] - df["official_delivery_kg"]).clip(lower=0)

    def _compute_delivery_efficiency(self, df: pd.DataFrame) -> pd.Series:
        denom = df["estimated_gold_yield_kg"].replace(0, np.nan)
        return (df["official_delivery_kg"] / denom).clip(0, 2).fillna(0)

    def _compute_delivery_gap_ratio(self, df: pd.DataFrame) -> pd.Series:
        denom = df["estimated_gold_yield_kg"].replace(0, np.nan)
        gap = df["delivery_gap_kg"] if "delivery_gap_kg" in df.columns else self._compute_delivery_gap_kg(df)
        return (gap / denom).clip(0, 1).fillna(0)

    def _compute_rolling_delivery_change_3m(self, df: pd.DataFrame) -> pd.Series:
        if "month_year" in df.columns:
            sorted_df = df.sort_values("month_year")
            delivery_3m = sorted_df["official_delivery_kg"].rolling(3, min_periods=1).mean()
            expected_3m = sorted_df["estimated_gold_yield_kg"].rolling(3, min_periods=1).mean()
            return (delivery_3m - expected_3m).values
        return pd.Series(0.0, index=df.index)

    def _compute_delivery_volatility_6m(self, df: pd.DataFrame) -> pd.Series:
        gap = self._compute_delivery_gap_kg(df)
        if "month_year" in df.columns:
            sorted_gap = gap.loc[df.sort_values("month_year").index]
            return sorted_gap.rolling(6, min_periods=1).std().values
        return gap.rolling(6, min_periods=1).std()
