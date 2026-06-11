import numpy as np
import pandas as pd

from gold360.features.registry import FeatureDefinition, FeatureRegistry
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class MacroFeatures:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._register()

    def _register(self):
        specs = [
            FeatureDefinition(
                name="fx_spread_pct",
                group="macro",
                description="Parallel to official FX rate spread percentage",
                formula="((parallel_rate - official_rate) / official_rate) * 100",
                dtype="percentage", category="core",
                dependencies=["fx_market_rate"],
                economic_rationale="Wider spreads increase incentive for informal channel diversion",
            ),
            FeatureDefinition(
                name="inflation_pressure",
                group="macro",
                description="3-month rolling inflation change",
                formula="rolling_3m_inflation_change",
                dtype="continuous", category="core",
                dependencies=["inflation_rate"],
            ),
            FeatureDefinition(
                name="macro_instability_index",
                group="macro",
                description="Composite of FX spread, inflation pressure, and gold volatility",
                formula="composite(fx_spread_pct, inflation_pressure, gold_volatility)",
                dtype="index", category="composite",
                dependencies=["fx_spread_pct", "inflation_pressure", "gold_price_volatility"],
            ),
            FeatureDefinition(
                name="gold_price_momentum",
                group="macro",
                description="3-month percentage change in gold price",
                formula="pct_change(gold_price_usd, 3m)",
                dtype="percentage", category="trend",
                dependencies=["gold_price_usd"],
            ),
            FeatureDefinition(
                name="gold_price_volatility",
                group="macro",
                description="6-month rolling std of gold price",
                formula="rolling_std(gold_price_usd, 6m)",
                dtype="continuous", category="risk",
                dependencies=["gold_price_usd"],
            ),
            FeatureDefinition(
                name="formal_delivery_attractiveness",
                group="macro",
                description="Composite index of formal delivery incentive",
                formula="composite(gold_price_momentum, -fx_spread_pct, -inflation_pressure)",
                dtype="index", category="composite",
                dependencies=["gold_price_momentum", "fx_spread_pct", "inflation_pressure"],
            ),
        ]
        for spec in specs:
            self.registry.register(spec, getattr(self, f"_compute_{spec.name}"))

    def _compute_fx_spread_pct(self, df: pd.DataFrame) -> pd.Series:
        if "fx_market_rate" not in df.columns:
            return pd.Series(0.0, index=df.index)
        official = df.get("official_fx_rate", pd.Series(1.0, index=df.index))
        parallel = df["fx_market_rate"]
        return ((parallel - official) / official.replace(0, np.nan) * 100).fillna(0)

    def _compute_inflation_pressure(self, df: pd.DataFrame) -> pd.Series:
        if "inflation_rate" not in df.columns:
            return pd.Series(0.0, index=df.index)
        return df["inflation_rate"].rolling(3, min_periods=1).mean().diff().fillna(0)

    def _compute_macro_instability_index(self, df: pd.DataFrame) -> pd.Series:
        fx = self._compute_fx_spread_pct(df)
        inf = self._compute_inflation_pressure(df)
        gv = self._compute_gold_price_volatility(df)

        fx_norm = (fx - fx.min()) / (fx.max() - fx.min() + 1e-8)
        inf_norm = (inf - inf.min()) / (inf.max() - inf.min() + 1e-8)
        gv_norm = (gv - gv.min()) / (gv.max() - gv.min() + 1e-8)

        composite = (0.4 * fx_norm + 0.35 * inf_norm + 0.25 * gv_norm)
        return (composite * 100).clip(0, 100)

    def _compute_gold_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        if "gold_price_usd" not in df.columns:
            return pd.Series(0.0, index=df.index)
        return df["gold_price_usd"].pct_change(periods=3).fillna(0)

    def _compute_gold_price_volatility(self, df: pd.DataFrame) -> pd.Series:
        if "gold_price_usd" not in df.columns:
            return pd.Series(0.0, index=df.index)
        return df["gold_price_usd"].rolling(6, min_periods=1).std().fillna(0)

    def _compute_formal_delivery_attractiveness(self, df: pd.DataFrame) -> pd.Series:
        momentum = self._compute_gold_price_momentum(df)
        fx_spread = self._compute_fx_spread_pct(df)
        inflation = self._compute_inflation_pressure(df)

        m_norm = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-8)
        fx_norm = 1 - (fx_spread - fx_spread.min()) / (fx_spread.max() - fx_spread.min() + 1e-8)
        inf_norm = 1 - (inflation - inflation.min()) / (inflation.max() - inflation.min() + 1e-8)

        composite = (0.4 * m_norm + 0.3 * fx_norm + 0.3 * inf_norm)
        return (composite * 100).clip(0, 100)
