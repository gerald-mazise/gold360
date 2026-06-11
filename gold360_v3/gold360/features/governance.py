import numpy as np
import pandas as pd

from gold360.features.registry import FeatureDefinition, FeatureRegistry
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class GovernanceFeatures:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._register()

    def _register(self):
        specs = [
            FeatureDefinition(
                name="policy_shock_flag",
                group="governance",
                description="Binary indicator of active policy event in month",
                formula="any_active_policy_in_month",
                dtype="binary", category="event",
                dependencies=["policy_events"],
                economic_rationale="Policy changes alter formal delivery incentives",
            ),
            FeatureDefinition(
                name="policy_shock_decay",
                group="governance",
                description="Exponential decay of policy shock effect over time",
                formula="exp(-months_since_policy / 3)",
                dtype="decay", category="event",
                dependencies=["policy_events", "month_year"],
            ),
            FeatureDefinition(
                name="formal_market_friction",
                group="governance",
                description="Inverse of compliance score as friction index",
                formula="1 - (compliance_score / 100)",
                dtype="index", category="risk",
                dependencies=["compliance_score"],
                economic_rationale="Higher friction increases informal channel incentives",
            ),
            FeatureDefinition(
                name="compliance_pressure",
                group="governance",
                description="Z-score of compliance score relative to mine history",
                formula="zscore(compliance_score, mine_history)",
                dtype="zscore", category="risk",
                dependencies=["compliance_score", "mine_id"],
            ),
            FeatureDefinition(
                name="policy_regime_age_months",
                group="governance",
                description="Months since last major policy regime change",
                formula="months_since_last_policy_event",
                dtype="continuous", category="context",
                dependencies=["policy_events", "month_year"],
            ),
            FeatureDefinition(
                name="regulatory_pressure_index",
                group="governance",
                description="Composite of active policy pressures",
                formula="sum(policy_shock_decay * transmission_channel_weight)",
                dtype="index", category="composite",
                dependencies=["policy_shock_decay"],
            ),
        ]
        for spec in specs:
            self.registry.register(spec, getattr(self, f"_compute_{spec.name}"))

    def _compute_policy_shock_flag(self, df: pd.DataFrame) -> pd.Series:
        if "policy_shock_flag" in df.columns:
            return df["policy_shock_flag"].astype(float)
        return pd.Series(0.0, index=df.index)

    def _compute_policy_shock_decay(self, df: pd.DataFrame) -> pd.Series:
        flag = self._compute_policy_shock_flag(df)
        result = flag.copy().astype(float)
        if flag.sum() > 0:
            shock_indices = df[flag > 0].index
            for idx in shock_indices:
                pos = df.index.get_loc(idx)
                for offset in range(1, 13):
                    future_pos = pos + offset
                    if future_pos < len(df):
                        future_idx = df.index[future_pos]
                        result.loc[future_idx] = max(
                            result.get(future_idx, 0),
                            np.exp(-offset / 3.0)
                        )
        return result

    def _compute_formal_market_friction(self, df: pd.DataFrame) -> pd.Series:
        if "compliance_score" not in df.columns:
            return pd.Series(0.5, index=df.index)
        return (1 - df["compliance_score"] / 100.0).clip(0, 1)

    def _compute_compliance_pressure(self, df: pd.DataFrame) -> pd.Series:
        if "compliance_score" not in df.columns:
            return pd.Series(0.0, index=df.index)
        if "mine_id" in df.columns:
            result = pd.Series(0.0, index=df.index)
            for _, group in df.groupby("mine_id"):
                mu = group["compliance_score"].mean()
                sigma = group["compliance_score"].std()
                if sigma > 0:
                    result.loc[group.index] = (group["compliance_score"] - mu) / sigma
            return result.clip(-3, 3)
        mu = df["compliance_score"].mean()
        sigma = df["compliance_score"].std()
        if sigma > 0:
            return ((df["compliance_score"] - mu) / sigma).clip(-3, 3)
        return pd.Series(0.0, index=df.index)

    def _compute_policy_regime_age_months(self, df: pd.DataFrame) -> pd.Series:
        flag = self._compute_policy_shock_flag(df)
        result = pd.Series(range(1, len(df) + 1), index=df.index, dtype=float)
        last_shock = -1
        for i, idx in enumerate(df.index):
            if flag.loc[idx] > 0:
                last_shock = i
            if last_shock >= 0:
                result.loc[idx] = float(i - last_shock)
        return result

    def _compute_regulatory_pressure_index(self, df: pd.DataFrame) -> pd.Series:
        decay = self._compute_policy_shock_decay(df)
        return (decay * 100).clip(0, 100)
