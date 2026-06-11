import numpy as np
import pandas as pd

from gold360.features.registry import FeatureDefinition, FeatureRegistry
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class OperationalFeatures:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._register()

    def _register(self):
        specs = [
            FeatureDefinition(
                name="ore_grade_efficiency",
                group="operational",
                description="Gold yield per tonne of ore processed",
                formula="estimated_gold_yield_kg / ore_processed_tonnes",
                dtype="ratio", category="efficiency",
                dependencies=["estimated_gold_yield_kg", "ore_processed_tonnes"],
            ),
            FeatureDefinition(
                name="recovery_efficiency",
                group="operational",
                description="Normalized recovery rate",
                formula="recovery_rate_pct / 100",
                dtype="ratio", category="efficiency",
                dependencies=["recovery_rate_pct"],
            ),
            FeatureDefinition(
                name="inventory_carryover_ratio",
                group="operational",
                description="Cumulative inventory gap relative to yield",
                formula="cumulative(inventory_carryover_kg) / estimated_gold_yield_kg",
                dtype="ratio", category="risk",
                dependencies=["inventory_carryover_kg", "estimated_gold_yield_kg"],
            ),
            FeatureDefinition(
                name="rainfall_disruption",
                group="operational",
                description="Z-score of rainfall deviation from provincial monthly norm",
                formula="zscore(rainfall_mm, province_monthly)",
                dtype="zscore", category="disruption",
                dependencies=["rainfall_mm", "province"],
                economic_rationale="Heavy rain disrupts ASM open-cast operations",
            ),
            FeatureDefinition(
                name="energy_stress",
                group="operational",
                description="Energy availability stress index",
                formula="1 - (energy_consumption / max_energy)",
                dtype="index", category="disruption",
                dependencies=["energy_consumption"],
            ),
            FeatureDefinition(
                name="operational_anomaly_score",
                group="operational",
                description="Composite of operational inconsistencies",
                formula="composite(rainfall_disruption, energy_stress, inventory_carryover_ratio)",
                dtype="index", category="composite",
                dependencies=["rainfall_disruption", "energy_stress", "inventory_carryover_ratio"],
            ),
        ]
        for spec in specs:
            self.registry.register(spec, getattr(self, f"_compute_{spec.name}"))

    def _compute_ore_grade_efficiency(self, df: pd.DataFrame) -> pd.Series:
        denom = df["ore_processed_tonnes"].replace(0, np.nan)
        return (df["estimated_gold_yield_kg"] / denom).fillna(0)

    def _compute_recovery_efficiency(self, df: pd.DataFrame) -> pd.Series:
        return (df["recovery_rate_pct"] / 100.0).clip(0, 1)

    def _compute_inventory_carryover_ratio(self, df: pd.DataFrame) -> pd.Series:
        if "inventory_carryover_kg" not in df.columns:
            return pd.Series(0.0, index=df.index)
        cum_inventory = df.groupby("mine_id", sort=False)["inventory_carryover_kg"].cumsum() if "mine_id" in df.columns else df["inventory_carryover_kg"].cumsum()
        denom = df["estimated_gold_yield_kg"].replace(0, np.nan)
        return (cum_inventory / denom).fillna(0).clip(0, 5)

    def _compute_rainfall_disruption(self, df: pd.DataFrame) -> pd.Series:
        if "rainfall_mm" not in df.columns:
            return pd.Series(0.0, index=df.index)
        province_groups = df.groupby("province")["rainfall_mm"] if "province" in df.columns else [(None, df["rainfall_mm"])]
        result = pd.Series(0.0, index=df.index)
        if "province" in df.columns:
            for prov, group in province_groups:
                idx = group.index
                mu = group.mean()
                sigma = group.std()
                if sigma > 0:
                    result.loc[idx] = (group - mu) / sigma
        else:
            mu = df["rainfall_mm"].mean()
            sigma = df["rainfall_mm"].std()
            if sigma > 0:
                result = (df["rainfall_mm"] - mu) / sigma
        return result.clip(-3, 3)

    def _compute_energy_stress(self, df: pd.DataFrame) -> pd.Series:
        if "energy_consumption" not in df.columns:
            return pd.Series(0.5, index=df.index)
        max_e = df["energy_consumption"].max()
        if max_e > 0:
            return (1 - df["energy_consumption"] / max_e).clip(0, 1)
        return pd.Series(0.5, index=df.index)

    def _compute_operational_anomaly_score(self, df: pd.DataFrame) -> pd.Series:
        rd = self._compute_rainfall_disruption(df)
        es = self._compute_energy_stress(df)
        ic = self._compute_inventory_carryover_ratio(df)

        rd_norm = (rd.abs() - rd.abs().min()) / (rd.abs().max() - rd.abs().min() + 1e-8)
        es_norm = (es - es.min()) / (es.max() - es.min() + 1e-8)
        ic_norm = (ic - ic.min()) / (ic.max() - ic.min() + 1e-8)

        return ((0.3 * rd_norm + 0.3 * es_norm + 0.4 * ic_norm) * 100).fillna(0)
