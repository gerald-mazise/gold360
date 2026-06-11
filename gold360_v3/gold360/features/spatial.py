import numpy as np
import pandas as pd

from gold360.features.registry import FeatureDefinition, FeatureRegistry
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class SpatialFeatures:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._register()

    def _register(self):
        specs = [
            FeatureDefinition(
                name="border_risk",
                group="spatial",
                description="Logistic-transformed border proximity risk",
                formula="sigmoid(-0.01 * (distance_to_border_km - 100))",
                dtype="index", category="risk",
                dependencies=["distance_to_border_km"],
                economic_rationale="Proximity to border increases diversion opportunity",
            ),
            FeatureDefinition(
                name="corridor_risk",
                group="spatial",
                description="Composite corridor risk score",
                formula="composite(border_risk, incident_density)",
                dtype="index", category="composite",
                dependencies=["border_risk", "nearest_border_post"],
            ),
            FeatureDefinition(
                name="fgr_access",
                group="spatial",
                description="Inverse-distance FGR office accessibility",
                formula="1 / (1 + distance_to_fidelity_km / 200)",
                dtype="index", category="access",
                dependencies=["distance_to_fidelity_km"],
                economic_rationale="Better FGR access reduces friction for formal sales",
            ),
            FeatureDefinition(
                name="province_risk_density",
                group="spatial",
                description="Rolling province-level anomaly rate",
                formula="rolling_province_anomaly_rate",
                dtype="continuous", category="risk",
                dependencies=["province", "anomaly_signal"],
            ),
            FeatureDefinition(
                name="distance_disparity",
                group="spatial",
                description="Ratio of border distance to FGR distance",
                formula="distance_to_border_km / (distance_to_fidelity_km + 1)",
                dtype="ratio", category="context",
                dependencies=["distance_to_border_km", "distance_to_fidelity_km"],
            ),
        ]
        for spec in specs:
            self.registry.register(spec, getattr(self, f"_compute_{spec.name}"))

    def _compute_border_risk(self, df: pd.DataFrame) -> pd.Series:
        if "distance_to_border_km" not in df.columns:
            return pd.Series(0.3, index=df.index)
        d = df["distance_to_border_km"]
        return 1.0 / (1.0 + np.exp(0.01 * (d - 100.0)))

    def _compute_corridor_risk(self, df: pd.DataFrame) -> pd.Series:
        border_risk = self._compute_border_risk(df)
        corridor_weights = {
            "Beitbridge": 0.30, "Plumtree": 0.25, "Mutare": 0.20,
            "Chirundu": 0.15, "Victoria Falls": 0.10
        }
        corridor_factor = pd.Series(0.0, index=df.index)
        if "nearest_border_post" in df.columns:
            for corridor, weight in corridor_weights.items():
                mask = df["nearest_border_post"].str.contains(corridor, case=False, na=False)
                corridor_factor.loc[mask] = weight
            corridor_factor = corridor_factor / max(corridor_weights.values())
        return ((0.6 * border_risk + 0.4 * corridor_factor) * 100).clip(0, 100)

    def _compute_fgr_access(self, df: pd.DataFrame) -> pd.Series:
        if "distance_to_fidelity_km" not in df.columns:
            return pd.Series(0.5, index=df.index)
        return (1.0 / (1.0 + df["distance_to_fidelity_km"] / 200.0))

    def _compute_province_risk_density(self, df: pd.DataFrame) -> pd.Series:
        if "province" not in df.columns:
            return pd.Series(0.0, index=df.index)
        if "anomaly_signal" not in df.columns:
            return pd.Series(0.0, index=df.index)
        province_rate = df.groupby("province")["anomaly_signal"].transform("mean")
        return province_rate.fillna(0) * 100

    def _compute_distance_disparity(self, df: pd.DataFrame) -> pd.Series:
        border = df.get("distance_to_border_km", pd.Series(100.0, index=df.index))
        fgr = df.get("distance_to_fidelity_km", pd.Series(100.0, index=df.index))
        return (border / (fgr + 1.0)).clip(0, 10)
