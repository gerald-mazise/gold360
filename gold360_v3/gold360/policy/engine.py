from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.policy.elasticities import PolicyElasticities
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    predicted_delivery_change_pct: float
    predicted_residual_direction: str
    predicted_anomaly_shift: float
    confidence: str
    historical_analogs: List[str] = field(default_factory=list)
    notes: str = ""


class ScenarioEngine:
    """Scenario Intelligence Engine — directional analysis only, no deterministic claims."""

    def __init__(self, elasticities: PolicyElasticities):
        self.elasticities = elasticities

    def simulate(self, scenario_name: str, parameter_changes: Dict[str, float],
                  df: pd.DataFrame) -> ScenarioResult:
        delivery_elast = self.elasticities.elasticities.get("delivery_response", 0)

        total_effect = 0.0
        for param, change_pct in parameter_changes.items():
            if param == "retention":
                total_effect += change_pct * 0.3
            elif param == "tax_rate":
                total_effect += change_pct * (-0.2)
            elif param == "fx_retention":
                total_effect += change_pct * 0.25
            elif param == "compliance":
                total_effect += change_pct * (-0.15)
            else:
                total_effect += change_pct * 0.1

        delivery_change = total_effect * (1 + delivery_elast)
        delivery_change = float(np.clip(delivery_change, -0.5, 0.5))

        residual_dir = "widening" if delivery_change < -0.02 else \
                       ("narrowing" if delivery_change > 0.02 else "stable")
        anomaly_shift = float(abs(delivery_change) * 100 * 0.6)

        conf = "high" if abs(delivery_change) > 0.2 else \
               ("medium" if abs(delivery_change) > 0.1 else "low")

        return ScenarioResult(
            scenario_name=scenario_name,
            description=f"Scenario: {', '.join(f'{k}={v:+.1f}' for k,v in parameter_changes.items())}",
            predicted_delivery_change_pct=delivery_change * 100,
            predicted_residual_direction=residual_dir,
            predicted_anomaly_shift=anomaly_shift,
            confidence=conf,
            notes="Directional estimate based on historical response patterns. "
                  "Not a deterministic forecast."
        )

    def compare_scenarios(self, scenarios: List[Tuple[str, Dict[str, float]]],
                           df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for name, params in scenarios:
            result = self.simulate(name, params, df)
            results.append({
                "scenario": result.scenario_name,
                "delivery_change_pct": result.predicted_delivery_change_pct,
                "residual_direction": result.predicted_residual_direction,
                "anomaly_shift": result.predicted_anomaly_shift,
                "confidence": result.confidence,
            })
        return pd.DataFrame(results)
