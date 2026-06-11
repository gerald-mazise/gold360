from typing import Any, Dict, List, Optional

import pandas as pd

from gold360.policy.elasticities import PolicyElasticities
from gold360.policy.engine import ScenarioEngine, ScenarioResult
from gold360.policy.scenarios import ScenarioDefinitions
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class SimulationPipeline:
    def __init__(self, elasticities: PolicyElasticities):
        self.engine = ScenarioEngine(elasticities)
        self.scenarios = ScenarioDefinitions(self.engine)

    def run_default(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.scenarios.run_defaults(df)
        logger.info(f"Simulation pipeline: {len(results)} scenarios")
        return results

    def run_custom(self, df: pd.DataFrame, custom_scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        results = []
        for name, params in custom_scenarios.items():
            result = self.engine.simulate(name, params, df)
            results.append({
                "scenario": result.scenario_name,
                "delivery_change_pct": result.predicted_delivery_change_pct,
                "residual_direction": result.predicted_residual_direction,
                "anomaly_shift": result.predicted_anomaly_shift,
                "confidence": result.confidence,
                "notes": result.notes,
            })
        return pd.DataFrame(results)
