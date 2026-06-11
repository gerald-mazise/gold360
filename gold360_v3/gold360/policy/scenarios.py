from typing import Any, Dict, List, Tuple

import pandas as pd

from gold360.policy.engine import ScenarioEngine, ScenarioResult
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class ScenarioDefinitions:
    DEFAULT_SCENARIOS: Dict[str, Dict[str, float]] = {
        "fx_retention_reduction": {
            "retention": -0.10,
            "fx_retention": -0.10,
        },
        "tax_increase": {
            "tax_rate": 0.05,
        },
        "compliance_tightening": {
            "compliance": 0.15,
        },
        "fx_retention_increase": {
            "retention": 0.10,
            "fx_retention": 0.10,
        },
        "combined_tightening": {
            "retention": -0.05,
            "tax_rate": 0.03,
            "compliance": 0.10,
        },
    }

    def __init__(self, engine: ScenarioEngine):
        self.engine = engine

    def run_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for name, params in self.DEFAULT_SCENARIOS.items():
            result = self.engine.simulate(name, params, df)
            results.append({
                "scenario": result.scenario_name,
                "delivery_change_pct": result.predicted_delivery_change_pct,
                "residual_direction": result.predicted_residual_direction,
                "anomaly_shift": result.predicted_anomaly_shift,
                "confidence": result.confidence,
                "notes": result.notes,
            })
        logger.info(f"Ran {len(results)} default scenarios")
        return pd.DataFrame(results)
